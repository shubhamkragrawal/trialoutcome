"""Pharma-domain M2 training runner: wires PharmaDatasetBuilder features into
core.training_pipeline's model-agnostic Optuna+MLflow harness. Every model
family, hyperparameter search space, and column list lives here -- core stays
domain-agnostic. Run as `python -m domains.pharma.train_pipeline`.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from core.dataset_builder_base import SplitDates
from core.training_pipeline import OptunaMLflowTrainer, SplitData
from domains.pharma.dataset_builder import PharmaDatasetBuilder, feature_pipeline_version

PACKAGE_DIR = Path(__file__).parent
REPO_ROOT = PACKAGE_DIR.parent.parent

CATEGORICAL_FEATURES = ["phase", "allocation", "masking", "has_dmc_str", "sponsor_class"]
NUMERIC_FEATURES = [
    # M9-1: log_enrollment_count / enrollment_missing removed as target
    # leakage. See config.yaml dropped_features and decisions.md M9-1.
    "num_primary_outcomes",
    "num_sites",
    "has_results",
    "eligibility_criteria_length",
    "exclusion_keyword_count",
    "sponsor_prior_trial_count",
    "sponsor_prior_termination_rate",
    "condition_rarity",
    "start_year",
    "start_quarter",
]
N_TRIALS_PER_FAMILY = 8
ABLATION_TEST_WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"))
CONDITION_TOP_N = 20


def _fit_condition_vocab(train_df: pd.DataFrame, top_n: int = CONDITION_TOP_N) -> list[str]:
    """
    Purpose: Determine the top-N most frequent condition_name values on a
        given training set. Unlike notebooks/02_leakage_demo.ipynb (which
        deliberately fits a separate vocabulary per split to isolate a
        vocabulary-leakage effect), M2's Optuna sweep needs ONE fixed feature
        space across every CV fold and both split regimes so a single cloned
        Pipeline template can be reused throughout core.training_pipeline
        without a column mismatch.
    Leakage guard: Fit on temporal train only -- never on calib/test or the
        random split -- so the vocabulary itself never depends on which
        conditions become common in evaluation-only data.
    Failure mode: If this is fit on anything other than temporal train, a
        condition that only becomes common in a later period could get its
        own column -- information a production model couldn't have had.
    """
    return train_df["condition_name"].value_counts().head(top_n).index.tolist()


def _apply_condition_one_hot(
    df: pd.DataFrame, top_conditions: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Purpose: Apply a fixed condition-vocabulary one-hot encoding to any
        DataFrame (train or test, any regime), so every regime shares the
        exact same feature columns.
    Leakage guard: N/A here -- the vocabulary itself was already fit
        leakage-safely by _fit_condition_vocab; this function only applies it.
    Failure mode: If top_conditions doesn't match what a model was actually
        trained on, predict_proba raises a column-mismatch error rather than
        silently scoring on the wrong feature space.
    """
    bucketed = df["condition_name"].where(
        df["condition_name"].isin(top_conditions) | (df["condition_name"] == "unknown"),
        other="other",
    )
    dummies = pd.get_dummies(bucketed, prefix="condition")
    all_cols = [f"condition_{c}" for c in top_conditions] + ["condition_other", "condition_unknown"]
    dummies = dummies.reindex(columns=all_cols, fill_value=False)
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return out, dummies.columns.tolist()


def _make_preprocessor(condition_cols: list[str]) -> ColumnTransformer:
    """
    Purpose: Shared sklearn ColumnTransformer for all four model families --
        one-hot the low-cardinality categoricals, passthrough the numeric and
        already-one-hot condition columns.
    Leakage guard: N/A -- categorical encoding happens after the leakage-safe
        feature engineering in dataset_builder.py.
    Failure mode: If a caller passes a DataFrame missing any of
        CATEGORICAL_FEATURES/NUMERIC_FEATURES/condition_cols, ColumnTransformer
        raises a KeyError rather than silently dropping the model's view of
        that feature.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES + condition_cols),
        ]
    )


def _build_logreg(trial: optuna.Trial, condition_cols: list[str]) -> tuple[Pipeline, dict]:
    params = {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=params["C"])
    return Pipeline([("pre", _make_preprocessor(condition_cols)), ("clf", clf)]), params


def _build_rf(trial: optuna.Trial, condition_cols: list[str]) -> tuple[Pipeline, dict]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)
    return Pipeline([("pre", _make_preprocessor(condition_cols)), ("clf", clf)]), params


def _build_xgboost(
    trial: optuna.Trial, condition_cols: list[str], scale_pos_weight: float
) -> tuple[Pipeline, dict]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("pre", _make_preprocessor(condition_cols)), ("clf", clf)]), params


def _build_lgbm(trial: optuna.Trial, condition_cols: list[str]) -> tuple[Pipeline, dict]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    clf = LGBMClassifier(
        is_unbalance=True,
        subsample_freq=1,  # LightGBM ignores `subsample` unless a bagging frequency is also set
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    return Pipeline([("pre", _make_preprocessor(condition_cols)), ("clf", clf)]), params


def run_controlled_ablation(
    builder: PharmaDatasetBuilder,
    feat: pd.DataFrame,
    model_type: str,
    best_params: dict,
    feature_pipeline_ver: str,
) -> None:
    """
    Purpose: M2 followup requested explicitly (decisions.md M1 finding) --
        re-run the same fixed-test-window leakage ablation from
        notebooks/02_leakage_demo.ipynb using the best Optuna-found
        hyperparameters for a tree-based model (XGBoost/LightGBM), which can
        exploit subtler cross-period effects a linear model cannot. Logs the
        result as its own MLflow run rather than only printing it.
    Leakage guard: Isolates the leakage mechanism directly (fixed test
        window, only varying whether training data may include trials that
        start after that window) -- see
        core.dataset_builder_base.TemporalDatasetBuilder.controlled_leakage_ablation.
    Failure mode: If this is skipped, the M1 negative result (no leakage
        effect for LogReg) could get misread as "leakage isn't a real risk
        here" for the whole project, when a higher-capacity model was never
        actually checked.
    """
    window_start, window_end = ABLATION_TEST_WINDOW
    fixed_test, honest_train, leaky_train = builder.controlled_leakage_ablation(
        feat, date_col="start_date", test_window_start=window_start, test_window_end=window_end
    )

    honest_top = _fit_condition_vocab(honest_train)
    honest_train_oh, honest_cond_cols = _apply_condition_one_hot(honest_train, honest_top)
    honest_test_oh, _ = _apply_condition_one_hot(fixed_test, honest_top)

    leaky_top = _fit_condition_vocab(leaky_train)
    leaky_train_oh, leaky_cond_cols = _apply_condition_one_hot(leaky_train, leaky_top)
    leaky_test_oh, _ = _apply_condition_one_hot(fixed_test, leaky_top)

    def build(cond_cols: list[str], scale_pos_weight: float) -> Pipeline:
        if model_type == "xgboost":
            clf = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
                **best_params,
            )
        else:
            clf = LGBMClassifier(
                is_unbalance=True,
                subsample_freq=1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                **best_params,
            )
        return Pipeline([("pre", _make_preprocessor(cond_cols)), ("clf", clf)])

    honest_cols_full = CATEGORICAL_FEATURES + NUMERIC_FEATURES + honest_cond_cols
    leaky_cols_full = CATEGORICAL_FEATURES + NUMERIC_FEATURES + leaky_cond_cols

    honest_spw = float(
        (honest_train_oh["label"] == 0).sum() / (honest_train_oh["label"] == 1).sum()
    )
    honest_pipe = build(honest_cond_cols, honest_spw)
    honest_pipe.fit(honest_train_oh[honest_cols_full], honest_train_oh["label"])
    honest_proba = honest_pipe.predict_proba(honest_test_oh[honest_cols_full])[:, 1]
    pr_honest = float(average_precision_score(honest_test_oh["label"], honest_proba))
    roc_honest = float(roc_auc_score(honest_test_oh["label"], honest_proba))

    leaky_spw = float((leaky_train_oh["label"] == 0).sum() / (leaky_train_oh["label"] == 1).sum())
    leaky_pipe = build(leaky_cond_cols, leaky_spw)
    leaky_pipe.fit(leaky_train_oh[leaky_cols_full], leaky_train_oh["label"])
    leaky_proba = leaky_pipe.predict_proba(leaky_test_oh[leaky_cols_full])[:, 1]
    pr_leaky = float(average_precision_score(leaky_test_oh["label"], leaky_proba))
    roc_leaky = float(roc_auc_score(leaky_test_oh["label"], leaky_proba))

    print(f"\n=== Controlled leakage ablation: {model_type} (best params: {best_params}) ===")
    print(f"Fixed test window: {window_start.date()} to {window_end.date()}, n={len(fixed_test)}")
    print(f"Honest train (n={len(honest_train)}): PR-AUC={pr_honest:.4f}, ROC-AUC={roc_honest:.4f}")
    print(f"Leaky train  (n={len(leaky_train)}): PR-AUC={pr_leaky:.4f}, ROC-AUC={roc_leaky:.4f}")
    print(
        f"Delta (leaky - honest): PR-AUC={pr_leaky - pr_honest:+.4f}, ROC-AUC={roc_leaky - roc_honest:+.4f}"
    )

    with mlflow.start_run(run_name=f"controlled_ablation_{model_type}_best"):
        mlflow.set_tag("run_type", "controlled_ablation")
        mlflow.set_tag("feature_pipeline_version", feature_pipeline_ver)
        mlflow.log_param("model_type", model_type)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("pr_auc_honest", pr_honest)
        mlflow.log_metric("roc_auc_honest", roc_honest)
        mlflow.log_metric("pr_auc_leaky", pr_leaky)
        mlflow.log_metric("roc_auc_leaky", roc_leaky)


def main() -> None:
    builder = PharmaDatasetBuilder()
    raw = builder.fetch_raw()
    feat = builder.build_features(raw)

    split_cfg = builder.config["split"]
    dates = SplitDates(
        train_end=pd.Timestamp(split_cfg["train_end"]),
        calib_end=pd.Timestamp(split_cfg["calib_end"]),
    )
    temporal = builder.temporal_split(feat, date_col="start_date", split_dates=dates)
    random_df = builder.random_split(feat, random_state=split_cfg["random_state"])

    top_conditions = _fit_condition_vocab(temporal[temporal["split"] == "train"])
    temporal, condition_cols = _apply_condition_one_hot(temporal, top_conditions)
    random_df, _ = _apply_condition_one_hot(random_df, top_conditions)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + condition_cols

    temporal_train = temporal[temporal["split"] == "train"]
    temporal_test = temporal[temporal["split"] == "test"]
    random_train = random_df[random_df["split"] == "train"]
    random_test = random_df[random_df["split"] == "test"]

    temporal_split_data = SplitData(
        X_train=temporal_train[feature_cols],
        y_train=temporal_train["label"],
        X_test=temporal_test[feature_cols],
        y_test=temporal_test["label"],
        dates_train=temporal_train["start_date"],
    )
    random_split_data = SplitData(
        X_train=random_train[feature_cols],
        y_train=random_train["label"],
        X_test=random_test[feature_cols],
        y_test=random_test["label"],
    )

    majority_baseline_pr_auc = float(temporal_split_data.y_test.mean())
    majority_baseline_pr_auc_random = float(random_split_data.y_test.mean())
    print(f"Majority-class baseline PR-AUC (temporal test): {majority_baseline_pr_auc:.4f}")
    print(f"Majority-class baseline PR-AUC (random test):   {majority_baseline_pr_auc_random:.4f}")

    scale_pos_weight = float(
        (temporal_split_data.y_train == 0).sum() / (temporal_split_data.y_train == 1).sum()
    )

    version = feature_pipeline_version()
    trainer = OptunaMLflowTrainer(
        experiment_name="trialoutcome_m2",
        tracking_uri=f"file:{REPO_ROOT / 'mlruns'}",
        feature_pipeline_version=version,
        majority_baseline_pr_auc=majority_baseline_pr_auc,
        majority_baseline_pr_auc_random=majority_baseline_pr_auc_random,
    )
    trainer.log_majority_baseline_run()

    families = {
        "logreg": lambda trial: _build_logreg(trial, condition_cols),
        "rf": lambda trial: _build_rf(trial, condition_cols),
        "xgboost": lambda trial: _build_xgboost(trial, condition_cols, scale_pos_weight),
        "lgbm": lambda trial: _build_lgbm(trial, condition_cols),
    }

    studies = {}
    for model_type, build_fn in families.items():
        print(f"\n=== Running Optuna study: {model_type} ({N_TRIALS_PER_FAMILY} trials) ===")
        study = trainer.run_family(
            model_type=model_type,
            build_pipeline=build_fn,
            temporal=temporal_split_data,
            random=random_split_data,
            n_trials=N_TRIALS_PER_FAMILY,
        )
        studies[model_type] = study
        print(f"{model_type} best val_pr_auc={study.best_value:.4f} params={study.best_params}")
        print(
            f"{model_type} best trial pr_auc_temporal={study.best_trial.user_attrs['pr_auc_temporal']:.4f}"
        )

    for model_type in ["xgboost", "lgbm"]:
        run_controlled_ablation(
            builder,
            feat,
            model_type,
            studies[model_type].best_params,
            version,
        )


if __name__ == "__main__":
    main()
