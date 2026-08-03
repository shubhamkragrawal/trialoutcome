"""Domain-agnostic Optuna + MLflow training scaffolding. Accepts any
sklearn-API Pipeline (steps named 'pre'/'clf' by convention) -- no pharma
column names or model-family choices belong in this file; those are wired up
by the caller (see domains/pharma/train_pipeline.py).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


@dataclass
class SplitData:
    """
    Purpose: Bundle the feature/label arrays for one split regime (temporal
        or random) so run_family() takes one small argument per regime
        instead of eight loose positional DataFrames.
    Leakage guard: N/A -- a value container.
    Failure mode: N/A.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    dates_train: pd.Series | None = None  # required for the temporal regime only


class OptunaMLflowTrainer:
    """
    Purpose: Model-agnostic Optuna study + MLflow logging harness. Each trial
        fits a caller-supplied Pipeline family via TimeSeriesSplit-CV'd PR-AUC
        on the temporal-train split (the Optuna objective), then separately
        fits the same hyperparameters on the full temporal train and full
        random train purely for cross-regime visibility, logging one MLflow
        run per trial with every field the locked API contract and M7's
        version-mismatch check depend on.
    Leakage guard: The Optuna objective is computed ONLY via TimeSeriesSplit
        on the temporal-train split -- hyperparameter selection never sees
        either test set or the random split. The random-split metrics logged
        alongside are for comparison/visibility only and never influence
        which hyperparameters Optuna picks.
    Failure mode: N/A (class-level).
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        feature_pipeline_version: str,
        majority_baseline_pr_auc: float,
        majority_baseline_pr_auc_random: float,
        n_cv_splits: int = 5,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.feature_pipeline_version = feature_pipeline_version
        self.majority_baseline_pr_auc = majority_baseline_pr_auc
        self.majority_baseline_pr_auc_random = majority_baseline_pr_auc_random
        self.n_cv_splits = n_cv_splits

    def log_majority_baseline_run(self) -> None:
        """
        Purpose: Log a standalone MLflow run named 'majority_class_baseline'
            so "beats baseline meaningfully" (M2 acceptance criteria) is a direct comparison
            in the MLflow UI, not a number that only lives in a docstring.
        Leakage guard: N/A.
        Failure mode: If this run is never logged, every model run's
            already-attached majority_baseline_pr_auc metric is the only
            record of the floor -- redundant but not wrong; this run exists
            for the side-by-side comparison view specifically.
        """
        with mlflow.start_run(run_name="majority_class_baseline"):
            mlflow.set_tag("feature_pipeline_version", self.feature_pipeline_version)
            mlflow.set_tag("run_type", "majority_baseline")
            mlflow.log_param("model_type", "majority_class")
            mlflow.log_metric("pr_auc_temporal", self.majority_baseline_pr_auc)
            mlflow.log_metric("pr_auc_random", self.majority_baseline_pr_auc_random)
            mlflow.log_metric("majority_baseline_pr_auc", self.majority_baseline_pr_auc)
            mlflow.log_metric(
                "majority_baseline_pr_auc_random", self.majority_baseline_pr_auc_random
            )

    def run_family(
        self,
        model_type: str,
        build_pipeline: Callable[[optuna.Trial], tuple[Pipeline, dict[str, Any]]],
        temporal: SplitData,
        random: SplitData,
        n_trials: int,
    ) -> optuna.Study:
        """
        Purpose: Run one Optuna study for a single model family. build_pipeline
            is called once per trial, returning an unfit Pipeline plus the
            dict of hyperparameters it sampled (for logging); this method
            handles CV scoring, the two visibility fits, and all MLflow
            logging.
        Leakage guard: See class docstring. Each of the 5 CV folds, the
            temporal-full fit, and the random-full fit uses sklearn.base.clone
            on the same unfit pipeline template so no fitted state leaks
            between folds/regimes.
        Failure mode: If a CV fold's validation slice ends up single-class
            (possible in principle with TimeSeriesSplit's smallest, earliest
            fold), average_precision_score raises -- not handled here, since
            the actual data profile (14-30% positive rate, thousands of rows
            per fold) makes this exceedingly unlikely; a real occurrence
            should surface as a crash, not a silently-skipped fold.
        """
        assert temporal.dates_train is not None, (
            "run_family's `temporal` argument must carry dates_train -- it's None only "
            "for the random regime, which has no CV ordering to sort by"
        )
        order = np.argsort(temporal.dates_train.to_numpy())
        X_train_sorted = temporal.X_train.iloc[order].reset_index(drop=True)
        y_train_sorted = temporal.y_train.iloc[order].reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            pipeline, params = build_pipeline(trial)

            cv_scores = []
            for fold_train_idx, fold_val_idx in tscv.split(X_train_sorted):
                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(
                    X_train_sorted.iloc[fold_train_idx], y_train_sorted.iloc[fold_train_idx]
                )
                fold_proba = fold_pipeline.predict_proba(X_train_sorted.iloc[fold_val_idx])[:, 1]
                cv_scores.append(
                    average_precision_score(y_train_sorted.iloc[fold_val_idx], fold_proba)
                )
            val_pr_auc = float(np.mean(cv_scores))

            temporal_pipeline = clone(pipeline)
            temporal_pipeline.fit(temporal.X_train, temporal.y_train)
            temporal_proba = temporal_pipeline.predict_proba(temporal.X_test)[:, 1]
            pr_auc_temporal = float(average_precision_score(temporal.y_test, temporal_proba))
            roc_auc_temporal = float(roc_auc_score(temporal.y_test, temporal_proba))

            random_pipeline = clone(pipeline)
            random_pipeline.fit(random.X_train, random.y_train)
            random_proba = random_pipeline.predict_proba(random.X_test)[:, 1]
            pr_auc_random = float(average_precision_score(random.y_test, random_proba))
            roc_auc_random = float(roc_auc_score(random.y_test, random_proba))

            with mlflow.start_run(run_name=f"{model_type}_trial_{trial.number}"):
                mlflow.set_tag("feature_pipeline_version", self.feature_pipeline_version)
                mlflow.set_tag("run_type", "optuna_trial")
                mlflow.log_param("model_type", model_type)
                for name, value in params.items():
                    mlflow.log_param(name, value)
                mlflow.log_metric("val_pr_auc", val_pr_auc)
                mlflow.log_metric("pr_auc_temporal", pr_auc_temporal)
                mlflow.log_metric("roc_auc_temporal", roc_auc_temporal)
                mlflow.log_metric("pr_auc_random", pr_auc_random)
                mlflow.log_metric("roc_auc_random", roc_auc_random)
                mlflow.log_metric("majority_baseline_pr_auc", self.majority_baseline_pr_auc)
                mlflow.log_metric(
                    "majority_baseline_pr_auc_random", self.majority_baseline_pr_auc_random
                )
                _log_importance_artifact(temporal_pipeline, model_type, trial.number)

            trial.set_user_attr("pr_auc_temporal", pr_auc_temporal)
            trial.set_user_attr("roc_auc_temporal", roc_auc_temporal)
            trial.set_user_attr("params", params)
            return val_pr_auc

        study = optuna.create_study(direction="maximize", study_name=model_type)
        study.optimize(objective, n_trials=n_trials)
        return study


def _log_importance_artifact(pipeline: Pipeline, model_type: str, trial_number: int) -> None:
    """
    Purpose: Extract per-feature coefficients (linear models) or
        feature_importances_ (tree models) from a fitted Pipeline's 'clf'
        step and log them as an MLflow CSV artifact.
    Leakage guard: N/A.
    Failure mode: If the pipeline's steps aren't named 'pre'/'clf', or the
        estimator exposes neither `coef_` nor `feature_importances_`, this
        silently no-ops rather than crashing an otherwise-successful trial
        over a cosmetic artifact -- caller should spot-check at least one
        trial's artifact tab if this matters.
    """
    try:
        clf = pipeline.named_steps["clf"]
        names = pipeline.named_steps["pre"].get_feature_names_out()
    except (KeyError, AttributeError):
        return
    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:
        return

    importance_df = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", key=np.abs, ascending=False
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"{model_type}_trial_{trial_number}_importance.csv"
        importance_df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
