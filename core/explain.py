"""Domain-agnostic SHAP explainability wrapper. No pharma-specific feature
names or template strings belong in this file -- see
domains/pharma/plain_english.py for the concrete TrialOutcome templating
layer built on top of this module's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Purpose: Wrap shap.TreeExplainer (XGBoost/LightGBM/RandomForest) with a
        fallback to shap.LinearExplainer (LogisticRegression), so the rest
        of the pipeline calls one consistent interface regardless of which
        model family won M2's Optuna sweep.
    Leakage guard: N/A -- SHAP explains an already-fitted model's
        predictions; it does not touch training data or labels. Callers
        are responsible for only calling compute_shap_values on TEST-split
        rows (see notebooks/04_shap_analysis.ipynb) -- computing SHAP on
        TRAIN would explain overfit/memorized behavior, not generalization.
    Failure mode: N/A (class-level).
    """

    def __init__(self, model, X_background: pd.DataFrame | None = None):
        """
        Purpose: Build the appropriate SHAP explainer for `model`'s type at
            construction time (not lazily), so a caller finds out
            immediately if the model type isn't supported rather than
            failing deep inside compute_shap_values.
        Leakage guard: N/A.
        Failure mode: Raises TypeError if `model` exposes neither a
            tree-ensemble interface (feature_importances_ / get_booster)
            nor a linear one (coef_) -- rather than silently returning an
            explainer that will fail on first use.
        """
        self.model = model
        if hasattr(model, "coef_"):
            if X_background is None:
                raise ValueError("X_background is required for shap.LinearExplainer")
            self.explainer = shap.LinearExplainer(model, X_background)
            self.model_family = "linear"
        elif hasattr(model, "feature_importances_") or hasattr(model, "get_booster"):
            self.explainer = shap.TreeExplainer(model)
            self.model_family = "tree"
        else:
            raise TypeError(
                f"SHAPExplainer does not support model type {type(model)!r} -- "
                "expected a tree ensemble (feature_importances_/get_booster) or "
                "a linear model (coef_)"
            )

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Purpose: Return the raw SHAP values array for X, shape
            (n_rows, n_features), for the positive class.
        Leakage guard: N/A here -- caller decides which split X comes from;
            this project's convention is TEST only (see class docstring).
        Failure mode: If X's columns don't match what `model` was fit on,
            the underlying SHAP explainer raises rather than silently
            misaligning values to the wrong feature names.
        """
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):
            # Some SHAP/model-family combinations return one array per
            # class; index 1 is the positive class for binary classification.
            raw = raw[1]
        raw = np.asarray(raw)
        if raw.ndim == 3:
            # Some TreeExplainer/XGBoost/LightGBM combinations return
            # (n_rows, n_features, n_classes) for binary classifiers.
            raw = raw[:, :, 1]
        return raw

    def global_importance(self, shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
        """
        Purpose: Aggregate per-row SHAP values into a global ranking:
            mean(|SHAP|) per feature, sorted descending.
        Leakage guard: N/A.
        Failure mode: Raises ValueError if shap_values.shape[1] !=
            len(feature_names) -- a silent misalignment here would mislabel
            every downstream importance number with the wrong feature name.
        """
        if shap_values.shape[1] != len(feature_names):
            raise ValueError(
                f"shap_values has {shap_values.shape[1]} columns but "
                f"{len(feature_names)} feature_names were given"
            )
        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(
        self, shap_values: np.ndarray, X: pd.DataFrame, idx: int, top_n: int = 5
    ) -> list[dict]:
        """
        Purpose: Return the top-`top_n` SHAP contributors for one row
            (X.iloc[idx]), each as {feature, value, shap_contribution},
            sorted by |shap_contribution| descending -- the shape
            domains/pharma/plain_english.generate_summary expects.
        Leakage guard: N/A.
        Failure mode: If idx is out of range, pandas/numpy raise IndexError
            -- not caught here, since a caller passing a bad index is a
            programming error that should surface immediately.
        """
        row_shap = shap_values[idx]
        row_values = X.iloc[idx]
        order = np.argsort(-np.abs(row_shap))[:top_n]
        return [
            {
                "feature": X.columns[i],
                "value": row_values.iloc[i],
                "shap_contribution": float(row_shap[i]),
            }
            for i in order
        ]

    def worst_false_negatives(
        self,
        X_test: pd.DataFrame,
        y_test,
        y_proba: np.ndarray,
        threshold: float,
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Purpose: Return the n false-negative rows (true label positive,
            predicted probability below threshold) with the LOWEST
            predicted probability -- the model's most confident wrong
            misses, the hardest cases for the error essay.
        Leakage guard: N/A.
        Failure mode: If fewer than n false negatives exist at the given
            threshold, returns all of them rather than raising or padding
            with unrelated rows.
        """
        y_test_arr = np.asarray(y_test).astype(int)
        y_proba_arr = np.asarray(y_proba)
        is_fn = (y_test_arr == 1) & (y_proba_arr < threshold)
        fn_idx = np.where(is_fn)[0]
        fn_idx_sorted = fn_idx[np.argsort(y_proba_arr[fn_idx])]
        chosen = fn_idx_sorted[:n]
        out = X_test.iloc[chosen].copy()
        out["proba"] = y_proba_arr[chosen]
        out["true_label"] = y_test_arr[chosen]
        out["row_index"] = chosen
        return out
