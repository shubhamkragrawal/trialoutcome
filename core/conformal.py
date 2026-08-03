"""Domain-agnostic conformal-prediction wrapper around MAPIE. No pharma-
specific strings, thresholds, or column names belong in this file -- see
notebooks/05_conformal.ipynb for the concrete TrialOutcome application.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _NamedColumnEstimatorAdapter:
    """
    Purpose: Restore a fitted estimator's expected DataFrame column names on
        any array-like input before delegating to predict/predict_proba.
    Leakage guard: N/A -- a pure interface shim around an already-fitted
        estimator; no fitting happens here.
    Failure mode: MAPIE's SplitConformalClassifier runs an internal
        "is this estimator actually fitted" probe by calling predict() on a
        bare `np.zeros((1, n_features_in_))` array, bypassing the caller's
        DataFrame entirely. A base estimator whose pipeline selects columns
        by *name* (e.g. `ColumnTransformer([...], ["phase", ...])`, as this
        project's `_make_preprocessor` does) raises
        "Specifying the columns using strings is only supported for
        dataframes" on that bare-array probe -- MAPIE turns that exception
        into a hard `raise UserWarning(...)` (not `warnings.warn`, seemingly
        a MAPIE bug) that aborts `conformalize()` entirely. Even after
        wrapping a bare array into a same-shaped DataFrame, an all-zero probe
        row still breaks any categorical column encoded via OneHotEncoder --
        0.0 is not a valid category, and sklearn's unknown-category path
        crashes trying to check a string categories array for NaN. Since this
        probe's *returned values* are discarded (MAPIE only checks whether
        the call raises), a non-DataFrame input is answered using a real
        template row's values (captured at construction time) rather than
        the literal probe content -- correct dtypes everywhere, and the
        probe's actual numbers were never meaningful anyway. This project
        only ever calls the adapter with real DataFrames for genuine
        predictions, so this branch is exercised solely by MAPIE's internal
        self-check.
    """

    def __init__(self, estimator, template_row: pd.DataFrame):
        self.estimator = estimator
        self.columns = list(template_row.columns)
        self.template_row = template_row.iloc[[0]].copy()
        self.classes_ = estimator.classes_
        self.n_features_in_ = len(self.columns)

    def _as_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        n_rows = np.asarray(X).shape[0]
        return pd.concat([self.template_row] * n_rows, ignore_index=True)

    def predict(self, X):
        return self.estimator.predict(self._as_dataframe(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._as_dataframe(X))


class MAPIEConformalWrapper:
    """
    Purpose: Wrap mapie.classification.SplitConformalClassifier (MAPIE
        >=1.0's renamed API -- the old `MapieClassifier(method="score")`
        class no longer exists; `conformity_score="lac"` is its direct
        successor for binary/multiclass split conformal) to produce a
        [low, high] confidence interval per prediction, calibrated to a
        target marginal coverage.
    Leakage guard: fit_conformal() must only ever be called with the CALIB
        split's (X, y) -- never TEST. TEST is held out for coverage
        verification only (verify_coverage()), exactly mirroring
        core/calibration.py's CalibratorWrapper convention.
    Failure mode: N/A (class-level).
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.calibrated_model_ = None
        self.mapie_ = None
        # NOTE (design decision, prompt was ambiguous here): "margin = 1 -
        # coverage_achieved" is interpreted as the *nominal* coverage this
        # wrapper was configured for (target_coverage), not the empirical
        # coverage measured by verify_coverage(). This is deliberate: margin
        # is used inside predict_with_interval(), which serves live requests
        # where a TEST split isn't available -- it must be computable right
        # after fit_conformal(), not depend on a verify_coverage() call
        # happening first. verify_coverage()'s empirical number is reported
        # separately for the M5 DoD gate, not fed back into the margin.
        self.margin_ = 1.0 - target_coverage

    def fit_conformal(self, calibrated_model, X_calib: pd.DataFrame, y_calib) -> "MAPIEConformalWrapper":
        """
        Purpose: Fit MAPIE's conformalization step on the CALIB split, using
            an already-fitted (and already probability-calibrated) model as
            the base estimator (`prefit=True` -- MAPIE does not refit it).
        CALIB split is used here, not TEST. TEST is held out for coverage
        verification only.
        Leakage guard: `X_calib`/`y_calib` must be the CALIB split -- passing
            TEST here would let the conformal quantile "see" the same data
            verify_coverage() reports on, invalidating the coverage check.
        Failure mode: N/A -- MAPIE's conformalize() raises if `calibrated_model`
            doesn't implement predict_proba.
        """
        from mapie.classification import SplitConformalClassifier

        self.calibrated_model_ = calibrated_model
        adapter = _NamedColumnEstimatorAdapter(calibrated_model, template_row=X_calib)
        self.mapie_ = SplitConformalClassifier(
            estimator=adapter,
            confidence_level=self.target_coverage,
            conformity_score="lac",
            prefit=True,
        )
        self.mapie_.conformalize(X_calib, y_calib)
        return self

    def predict_with_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """
        Purpose: Return (proba, interval) for each row in X.
            proba = calibrated_model_.predict_proba(X)[:, 1].
            interval is derived from MAPIE's prediction SET for the positive
            class ({0}, {1}, or {0,1} -- binary classification), converted
            to a [low, high] float interval as follows:
              {1} only    -> [proba - margin, proba + margin], clipped to [0,1]
              {0, 1} both -> [0.0, 1.0]                          (uncertain)
              {0} only    -> [0.0, proba]
            where margin = 1 - target_coverage (see __init__ note). The
            intuition: a prediction set that (correctly, per the conformal
            guarantee) contains only the positive class is treated as
            "confidently high risk" and gets a narrow band around its own
            proba; a set containing both classes means MAPIE itself couldn't
            rule out the negative class at this coverage level, so the
            interval widens to the full [0, 1] range rather than
            understating that uncertainty.
        Leakage guard: N/A -- inference only.
        Failure mode: Raises RuntimeError if called before fit_conformal().
        """
        if self.mapie_ is None:
            raise RuntimeError("MAPIEConformalWrapper.predict_with_interval() called before fit_conformal()")

        proba = self.calibrated_model_.predict_proba(X)[:, 1]
        _, pred_sets = self.mapie_.predict_set(X)
        # pred_sets shape: (n_samples, n_classes, n_confidence_levels); a
        # single confidence_level was passed at init, so squeeze that axis.
        # Class column order is [0, 1] (MAPIE sorts labels internally).
        sets = pred_sets[:, :, 0]

        intervals: list[tuple[float, float]] = []
        for p, (in_class_0, in_class_1) in zip(proba, sets):
            if in_class_1 and not in_class_0:
                low = max(0.0, float(p) - self.margin_)
                high = min(1.0, float(p) + self.margin_)
            elif in_class_0 and in_class_1:
                low, high = 0.0, 1.0
            elif in_class_0 and not in_class_1:
                low, high = 0.0, float(p)
            else:
                # Degenerate empty set (can happen at very low coverage
                # targets) -- fall back to the full range rather than
                # claiming false precision with an empty interval.
                low, high = 0.0, 1.0
            intervals.append((low, high))
        return proba, intervals

    def verify_coverage(self, X_test: pd.DataFrame, y_test) -> dict:
        """
        Purpose: Compute empirical coverage on the TEST split -- the
            fraction of rows whose true label is inside MAPIE's predicted
            set -- and compare against the target with a 2pp tolerance.
        Leakage guard: X_test/y_test must be TEST, never CALIB (CALIB was
            already used to fit the conformal quantile in fit_conformal();
            checking coverage on the same data it was calibrated on would
            be circular and not a genuine generalization check).
        Failure mode: Raises RuntimeError if called before fit_conformal().
        """
        if self.mapie_ is None:
            raise RuntimeError("MAPIEConformalWrapper.verify_coverage() called before fit_conformal()")

        y_test_arr = np.asarray(y_test).astype(int)
        _, pred_sets = self.mapie_.predict_set(X_test)
        sets = pred_sets[:, :, 0]
        in_set = sets[np.arange(len(y_test_arr)), y_test_arr]
        empirical = float(in_set.mean())
        passed = empirical >= 0.88
        result = {"target": self.target_coverage, "empirical": empirical, "passed": passed}
        print(f"Conformal coverage: {empirical:.3f} (target {self.target_coverage:.2f})")
        return result
