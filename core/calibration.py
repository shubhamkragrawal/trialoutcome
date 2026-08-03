"""Domain-agnostic probability calibration: isotonic regression, Platt scaling,
and Expected Calibration Error. No pharma-specific strings belong in this file --
see notebooks/03_calibration.ipynb for the concrete TrialOutcome application.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """
    Purpose: Compute ECE = sum_b (n_b/n) * |bin_accuracy - bin_confidence|
        over n_bins equal-width bins on [0,1], plus the per-bin
        reliability-curve data (mean predicted prob vs fraction of
        positives) needed to plot it.
    Leakage guard: N/A -- pure metric computation on already-produced
        (y_true, y_proba) pairs; callers decide which split those pairs
        came from.
    Failure mode: Bins with zero samples contribute 0 to the ECE sum
        (correct -- an empty bin has no calibration error to report) but
        are dropped from the returned reliability-curve DataFrame rather
        than plotted as a misleading 0/0 point.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_proba[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        rows.append(
            {
                "bin": b,
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "count": count,
                "mean_predicted_prob": bin_conf,
                "fraction_positives": bin_acc,
            }
        )
    return ece, pd.DataFrame(rows)


@dataclass
class CalibrationResult:
    """
    Purpose: Bundle everything one CalibratorWrapper.fit() call produces --
        both fitted calibrators, every ECE number, and the reliability-curve
        data for all three states (raw/isotonic/Platt) -- so callers (the
        M3 notebook) get one object instead of unpacking a long tuple.
    Leakage guard: N/A (value container).
    Failure mode: N/A.
    """

    isotonic: IsotonicRegression
    platt: LogisticRegression
    ece_before: float
    ece_after_isotonic: float
    ece_after_platt: float
    chosen_method: str
    reliability_before: pd.DataFrame
    reliability_after_isotonic: pd.DataFrame
    reliability_after_platt: pd.DataFrame


class CalibratorWrapper:
    """
    Purpose: Fit isotonic regression and Platt scaling on a CALIB split's
        raw model probabilities, compare ECE before/after each, and select
        the winner (lowest post-calibration ECE) -- isotonic usually wins
        since it makes no parametric shape assumption, but this is decided
        empirically per run, not hardcoded.
    Leakage guard: fit() must only ever be called with the CALIB split's
        (y_true, y_proba) -- never TEST. TEST exists purely to report the
        final ECE of the already-chosen calibrator once, with no further
        selection happening against it (see notebooks/03_calibration.ipynb).
    Failure mode: N/A (class-level).
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.isotonic_: IsotonicRegression | None = None
        self.platt_: LogisticRegression | None = None
        self.result_: CalibrationResult | None = None

    def fit(self, y_true, y_proba) -> CalibrationResult:
        """
        Purpose: Fit both calibrators on CALIB-split (y_true, y_proba),
            compute ECE before/after each, and pick the winner by lowest
            post-calibration ECE.
        Leakage guard: Both calibrators are fit exclusively on the arrays
            passed in here -- callers must pass the CALIB split, not TEST.
        Failure mode: If y_proba has near-zero variance (a degenerate
            model), IsotonicRegression still fits (returns a near-constant
            step function) rather than raising -- the resulting ECE would
            just be poor, which is a legitimate finding to surface, not
            hide behind an exception.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_proba = np.asarray(y_proba, dtype=float)

        ece_before, reliability_before = expected_calibration_error(y_true, y_proba, self.n_bins)

        self.isotonic_ = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_.fit(y_proba, y_true)
        proba_isotonic = self.isotonic_.predict(y_proba)
        ece_after_isotonic, reliability_after_isotonic = expected_calibration_error(
            y_true, proba_isotonic, self.n_bins
        )

        # Platt scaling: sigmoid(a*raw_proba + b) fit as a 1-feature logistic
        # regression, unregularized (C very large) so it approximates the
        # classic Platt (1999) MLE fit rather than being shrunk toward 0.
        self.platt_ = LogisticRegression(C=1e10, solver="lbfgs")
        self.platt_.fit(y_proba.reshape(-1, 1), y_true)
        proba_platt = self.platt_.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        ece_after_platt, reliability_after_platt = expected_calibration_error(
            y_true, proba_platt, self.n_bins
        )

        chosen_method = "isotonic" if ece_after_isotonic <= ece_after_platt else "platt"

        self.result_ = CalibrationResult(
            isotonic=self.isotonic_,
            platt=self.platt_,
            ece_before=ece_before,
            ece_after_isotonic=ece_after_isotonic,
            ece_after_platt=ece_after_platt,
            chosen_method=chosen_method,
            reliability_before=reliability_before,
            reliability_after_isotonic=reliability_after_isotonic,
            reliability_after_platt=reliability_after_platt,
        )
        return self.result_

    def predict(self, y_proba, method: str | None = None) -> np.ndarray:
        """
        Purpose: Apply a fitted calibrator (default: the method chosen by
            fit()) to new raw probabilities -- e.g. the TEST split, for
            final reporting only.
        Leakage guard: N/A -- applies an already-fit transform; no fitting
            happens here.
        Failure mode: Raises RuntimeError if called before fit().
        """
        if self.result_ is None:
            raise RuntimeError("CalibratorWrapper.predict() called before fit()")
        method = method or self.result_.chosen_method
        y_proba = np.asarray(y_proba, dtype=float)
        if method == "isotonic":
            return self.isotonic_.predict(y_proba)
        return self.platt_.predict_proba(y_proba.reshape(-1, 1))[:, 1]

    def log_to_mlflow(
        self, tracking_uri: str, experiment_name: str, run_name: str = "xgboost_best_calibrated"
    ) -> str:
        """
        Purpose: Log ece_before/ece_after_isotonic/ece_after_platt as MLflow
            metrics and calibration_method_chosen as a tag on a new run
            (default name 'xgboost_best_calibrated' per M3 spec).
        Leakage guard: N/A -- logging only.
        Failure mode: Raises RuntimeError if called before fit().
        """
        if self.result_ is None:
            raise RuntimeError("CalibratorWrapper.log_to_mlflow() called before fit()")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("calibration_method_chosen", self.result_.chosen_method)
            mlflow.log_metric("ece_before", self.result_.ece_before)
            mlflow.log_metric("ece_after_isotonic", self.result_.ece_after_isotonic)
            mlflow.log_metric("ece_after_platt", self.result_.ece_after_platt)
            return run.info.run_id
