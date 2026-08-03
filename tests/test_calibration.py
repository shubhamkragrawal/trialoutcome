"""Fast, dependency-free unit tests for core/calibration.py's
expected_calibration_error -- no DB, no MLflow required (see
tests/test_plain_english.py's module docstring for why CI needs these)."""

import numpy as np

from core.calibration import expected_calibration_error


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    y_proba = rng.uniform(0, 1, size=5000)
    y_true = (rng.uniform(0, 1, size=5000) < y_proba).astype(int)
    ece, _ = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece < 0.03


def test_badly_miscalibrated_predictions_have_high_ece():
    y_proba = np.full(1000, 0.9)
    y_true = np.zeros(1000, dtype=int)  # model says 90% positive, actual rate is 0%
    ece, _ = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece > 0.8


def test_reliability_dataframe_only_contains_nonempty_bins():
    y_proba = np.array([0.05, 0.05, 0.95, 0.95])
    y_true = np.array([0, 0, 1, 1])
    _, reliability = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert reliability["count"].sum() == 4
    assert (reliability["count"] > 0).all()
