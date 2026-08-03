"""Unit tests for core/monitoring/drift_base.py against tiny synthetic
DataFrames -- no DB required. Requires `evidently` importable; see
core/monitoring/drift_base.py's module docstring for the
NLTK_DISABLE_IMPORT_SECURITY=1 env var needed to import it in this
project's venv-inside-repo layout (set at the workflow level in
.github/workflows/ci.yml for both CI jobs).

test_per_feature_drift_direction_matches_evidently_dataset_verdict is a
regression test for a real bug caught while building this file (see
decisions.md's M6 entry and per_feature_drift's docstring): a first version
flagged "drifted" via `score < threshold` unconditionally, which is only
correct for p-value-based drift methods (K-S, chi-square) and is the
OPPOSITE of correct for distance-based methods (Wasserstein, Jensen-Shannon)
-- exactly the methods Evidently auto-selects for this project's real
numeric features (see notebooks/06_drift_report.ipynb).
"""

import numpy as np
import pandas as pd

from core.monitoring.drift_base import DriftMonitorBase


class _FixtureDriftMonitor(DriftMonitorBase):
    """Minimal concrete subclass -- reference/current are supplied directly
    by each test rather than loaded from a DB."""

    def load_reference(self) -> pd.DataFrame:
        raise NotImplementedError("not needed for these tests")

    def load_current(self) -> pd.DataFrame:
        raise NotImplementedError("not needed for these tests")


def _make_frames(n: int = 300, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "stable_numeric": rng.normal(0, 1, n),
            "shifted_numeric": rng.normal(0, 1, n),
        }
    )
    current = pd.DataFrame(
        {
            "stable_numeric": rng.normal(0, 1, n),
            "shifted_numeric": rng.normal(shift, 1, n),
        }
    )
    return reference, current


def test_check_thresholds_flags_dataset_drift_above_share_threshold():
    monitor = _FixtureDriftMonitor()
    # _make_frames only shifts "shifted_numeric" -- "stable_numeric" is drawn
    # from the identical distribution in both frames and should stay clean.
    reference, current = _make_frames(shift=5.0)
    result = monitor.check_thresholds(
        monitor.score_batch(reference, current, drift_share=0.5), feature_drift_threshold=0.5
    )
    assert result.n_features_drifted == 1
    assert result.drift_share == 0.5
    assert result.drifted is True  # 0.5 share meets the >=0.5 threshold


def test_check_thresholds_does_not_flag_when_nothing_drifts():
    monitor = _FixtureDriftMonitor()
    reference, current = _make_frames(shift=0.0)  # no real shift in either column
    snapshot = monitor.score_batch(reference, current, drift_share=0.5)
    result = monitor.check_thresholds(snapshot, feature_drift_threshold=0.5)
    assert result.n_features_drifted == 0
    assert result.drifted is False


def test_per_feature_drift_direction_matches_evidently_dataset_verdict():
    """Regression test for the score-direction bug: whichever features
    per_feature_drift() marks drifted=True must be the same count Evidently's
    own DriftedColumnsCount reports -- if the direction were inverted again,
    these two counts would disagree."""
    monitor = _FixtureDriftMonitor()
    reference, current = _make_frames(shift=5.0)
    snapshot = monitor.score_batch(reference, current, drift_share=0.5)

    dataset_result = monitor.check_thresholds(snapshot, feature_drift_threshold=0.5)
    per_feature = monitor.per_feature_drift(snapshot)

    assert int(per_feature["drifted"].sum()) == dataset_result.n_features_drifted
    assert set(per_feature.loc[per_feature["drifted"], "feature"]) == {"shifted_numeric"}


def test_per_feature_drift_severity_ranks_most_drifted_first():
    monitor = _FixtureDriftMonitor()
    reference, current = _make_frames(n=500, shift=3.0)
    snapshot = monitor.score_batch(reference, current, drift_share=0.5)
    per_feature = monitor.per_feature_drift(snapshot)

    assert per_feature.iloc[0]["feature"] == "shifted_numeric"
    assert per_feature.iloc[0]["severity"] >= per_feature.iloc[1]["severity"]
