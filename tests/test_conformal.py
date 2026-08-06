"""Fast, dependency-free unit tests for M9-21: core/conformal.py's
verify_coverage() tolerance derived from target_coverage (not hardcoded
0.88), plus its over-coverage warning. No DB, no MLflow, no real MAPIE fit
required -- verify_coverage() only touches self.mapie_.predict_set(),
stubbed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.conformal import MAPIEConformalWrapper


class _FakeMapie:
    def __init__(self, pred_sets: np.ndarray):
        self._pred_sets = pred_sets

    def predict_set(self, X):
        return None, self._pred_sets


def _wrapper_with_coverage(
    target_coverage: float, empirical: float, n: int = 100
) -> MAPIEConformalWrapper:
    """Builds a wrapper whose verify_coverage() measures exactly `empirical`
    coverage against y_test=[1]*n, via pred_sets where round(empirical*n)
    rows have class 1 in their predicted set."""
    wrapper = MAPIEConformalWrapper(target_coverage=target_coverage)
    n_covered = round(empirical * n)
    # shape (n, 2, 1): [:, 0, 0]=class-0 membership, [:, 1, 0]=class-1 membership
    pred_sets = np.zeros((n, 2, 1), dtype=bool)
    pred_sets[:n_covered, 1, 0] = True
    wrapper.mapie_ = _FakeMapie(pred_sets)
    return wrapper


def _y_test(n: int = 100) -> list[int]:
    return [1] * n


def test_passed_uses_target_minus_2pp_not_hardcoded_88():
    """target_coverage=0.95 -> tolerance is 0.93, not the old hardcoded
    0.88. Under the pre-M9-21 hardcoded gate, empirical=0.90 would have
    (incorrectly) passed even though it under-covers a 0.95 target by 5pp."""
    wrapper = _wrapper_with_coverage(target_coverage=0.95, empirical=0.90)
    result = wrapper.verify_coverage(pd.DataFrame({"x": [0] * 100}), _y_test())
    assert abs(result["empirical"] - 0.90) < 1e-9
    assert result["passed"] is False


def test_passed_true_at_default_target_matches_old_088_boundary():
    """No behavior change at this project's actual target_coverage=0.90:
    0.90 - 0.02 == 0.88, the old hardcoded value."""
    wrapper = _wrapper_with_coverage(target_coverage=0.90, empirical=0.88)
    result = wrapper.verify_coverage(pd.DataFrame({"x": [0] * 100}), _y_test())
    assert result["passed"] is True


def test_over_covered_flagged_when_empirical_exceeds_target_plus_5pp():
    wrapper = _wrapper_with_coverage(target_coverage=0.90, empirical=0.97)
    result = wrapper.verify_coverage(pd.DataFrame({"x": [0] * 100}), _y_test())
    assert result["over_covered"] is True
    assert result["passed"] is True  # over-coverage still passes the floor


def test_not_over_covered_within_5pp_of_target():
    wrapper = _wrapper_with_coverage(target_coverage=0.90, empirical=0.93)
    result = wrapper.verify_coverage(pd.DataFrame({"x": [0] * 100}), _y_test())
    assert result["over_covered"] is False
