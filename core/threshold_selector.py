"""Domain-agnostic cost-based threshold selection. No pharma cost-matrix
numbers belong in this file -- see domains/pharma/cost_matrix.yaml for the
concrete FN:FP ratio TrialOutcome sweeps against.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class ThresholdSelector:
    """
    Purpose: Sweep classification thresholds against a caller-supplied cost
        matrix (FN cost, FP cost) to find the threshold minimizing expected
        cost, alongside F1-max and default-0.5 thresholds for comparison.
    Leakage guard: THRESHOLD SELECTION IS A ONE-PARAMETER FIT AND MUST RUN ON
        A SPLIT HELD OUT FROM EVALUATION. Selecting one of 99 candidates by
        minimizing an objective computed on a split is fitting that split,
        regardless of how cheap the parameter is -- metrics reported at the
        winning threshold on that same split are optimistically biased.
        Callers must sweep on CALIB (find_cost_optimal_threshold) and report
        on TEST (metrics_at_threshold). This class does NOT and CANNOT enforce
        which split it is handed -- it sees only (y_true, y_proba) arrays, so
        the discipline is the caller's responsibility.

        CORRECTED IN M9 (review section 1.2). This docstring previously argued
        that threshold selection was "a downstream reporting step, not a
        fitting step, so which split it runs on isn't a leakage risk the way
        calibrator fitting is." That reasoning was wrong and the M3 notebook
        acted on it, sweeping and reporting on TEST. The bias is small (one
        parameter, 5,700 rows) but the argument does not hold. See
        decisions.md M9-4.
    Failure mode: N/A (class-level).
    """

    def find_cost_optimal_threshold(
        self,
        y_true,
        y_proba,
        fn_cost: float,
        fp_cost: float,
        step: float = 0.01,
    ) -> dict:
        """
        Purpose: Sweep thresholds 0.01..0.99 in `step` increments; at each,
            compute expected_cost = FN_count*fn_cost + FP_count*fp_cost, and
            return the cost-optimal threshold alongside F1-max and the
            default 0.5, plus each candidate's expected cost.
        Leakage guard: pass CALIB probabilities here, never TEST -- this is
            the fitting call (see the class docstring). Reporting at the
            returned threshold belongs in metrics_at_threshold() against TEST.
        Failure mode: N/A -- a 99-point grid sweep over already-computed
            probabilities is O(n) per threshold and always terminates.
            Sweeping 0-1 by hand (rather than an optimizer) is fine here
            because the search space is tiny (99 points) and the cost
            function is piecewise-constant in the threshold (it only
            changes at points where a prediction flips), not smooth -- a
            gradient-based optimizer gains nothing over a grid here. At
            production scale, with a genuinely continuous/differentiable
            cost function (e.g. cost that also varies by predicted
            probability, not just by which side of the threshold a
            prediction falls), scipy.optimize.minimize_scalar (bounded to
            [0, 1]) would replace this grid sweep.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        thresholds = np.round(np.arange(0.01, 1.0, step), 2)

        costs_by_threshold: dict[float, float] = {}
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            costs_by_threshold[float(t)] = float(fn * fn_cost + fp * fp_cost)

        cost_optimal = min(costs_by_threshold, key=costs_by_threshold.get)

        f1_scores = {
            float(t): f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            for t in thresholds
        }
        f1_max = max(f1_scores, key=f1_scores.get)

        default = 0.5
        if default not in costs_by_threshold:
            y_pred = (y_proba >= default).astype(int)
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            costs_by_threshold[default] = float(fn * fn_cost + fp * fp_cost)

        return {
            "cost_optimal": cost_optimal,
            "f1_max": f1_max,
            "default": default,
            "cost_at_cost_optimal": costs_by_threshold[cost_optimal],
            "cost_at_f1_max": costs_by_threshold[f1_max],
            "cost_at_default": costs_by_threshold[default],
            "costs_by_threshold": costs_by_threshold,
        }

    def metrics_at_threshold(
        self, y_true, y_proba, threshold: float, fn_cost: float, fp_cost: float
    ) -> dict:
        """
        Purpose: Compute precision/recall/f1/expected-cost at one specific
            threshold -- used to build a decision table across the three
            candidate thresholds (cost-optimal / F1-max / default).
        Leakage guard: N/A.
        Failure mode: N/A -- sklearn's precision/recall/f1 all handle the
            zero-positive-prediction edge case via zero_division=0 rather
            than raising.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        return {
            "threshold": threshold,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "expected_cost": float(fn * fn_cost + fp * fp_cost),
        }
