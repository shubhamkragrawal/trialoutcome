"""Domain-agnostic batch drift-monitoring scaffolding: Evidently report
generation and threshold evaluation. No pharma column names, pharma table
names, or pharma threshold values belong in this file -- see
domains/pharma/monitoring/drift_job.py for the concrete subclass.

NOTE (env quirk, not a code bug): importing `evidently` pulls in `nltk`
transitively (evidently.legacy.metrics -> text-drift features -> nltk), and
this project's `.venv` lives *inside* the repo root. NLTK 2026's own
`inisec.py` import-security hook (CWE-427 mitigation) blocks any import that
resolves to a path under the current working directory when NLTK is an
ancestor frame -- a legitimate check in general, but a false positive here
specifically because the venv's site-packages happen to be a subdirectory of
this project's CWD, not because anything is actually being hijacked. NLTK
ships a documented escape hatch for exactly this case:
`NLTK_DISABLE_IMPORT_SECURITY=1`. `make drift` and the CI job both set it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset


@dataclass(frozen=True)
class DriftResult:
    """
    Purpose: Threshold-evaluation output of check_thresholds() -- whether
        the batch is considered drifted, how many features drifted, and
        what fraction of compared features that represents.
    Leakage guard: N/A.
    Failure mode: N/A (plain value container).
    """

    drifted: bool
    n_features_drifted: int
    drift_share: float


class DriftMonitorBase(ABC):
    """
    Purpose: Abstract base for batch drift-monitoring jobs. Evidently report
        generation and threshold evaluation are identical across domains;
        only what counts as "reference" data and "current batch" data is
        domain-specific (a subclass's job is to know where its own training
        set and its own newly-scored batch live).
    Leakage guard: N/A -- this runs entirely after training against
        already-built, already-split datasets; there is no train/test
        boundary for this class to protect.
    Failure mode: N/A (abstract).
    """

    @abstractmethod
    def load_reference(self) -> pd.DataFrame:
        """Return the reference (training-time) feature DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def load_current(self) -> pd.DataFrame:
        """
        Purpose: Return the current-batch feature DataFrame to compare
            against the reference.
        Leakage guard: N/A.
        Failure mode: N/A here -- the honesty of what "current" actually
            means (real live traffic vs. a proxy) is the subclass's
            responsibility to document, not this base class's.

        NOTE: not one of the four method names the M6 brief listed
        (load_reference/score_batch/generate_report/check_thresholds) --
        added because score_batch() needs two DataFrames to compare and a
        base class has no legitimate way to guess where a domain's "current
        batch" data lives. Flagged here rather than silently expanding the
        brief's method list without saying so.
        """
        raise NotImplementedError

    def score_batch(
        self, reference: pd.DataFrame, current: pd.DataFrame, drift_share: float = 0.5
    ) -> Snapshot:
        """
        Purpose: Run Evidently's DataDriftPreset across every feature column
            shared by reference and current, returning the raw Snapshot for
            the caller to render (generate_report) and/or evaluate
            (check_thresholds).
        Leakage guard: N/A.
        Failure mode: Evidently infers each column's type (numeric vs.
            categorical) from dtype and silently skips columns it can't
            compare rather than raising -- if reference/current don't share
            an (almost) identical column set, a caller could get a
            misleadingly low drift count without any error being raised.
            This method restricts both frames to their common columns
            explicitly so at least the comparison set is auditable.
        """
        common_cols = [c for c in reference.columns if c in current.columns]
        ref_ds = Dataset.from_pandas(reference[common_cols])
        cur_ds = Dataset.from_pandas(current[common_cols])
        report = Report(metrics=[DataDriftPreset(drift_share=drift_share)])
        return report.run(current_data=cur_ds, reference_data=ref_ds)

    def generate_report(self, snapshot: Snapshot, report_path: Path) -> Path:
        """
        Purpose: Persist the Evidently Snapshot as a standalone HTML report.
        Leakage guard: N/A.
        Failure mode: N/A -- Snapshot.save_html raises loudly on a bad path
            rather than silently writing nothing.
        """
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_path))
        return report_path

    def check_thresholds(self, snapshot: Snapshot, feature_drift_threshold: float) -> DriftResult:
        """
        Purpose: Parse the Snapshot's DriftedColumnsCount metric into the
            {drifted, n_features_drifted, drift_share} verdict a caller logs
            to its own drift-log table.
        Leakage guard: N/A.
        Failure mode: If a future Evidently version renames
            "DriftedColumnsCount" or restructures Snapshot.dict()'s shape,
            this raises StopIteration/KeyError rather than silently
            returning a zero-drift false negative -- a loud failure here is
            safer than a monitoring job that always reports "no drift"
            because it can no longer parse its own dependency's output.
        """
        payload = snapshot.dict()["metrics"]
        counts = next(m for m in payload if m["metric_name"].startswith("DriftedColumnsCount"))
        n_drifted = int(counts["value"]["count"])
        drift_share = float(counts["value"]["share"])
        return DriftResult(
            drifted=drift_share >= feature_drift_threshold,
            n_features_drifted=n_drifted,
            drift_share=drift_share,
        )

    def per_feature_drift(self, snapshot: Snapshot) -> pd.DataFrame:
        """
        Purpose: Return one row per compared feature with its Evidently
            drift score/method/threshold, a per-column drifted flag, and a
            `severity` column (drift-score margin past that column's
            threshold, always positive-means-worse regardless of method)
            usable for ranking "most drifted" across mixed methods -- sorted
            most-severe first.
        Leakage guard: N/A.
        Failure mode (a real bug caught while building this): Evidently
            auto-selects a *different* drift-detection method per column
            depending on type/cardinality (e.g. "K-S p_value" for
            well-behaved numeric columns in a small toy test, but
            "Wasserstein distance (normed)" or "Jensen-Shannon distance" for
            this project's actual features). p-value methods flag drift when
            `score < threshold`; distance/divergence methods flag drift when
            `score > threshold` -- the OPPOSITE direction. A first version of
            this method used `score < threshold` unconditionally, which
            silently inverted the drifted flag for every Wasserstein/JS
            column (the majority of this project's real features) --
            verified by cross-checking against Evidently's own per-column
            pass/fail test status (`Report(..., include_tests=True)`) before
            landing on the `"p_value" in method` direction rule below, which
            matched every row in that cross-check.
        """
        payload = snapshot.dict()["metrics"]
        rows = []
        for m in payload:
            if not m["metric_name"].startswith("ValueDrift"):
                continue
            score = float(m["value"])
            method = m["config"].get("method", "")
            threshold = float(m["config"].get("threshold", 0.05))
            is_pvalue_method = "p_value" in method.lower()
            drifted = score < threshold if is_pvalue_method else score > threshold
            severity = (threshold - score) if is_pvalue_method else (score - threshold)
            rows.append(
                {
                    "feature": m["config"]["column"],
                    "drift_score": score,
                    "method": method,
                    "threshold": threshold,
                    "drifted": drifted,
                    "severity": severity,
                }
            )
        return pd.DataFrame(rows).sort_values("severity", ascending=False).reset_index(drop=True)
