"""Domain-agnostic dataset-building scaffolding: temporal split, random split,
and DB write-back. No pharma SQL or pharma column names belong in this file --
see domains/pharma/dataset_builder.py for the concrete subclass.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class SplitDates:
    """
    Purpose: Carry the two boundary dates that separate train/calib/test for
        temporal_split(). Immutable so a builder can't accidentally mutate
        split boundaries mid-run.
    Leakage guard: N/A (a plain value container).
    Failure mode: N/A.
    """

    train_end: pd.Timestamp
    calib_end: pd.Timestamp


class TemporalDatasetBuilder(ABC):
    """
    Purpose: Abstract base for point-in-time-safe dataset builders. Subclasses
        supply domain SQL (fetch_raw) and domain feature engineering
        (build_features); this class supplies the split and persistence logic
        that is identical across domains (trials, loans, claims, ...).
    Leakage guard: Centralizing temporal_split/random_split here means every
        domain subclass gets the same, once-reviewed split logic instead of
        each domain reimplementing (and potentially breaking) it.
    Failure mode: N/A (abstract).
    """

    @abstractmethod
    def fetch_raw(self) -> pd.DataFrame:
        """
        Purpose: Pull the raw, entity-level rows needed for feature
            engineering from the domain's data source.
        Leakage guard: N/A here -- leakage avoidance is the subclass's
            responsibility (e.g. point-in-time joins in SQL).
        Failure mode: If this returns rows that aren't one-per-entity (e.g. a
            fanned-out join), every downstream count and split proportion is
            silently wrong.
        """
        raise NotImplementedError

    @abstractmethod
    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose: Turn raw rows into a feature-engineered DataFrame with, at
            minimum, an entity id column, an event-date column (used for
            temporal_split), and a label column.
        Leakage guard: N/A here -- must be enforced by the subclass (e.g. only
            using information available at the entity's start/event date).
        Failure mode: If a feature here is computed using information that
            postdates the event, temporal_split will not catch it -- the
            leakage lives in feature values, not in row placement.
        """
        raise NotImplementedError

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_col: str,
        split_dates: SplitDates,
        split_col: str = "split",
    ) -> pd.DataFrame:
        """
        Purpose: Assign each row to 'train' / 'calib' / 'test' by comparing
            date_col against split_dates, so evaluation always happens on
            strictly-later data than training -- the honest, leakage-safe
            split.
        Leakage guard: This IS the leakage guard. Without it, a model can see
            future outcomes (e.g. future sponsor termination rates) during
            training, inflating validation metrics in a way that will not
            hold in production, where the future is genuinely unknown.
        Failure mode: If date_col contains nulls or out-of-range values, those
            rows get miscategorized silently (pandas comparisons against NaT
            return False, so null-dated rows fall through to whatever the
            last branch is) -- callers must filter implausible dates before
            calling this.
        """
        out = df.copy()
        dates = pd.to_datetime(out[date_col])
        conditions = [
            dates < split_dates.train_end,
            (dates >= split_dates.train_end) & (dates < split_dates.calib_end),
            dates >= split_dates.calib_end,
        ]
        choices = ["train", "calib", "test"]
        out[split_col] = np.select(conditions, choices, default="test")
        return out

    def random_split(
        self,
        df: pd.DataFrame,
        split_col: str = "split",
        train_frac: float = 0.6,
        calib_frac: float = 0.2,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Purpose: Assign each row to 'train' / 'calib' / 'test' by random
            shuffle at the given proportions, ignoring event date entirely.
            Exists ONLY as a contrast case for the leakage demo -- it is
            deliberately the leakage-unsafe baseline, not a recommended split.
        Leakage guard: This function is the negative control, not a guard --
            it is what temporal_split is protecting against. Any feature that
            encodes future information (e.g. lifetime sponsor stats) will
            leak here and inflate metrics versus the temporal split.
        Failure mode: If this is ever used to build the dataset that actually
            gets deployed (instead of only for the leakage-demo notebook), the
            resulting model's offline metrics will be optimistic versus real
            production performance -- the exact bug this whole file exists to
            prevent.
        """
        out = df.copy()
        rng = np.random.default_rng(random_state)
        n = len(out)
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)
        assignment = np.empty(n, dtype=object)
        assignment[indices[:n_train]] = "train"
        assignment[indices[n_train : n_train + n_calib]] = "calib"
        assignment[indices[n_train + n_calib :]] = "test"
        out[split_col] = assignment
        return out

    def controlled_leakage_ablation(
        self,
        df: pd.DataFrame,
        date_col: str,
        test_window_start: pd.Timestamp,
        test_window_end: pd.Timestamp,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Purpose: Build the three DataFrames needed to isolate the temporal-
            leakage mechanism directly, rather than comparing two differently-
            composed splits (which confounds leakage with base-rate and
            data-richness differences -- see notebooks/02_leakage_demo.ipynb).
            Returns (fixed_test, honest_train, leaky_train): one identical
            test set, and two same-size training sets that differ only in
            whether rows postdating the test window are allowed in.
        Leakage guard: This IS the leakage-isolation tool. `honest_train`
            contains only rows strictly before `test_window_start` -- what a
            production model could actually have seen. `leaky_train` is drawn
            from everywhere outside the test window, so it may include rows
            after `test_window_end` -- information a production model
            deployed at `test_window_start` could never have had. Comparing
            a model fit on each, evaluated on the same `fixed_test`, isolates
            whether that specific extra information changes performance.
        Failure mode: If `honest_train` and `leaky_train` aren't the same
            size, any performance difference is confounded with sample-size
            effects instead of isolating the leakage mechanism -- this method
            enforces equal size by construction (leaky_train is sampled down
            to len(honest_train)).
        """
        dates = pd.to_datetime(df[date_col])
        fixed_test = df[(dates >= test_window_start) & (dates < test_window_end)].copy()
        honest_train = df[dates < test_window_start].copy()
        leaky_pool = df[(dates < test_window_start) | (dates >= test_window_end)].copy()
        leaky_train = leaky_pool.sample(n=len(honest_train), random_state=random_state)
        return fixed_test, honest_train, leaky_train

    def write_to_db(
        self,
        df: pd.DataFrame,
        engine: Engine,
        schema: str,
        table: str,
        pk_col: str,
        feature_cols: list[str],
        label_col: str,
        split_col: str = "split",
    ) -> int:
        """
        Purpose: Persist a built dataset to a Postgres table with the shape
            (pk, features JSONB, label BOOL, split TEXT), upserting on pk_col
            so reruns of the same build are idempotent.
        Leakage guard: N/A -- this only writes what build_features/split
            already produced; it does not itself touch dates or labels.
        Failure mode: If feature_cols omits a column the model later expects,
            or includes a column that shouldn't be a model feature (e.g. the
            label itself), that mistake is now baked into every downstream
            training run reading from this table.
        """
        payload = df[[pk_col, label_col, split_col] + feature_cols].copy()
        payload["features_json"] = payload[feature_cols].apply(
            lambda row: json.dumps(row.to_dict(), default=_json_default), axis=1
        )
        records = payload[[pk_col, "features_json", label_col, split_col]].rename(
            columns={pk_col: "pk", label_col: "label", split_col: "split"}
        )

        from sqlalchemy import text

        if not pk_col.isidentifier():
            raise ValueError(f"pk_col {pk_col!r} is not a safe SQL identifier")

        insert_sql = text(
            f"""
            INSERT INTO {schema}.{table} ({pk_col}, features, label, split)
            VALUES (:pk, CAST(:features_json AS JSONB), :label, :split)
            ON CONFLICT ({pk_col}) DO UPDATE SET
                features = EXCLUDED.features,
                label = EXCLUDED.label,
                split = EXCLUDED.split
            """
        )
        with engine.begin() as conn:
            conn.execute(insert_sql, records.to_dict(orient="records"))
        return len(records)


def _json_default(value):
    """
    Purpose: Make numpy scalar types (int64, float64, bool_) JSON-serializable
        when features are dumped to JSONB.
    Leakage guard: N/A.
    Failure mode: Without this, json.dumps raises TypeError on any numpy
        scalar (e.g. a value computed via .mean()), crashing the write step.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
