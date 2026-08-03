"""Pharma-domain dataset builder for TrialOutcome M1. All pharma SQL, label
definition, and feature-engineering rules live here -- core/dataset_builder_base.py
stays domain-agnostic. Run as `python -m domains.pharma.dataset_builder`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from core.dataset_builder_base import SplitDates, TemporalDatasetBuilder

PACKAGE_DIR = Path(__file__).parent
REPO_ROOT = PACKAGE_DIR.parent.parent
CONFIG_PATH = PACKAGE_DIR / "config.yaml"

# --- The single feature-fetch query. All joins, tie-breaks, and point-in-time
# self-joins live in this string per "no raw SQL in core/". -----------------
_RAW_FEATURE_SQL = """
WITH base AS (
    SELECT
        t.nct_id,
        t.sponsor_key,
        t.phase,
        t.overall_status,
        t.enrollment_count,
        t.start_date,
        t.num_primary_outcomes,
        t.num_sites,
        t.has_results,
        t.allocation,
        t.masking,
        t.has_dmc,
        t.eligibility_criteria,
        (t.overall_status IN ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')) AS label
    FROM marts.fct_trials t
    WHERE t.phase = ANY(:phase_in)
      AND t.overall_status = ANY(:status_in)
      AND t.start_date >= :start_date_min
      AND t.start_date <= :start_date_max
      AND t.phase IS NOT NULL
),
-- LOCKED cross-project tie-break rule: one condition per trial, deterministic.
primary_condition AS (
    SELECT nct_id, condition_key
    FROM (
        SELECT
            b.nct_id,
            b.condition_key,
            ROW_NUMBER() OVER (PARTITION BY b.nct_id ORDER BY b.condition_key ASC) AS rn
        FROM marts.bridge_trial_condition b
    ) ranked
    WHERE rn = 1
),
-- Point-in-time sponsor history: only trials started strictly before this one.
sponsor_history AS (
    SELECT
        t.nct_id,
        COUNT(hist.nct_id) AS sponsor_prior_trial_count,
        AVG(
            CASE WHEN hist.overall_status IN ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')
                 THEN 1.0 ELSE 0.0 END
        ) AS sponsor_prior_termination_rate
    FROM base t
    LEFT JOIN marts.fct_trials hist
        ON hist.sponsor_key = t.sponsor_key
        AND hist.start_date < t.start_date
    GROUP BY t.nct_id
),
-- Point-in-time condition rarity, computed on each trial's PRIMARY (tie-broken)
-- condition to avoid the bridge-table fan-out warned about in the spec.
condition_rarity AS (
    SELECT
        t.nct_id,
        COUNT(hist.nct_id) AS condition_rarity
    FROM base t
    JOIN primary_condition pc ON pc.nct_id = t.nct_id
    LEFT JOIN primary_condition hist_pc ON hist_pc.condition_key = pc.condition_key
    LEFT JOIN marts.fct_trials hist
        ON hist.nct_id = hist_pc.nct_id
        AND hist.start_date < t.start_date
    GROUP BY t.nct_id
)
SELECT
    b.nct_id,
    b.phase,
    b.overall_status,
    b.enrollment_count,
    b.start_date,
    b.num_primary_outcomes,
    b.num_sites,
    b.has_results,
    b.allocation,
    b.masking,
    b.has_dmc,
    b.eligibility_criteria,
    b.label,
    cond.condition_name,
    s.sponsor_class,
    sh.sponsor_prior_trial_count,
    sh.sponsor_prior_termination_rate,
    cr.condition_rarity
FROM base b
LEFT JOIN primary_condition pc ON pc.nct_id = b.nct_id
LEFT JOIN marts.dim_condition cond ON cond.condition_key = pc.condition_key
LEFT JOIN marts.dim_sponsor s ON s.sponsor_key = b.sponsor_key
LEFT JOIN sponsor_history sh ON sh.nct_id = b.nct_id
LEFT JOIN condition_rarity cr ON cr.nct_id = b.nct_id
"""

_EXCLUSION_HEADER_RE = re.compile(r"exclusion criteria", re.IGNORECASE)
_BULLET_ITEM_RE = re.compile(r"(?m)^\s*(?:[*\-•]|\d+[.)])\s+")


def _load_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Purpose: Load the pharma domain config (db params, split dates, feature
        groups, missingness policy, dropped features).
    Leakage guard: N/A.
    Failure mode: A missing/malformed key here surfaces as a KeyError at
        build time rather than a silent wrong default -- deliberate, since a
        silently wrong split date or filter is worse than a crash.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_engine(config: dict) -> Engine:
    """
    Purpose: Build a SQLAlchemy engine from POSTGRES_* environment variables
        named in config['db'] (never from hardcoded credentials).
    Leakage guard: N/A.
    Failure mode: If env vars are unset, connection fails loudly at engine
        use time (fail fast rather than connect to an unintended database).
    """
    load_dotenv(REPO_ROOT / ".env")
    db_cfg = config["db"]
    host = os.environ[db_cfg["host_env"]]
    port = os.environ[db_cfg["port_env"]]
    dbname = os.environ[db_cfg["dbname_env"]]
    user = os.environ[db_cfg["user_env"]]
    password = os.environ[db_cfg["password_env"]]
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url)


def feature_pipeline_version() -> str:
    """
    Purpose: Return the git commit hash of this file at call time, logged as
        an MLflow tag on every training run (M2+).
    Leakage guard: N/A.
    Failure mode: If this is wrong or stale, a promotion-time version-mismatch
        check (M7) cannot detect that a candidate model was trained on
        different feature logic than the current Production model -- a real
        silent-feature-drift incident waiting to happen.
    """
    try:
        # NOTE (M5 bugfix): `git rev-parse HEAD -- <path>` does NOT filter by
        # path -- rev-parse's job is argument disambiguation, so it just
        # echoes the path back as a second/third output line ("HEAD's hash",
        # "--", "<path>") rather than restricting the hash to the last commit
        # that touched <path>. That polluted every tag this function set with
        # trailing "--\n<path>" garbage the moment the repo had a real commit
        # (silent while the repo had zero commits, since the fallback
        # "unknown" masked it). `git rev-list -1 HEAD -- <path>` is the
        # correct incantation for "the hash of the last commit touching this
        # path".
        result = subprocess.run(
            ["git", "rev-list", "-1", "HEAD", "--", str(Path(__file__).relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "no-commits-yet"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class PharmaDatasetBuilder(TemporalDatasetBuilder):
    """
    Purpose: Concrete TrialOutcome dataset builder -- pulls Phase 2/3
        interventional trials from PharmaPulse marts, engineers the design /
        sponsor-history / condition / temporal / text-lite feature groups,
        and writes the result to ml.training_dataset.
    Leakage guard: Owns every point-in-time join (sponsor history, condition
        rarity) and the label definition, so leakage-sensitive logic lives in
        exactly one place. M9-1: also owns the decision to exclude
        enrollment_count -- the point-in-time joins guard row placement, but
        feature *semantics* ("is this value knowable at start_date?") is a
        separate question this class is the only place to answer.
    Failure mode: N/A (class-level).
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = _load_config(config_path)
        self.engine = _get_engine(self.config)

    def fetch_raw(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Purpose: Execute _RAW_FEATURE_SQL against marts and return one row per
            trial with all raw feature-group columns plus the derived label.
            Caches to a local parquet file since `marts` has zero indexes and
            this query's correlated self-joins take several minutes -- without
            caching, every notebook re-run re-pays that cost.
        Leakage guard: The sponsor-history and condition-rarity CTEs only join
            historical trials with start_date < this trial's start_date --
            the point-in-time discipline this whole project is built around.
        Failure mode: If the primary_condition CTE's tie-break rule changes
            or the join to it is dropped, rows fan out and enrollment/count
            aggregates and split proportions silently corrupt. If the cache
            file goes stale relative to the warehouse (e.g. marts rebuilt),
            callers must pass use_cache=False to force a refresh -- this
            method does not detect staleness on its own.
        """
        cache_path = REPO_ROOT / "data" / "raw_trials_cache.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        filters = self.config["filters"]
        params = {
            "phase_in": filters["phase_in"],
            "status_in": filters["overall_status_in"],
            "start_date_min": filters["start_date_min"],
            "start_date_max": pd.Timestamp.today().date().isoformat(),
        }
        with self.engine.connect() as conn:
            df = pd.read_sql(text(_RAW_FEATURE_SQL), conn, params=params)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        return df

    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose: Turn raw fetch_raw() rows into model-ready features: temporal
            parts, text-lite eligibility features, missingness
            imputation/sentinels per config.yaml's missingness_policy. Leaves
            condition_name as a raw categorical -- one-hot encoding happens
            after temporal_split so the top-N vocabulary is fit on the train
            split only (see _one_hot_condition).
        Leakage guard: Null-rate printing (stdout) makes imputation decisions
            visible/auditable rather than silently smoothing over missingness
            that might itself be informative (e.g. a trial too new to have a
            site count yet). M9-1: enrollment_count is deliberately NOT
            engineered here -- a missingness indicator on it encoded
            post-hoc record state, which is why the M9 review's fix path A
            was rejected on evidence.
        Failure mode: If the missingness policy here drifts from
            config.yaml's documented policy, the config file becomes
            misleading documentation instead of a source of truth.
        """
        df = raw.copy()
        n = len(df)
        policy = self.config["missingness_policy"]

        print(f"[build_features] {n} rows after filters. Null rates:")
        for col in [
            "enrollment_count",
            "num_primary_outcomes",
            "num_sites",
            "allocation",
            "masking",
            "has_dmc",
            "eligibility_criteria",
            "condition_name",
            "sponsor_class",
            "sponsor_prior_termination_rate",
        ]:
            rate = df[col].isna().mean()
            print(f"  {col}: {rate:.4f}")

        # phase NULL -> drop row (already excluded by fetch_raw's WHERE clause,
        # re-asserted here so build_features is safe to call standalone).
        df = df.dropna(subset=["phase"])

        # --- Design group -----------------------------------------------
        # M9-1: enrollment_count is NOT engineered into a feature any more --
        # it is target leakage (see the class docstring and config.yaml's
        # dropped_features). The raw column is still fetched and carried
        # through so notebooks/01_dataset_audit.ipynb and docs/error_analysis.md
        # can show the leak, but nothing downstream of feature_columns() sees
        # it. Do not re-derive log_enrollment_count/enrollment_missing here
        # without reading decisions.md M9-1 first.
        df["num_primary_outcomes"] = df["num_primary_outcomes"].fillna(
            df["num_primary_outcomes"].median()
        )
        df["num_sites"] = df["num_sites"].fillna(df["num_sites"].median())
        df["has_results"] = df["has_results"].fillna(False)
        df["allocation"] = df["allocation"].fillna(policy["allocation"]["sentinel_value"])
        df["masking"] = df["masking"].fillna(policy["masking"]["sentinel_value"])
        df["has_dmc_str"] = df["has_dmc"].map(
            {True: "true", False: "false"}
        ).fillna(policy["has_dmc"]["sentinel_value"])

        # --- Text-lite group ----------------------------------------------
        df["eligibility_criteria"] = df["eligibility_criteria"].fillna("")
        df["eligibility_criteria_length"] = df["eligibility_criteria"].str.split().str.len()
        df["exclusion_keyword_count"] = df["eligibility_criteria"].apply(_count_exclusion_items)

        # --- Sponsor history group -----------------------------------------
        df["sponsor_prior_trial_count"] = df["sponsor_prior_trial_count"].fillna(0).astype(int)
        df["sponsor_prior_termination_rate"] = df["sponsor_prior_termination_rate"].fillna(
            df["sponsor_prior_termination_rate"].median()
        )
        df["sponsor_class"] = df["sponsor_class"].fillna(policy["sponsor_class"]["sentinel_value"])

        # --- Condition group (raw; one-hot happens post-split) --------------
        df["condition_name"] = df["condition_name"].fillna(policy["condition_name"]["sentinel_value"])
        df["condition_rarity"] = df["condition_rarity"].fillna(0).astype(int)

        # --- Temporal group ---------------------------------------------
        start = pd.to_datetime(df["start_date"])
        df["start_year"] = start.dt.year
        df["start_quarter"] = start.dt.quarter

        return df

    def _one_hot_condition(self, df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
        """
        Purpose: One-hot encode condition_name into the top-N most frequent
            categories (fit on the train split only) plus 'other' and
            'unknown', then apply that fixed vocabulary to calib/test rows.
        Leakage guard: Fitting the top-N vocabulary on the full dataset
            (including calib/test) would let future condition popularity
            influence which categories exist as features -- a subtle
            leakage class beyond the row-level split. Fitting on train only
            avoids it.
        Failure mode: If vocabulary is fit on the full dataset instead, a
            condition that only becomes common in the test period could
            get its own one-hot column, which is information a real
            production model deployed before that period could not have had.
        """
        top_n = self.config["condition_one_hot"]["top_n"]
        train_mask = df[split_col] == "train"
        top_conditions = (
            df.loc[train_mask, "condition_name"]
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
        bucketed = df["condition_name"].where(
            df["condition_name"].isin(top_conditions) | (df["condition_name"] == "unknown"),
            other="other",
        )
        dummies = pd.get_dummies(bucketed, prefix="condition")
        out = pd.concat([df, dummies], axis=1)
        return out, dummies.columns.tolist()

    def feature_columns(self, one_hot_condition_cols: list[str]) -> list[str]:
        """
        Purpose: Single source of truth for which engineered columns are fed
            to the model / written into the JSONB features blob.
        Leakage guard: N/A.
        Failure mode: If this list omits a column build_features() creates,
            that feature silently never reaches training despite the null-rate
            audit implying it was handled.
        """
        return [
            "phase",
            "num_primary_outcomes",
            "num_sites",
            "has_results",
            "allocation",
            "masking",
            "has_dmc_str",
            "eligibility_criteria_length",
            "exclusion_keyword_count",
            "sponsor_prior_trial_count",
            "sponsor_prior_termination_rate",
            "sponsor_class",
            "condition_rarity",
            "start_year",
            "start_quarter",
        ] + one_hot_condition_cols

    def run(self) -> pd.DataFrame:
        """
        Purpose: End-to-end M1 build: fetch -> engineer features -> temporal
            split -> one-hot condition (train-fit) -> write to
            ml.training_dataset. Returns the final DataFrame for inspection.
        Leakage guard: Runs temporal_split (not random_split) -- the dataset
            actually persisted for training is always the leakage-safe split.
        Failure mode: If this is ever swapped to call random_split, the
            written ml.training_dataset silently becomes leakage-inflated,
            and every downstream model trained from it inherits the bug.
        """
        raw = self.fetch_raw()
        feat = self.build_features(raw)

        split_cfg = self.config["split"]
        dates = SplitDates(
            train_end=pd.Timestamp(split_cfg["train_end"]),
            calib_end=pd.Timestamp(split_cfg["calib_end"]),
        )
        split_df = self.temporal_split(feat, date_col="start_date", split_dates=dates)
        split_df, one_hot_cols = self._one_hot_condition(split_df)

        print("Split counts:\n", split_df["split"].value_counts())
        print(
            "Class balance overall: "
            f"{split_df['label'].mean():.4f} positive (is_terminated-equivalent)"
        )

        feature_cols = self.feature_columns(one_hot_cols)
        n_written = self.write_to_db(
            split_df,
            engine=self.engine,
            schema=self.config["db"]["write_schema"],
            table="training_dataset",
            pk_col="nct_id",
            feature_cols=feature_cols,
            label_col="label",
            split_col="split",
        )
        print(f"Wrote {n_written} rows to ml.training_dataset.")
        return split_df


def _count_exclusion_items(text_value: str) -> int:
    """
    Purpose: Count bullet/numbered items appearing after the "Exclusion
        Criteria" section header in a trial's eligibility_criteria text --
        the exclusion_keyword_count text-lite feature.
    Leakage guard: N/A -- eligibility_criteria is set at trial registration,
        available at trial start.
    Failure mode: If a trial's criteria text doesn't use the standard CT.gov
        "Exclusion Criteria:" header (free-text variance), this returns 0
        rather than raising -- undercounts rather than crashing the build.
    """
    match = _EXCLUSION_HEADER_RE.search(text_value)
    if not match:
        return 0
    tail = text_value[match.end():]
    return len(_BULLET_ITEM_RE.findall(tail))


if __name__ == "__main__":
    builder = PharmaDatasetBuilder()
    builder.run()
