"""
Append-only decision log manager.

The decision log is the audit backbone for the endpoint matching pipeline.
Pipeline writes append new rows. Human review updates are restricted to the
`human_*` fields for an existing pair, preserving Layer 1 and Layer 2 outputs.
"""

from __future__ import annotations

import csv
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import portalocker

from src.models.schemas import DecisionLogEntry, HumanDecision, HumanReviewStatus, SwitchType
from src.pipeline.config import DECISION_LOG_PATH, PIPELINE_VERSION
from src.pipeline.validation import select_spot_check_pairs

logger = logging.getLogger(__name__)

_COLUMNS: list[str] = [
    "pair_id",
    "registered_endpoint",
    "published_endpoint",
    "similarity_score",
    "ssi",
    "routing",
    "llm_model",
    "llm_switch_type",
    "llm_reasoning",
    "llm_confidence",
    "llm_comparability",
    "llm_flag",
    "human_reviewed",
    "human_decision",
    "human_final_class",
    "human_poolable",
    "override_reason",
    "reviewer_initials",
    "review_timestamp",
    "pipeline_version",
    "created_at",
]


class DecisionLog:
    """
    CSV-backed audit log with a cross-platform lock file for write operations.

    Usage
    -----
    log = DecisionLog()
    log.append(entry)
    df = log.read()
    log.record_human_review(pair_id, ...)
    """

    def __init__(self, path: Path = DECISION_LOG_PATH) -> None:
        self.path = path
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._initialise_csv()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, entry: DecisionLogEntry) -> None:
        """Append a single pipeline-generated entry to the log."""
        row = self._entry_to_row(entry)
        with self._locked():
            if not self.path.exists():
                self._initialise_csv()
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
                writer.writerow(row)
        logger.debug("DecisionLog: appended pair_id=%s", entry.pair_id)

    def read(self) -> pd.DataFrame:
        """Return the full decision log as a DataFrame."""
        if not self.path.exists():
            self._initialise_csv()
        return pd.read_csv(self.path, dtype=str, keep_default_na=False)

    def record_human_review(
        self,
        pair_id: str,
        human_decision: HumanDecision,
        human_final_class: SwitchType,
        human_poolable: bool,
        reviewer_initials: str,
        override_reason: Optional[str] = None,
        review_status: HumanReviewStatus = HumanReviewStatus.YES,
    ) -> None:
        """
        Update the human review columns for an existing pair_id.

        This is the only permitted in-place mutation of the log. Layer 1 and
        Layer 2 columns are never touched.
        """
        if human_decision == HumanDecision.OVERRIDE and not override_reason:
            raise ValueError("override_reason is mandatory for override decisions")

        reviewer_initials = reviewer_initials.strip().upper()
        if not reviewer_initials:
            raise ValueError("reviewer_initials must not be empty")

        with self._locked():
            df = self.read()
            if pair_id not in df["pair_id"].values:
                raise KeyError(f"pair_id {pair_id!r} not found in decision log")

            idx = df.index[df["pair_id"] == pair_id]
            df.loc[idx, "human_reviewed"] = review_status.value
            df.loc[idx, "human_decision"] = human_decision.value
            df.loc[idx, "human_final_class"] = human_final_class.value
            df.loc[idx, "human_poolable"] = str(human_poolable)
            df.loc[idx, "override_reason"] = override_reason or ""
            df.loc[idx, "reviewer_initials"] = reviewer_initials
            df.loc[idx, "review_timestamp"] = datetime.now(UTC).isoformat()
            df.to_csv(self.path, index=False)

        logger.info(
            "DecisionLog: human review recorded pair_id=%s decision=%s",
            pair_id,
            human_decision.value,
        )

    def pending_review(self) -> pd.DataFrame:
        """Return rows that require human review but have not yet been actioned."""
        df = self.read()
        spot_check_pairs = select_spot_check_pairs(df)
        needs_review = (
            (df["routing"] == "llm")
            | (df["llm_confidence"].str.lower() == "low")
            | (df["llm_flag"].str.lower().isin(["true", "yes", "1"]))
            | (df["published_endpoint"] == "")
            | (df["pair_id"].isin(spot_check_pairs))
        )
        not_yet_reviewed = df["human_reviewed"] == "no"
        return df[needs_review & not_yet_reviewed].reset_index(drop=True)

    def governance_summary(self) -> dict:
        """Compute the governance metrics required for the methods paper."""
        df = self.read()
        if df.empty:
            return {}

        total = len(df)
        reviewed = df[df["human_reviewed"].isin(["yes", "spot_check"])]
        overrides = reviewed[reviewed["human_decision"] == "override"]

        routing_counts = df["routing"].value_counts(normalize=True).mul(100).round(1)
        llm_conf_counts = (
            df[df["llm_confidence"].fillna("").ne("")]["llm_confidence"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
        )

        return {
            "total_pairs": total,
            "routing_pct": routing_counts.to_dict(),
            "llm_call_rate_pct": round(
                len(df[df["routing"] == "llm"]) / total * 100, 1
            ),
            "llm_confidence_distribution_pct": llm_conf_counts.to_dict(),
            "human_review_rate_pct": round(len(reviewed) / total * 100, 1),
            "human_override_rate_pct": round(
                len(overrides) / max(len(reviewed), 1) * 100, 1
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise_csv(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            writer.writeheader()
        logger.info("DecisionLog: initialised new log at %s", self.path)

    def _entry_to_row(self, entry: DecisionLogEntry) -> dict:
        data = entry.model_dump()
        data["pipeline_version"] = PIPELINE_VERSION
        data["created_at"] = datetime.now(UTC).isoformat()
        for key, value in data.items():
            if hasattr(value, "value"):
                data[key] = value.value
            elif isinstance(value, bool):
                data[key] = str(value)
            elif value is None:
                data[key] = ""
        return {column: data.get(column, "") for column in _COLUMNS}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(self.lock_path, mode="a", timeout=10):
            yield
