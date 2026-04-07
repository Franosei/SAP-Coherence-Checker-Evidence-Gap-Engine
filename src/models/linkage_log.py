"""
Append-only linkage audit log manager.

Each linkage decision is written to a CSV audit log. Writes are protected with
the same cross-platform lock strategy used by the decision log so Windows and
multi-process runs behave consistently.
"""

from __future__ import annotations

import csv
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import portalocker

from src.models.schemas import LinkageAuditEntry, LinkageConfidence
from src.pipeline.config import LINKAGE_LOG_PATH, PIPELINE_VERSION

logger = logging.getLogger(__name__)

_COLUMNS: list[str] = [
    "nct_id",
    "pmid",
    "linkage_method",
    "linkage_confidence",
    "timestamp",
    "linked_by",
    "notes",
    "pipeline_version",
]


class LinkageLog:
    """CSV-backed append-only linkage audit log."""

    def __init__(self, path: Path = LINKAGE_LOG_PATH) -> None:
        self.path = path
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._initialise_csv()

    def append(self, entry: LinkageAuditEntry) -> None:
        row = self._entry_to_row(entry)
        with self._locked():
            if not self.path.exists():
                self._initialise_csv()
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
                writer.writerow(row)
        logger.debug(
            "LinkageLog: appended nct_id=%s confidence=%s",
            entry.nct_id,
            entry.linkage_confidence.value,
        )

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            self._initialise_csv()
        return pd.read_csv(self.path, dtype=str, keep_default_na=False)

    def flagged_for_review(self) -> pd.DataFrame:
        """Return entries that require human review (Low or Unlinked confidence)."""
        df = self.read()
        return df[
            df["linkage_confidence"].isin(
                [LinkageConfidence.LOW.value, LinkageConfidence.UNLINKED.value]
            )
        ].reset_index(drop=True)

    def confidence_summary(self) -> dict:
        """
        Return a governance summary of linkage confidence distribution.

        Returns
        -------
        dict
            Keys: ``total``, ``high``, ``medium``, ``low``, ``unlinked``,
            ``flagged_for_review``, ``linkage_method_counts``.
        """
        df = self.read()
        if df.empty:
            return {"total": 0}

        total    = len(df)
        conf_cts = df["linkage_confidence"].value_counts().to_dict()
        meth_cts = df["linkage_method"].value_counts().to_dict()
        flagged  = df["linkage_confidence"].isin(
            [LinkageConfidence.LOW.value, LinkageConfidence.UNLINKED.value]
        ).sum()

        return {
            "total":                total,
            "high":                 conf_cts.get("High", 0),
            "medium":               conf_cts.get("Medium", 0),
            "low":                  conf_cts.get("Low", 0),
            "unlinked":             conf_cts.get("Unlinked", 0),
            "flagged_for_review":   int(flagged),
            "linkage_method_counts": meth_cts,
        }

    def _initialise_csv(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            writer.writeheader()
        logger.info("LinkageLog: initialised at %s", self.path)

    def _entry_to_row(self, entry: LinkageAuditEntry) -> dict:
        data = entry.model_dump()
        data["pipeline_version"] = PIPELINE_VERSION
        for key, value in data.items():
            if hasattr(value, "value"):
                data[key] = value.value
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif value is None:
                data[key] = ""
        return {column: data.get(column, "") for column in _COLUMNS}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(self.lock_path, mode="a", timeout=10):
            yield
