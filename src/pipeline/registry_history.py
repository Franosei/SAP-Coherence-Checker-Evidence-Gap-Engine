"""Retrieve historical ClinicalTrials.gov endpoint snapshots conservatively.

The public API v2 exposes the current record only. ClinicalTrials.gov's web
application uses an undocumented internal history endpoint; this module treats
that endpoint as optional and fails closed. If history cannot be retrieved, the
current endpoint is retained for audit, but is not used for coherence scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HISTORY_INDEX = "https://clinicaltrials.gov/api/int/studies/{nct_id}?history=true"
_HISTORY_VERSION = "https://clinicaltrials.gov/api/int/studies/{nct_id}/history/{version}"
_HISTORY_PAGE = "https://clinicaltrials.gov/study/{nct_id}?tab=history"


class RegistryHistoryUnavailable(RuntimeError):
    """Raised when the optional internal history service cannot be used."""


@dataclass(frozen=True)
class HistoryVersion:
    version: str
    version_date: pd.Timestamp


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _date(value: object) -> Optional[pd.Timestamp]:
    text = _text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _boolean(value: object) -> bool:
    return _text(value).lower() in {"true", "1", "yes"}


def _format_primary_outcomes(study: dict[str, Any]) -> str:
    protocol = study.get("protocolSection", study.get("protocol", {}))
    outcomes = protocol.get("outcomesModule", {}).get("primaryOutcomes", [])
    values: list[str] = []
    for outcome in outcomes:
        measure = _text(outcome.get("measure"))
        timeframe = _text(outcome.get("timeFrame") or outcome.get("timeframe"))
        if not measure:
            continue
        values.append(f"{measure} [Time Frame: {timeframe}]" if timeframe else measure)
    return " | ".join(values)


def _study_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("study", "Study", "studyRecord"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _history_versions(payload: Any) -> list[HistoryVersion]:
    """Find version/date pairs while tolerating minor internal-schema changes."""
    found: dict[str, pd.Timestamp] = {}
    version_keys = ("version", "versionNumber", "versionNo", "versionId")
    date_keys = ("date", "versionDate", "submittedDate", "submissionDate")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            version = next((_text(value.get(key)) for key in version_keys if value.get(key)), "")
            version_date = next(
                (_date(value.get(key)) for key in date_keys if value.get(key)), None
            )
            if version and version_date is not None:
                found[version] = version_date
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return sorted(
        (HistoryVersion(version, version_date) for version, version_date in found.items()),
        key=lambda item: (item.version_date, int(item.version) if item.version.isdigit() else 0),
    )


class RegistryHistoryClient:
    """Small client for the history endpoint used by the CT.gov website."""

    def __init__(self, timeout_s: int = 30) -> None:
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "SAP-Coherence-Checker/4.0 (research; registry audit)",
                "Accept": "application/json",
            }
        )
        self._available: Optional[bool] = None

    def list_versions(self, nct_id: str) -> list[HistoryVersion]:
        if self._available is False:
            raise RegistryHistoryUnavailable("history service disabled after an earlier failure")
        response = self.session.get(_HISTORY_INDEX.format(nct_id=nct_id), timeout=self.timeout_s)
        try:
            response.raise_for_status()
            versions = _history_versions(response.json())
        except (requests.RequestException, ValueError) as exc:
            self._available = False
            raise RegistryHistoryUnavailable(str(exc)) from exc
        self._available = True
        return versions

    def fetch_version(self, nct_id: str, version: str) -> dict[str, Any]:
        response = self.session.get(
            _HISTORY_VERSION.format(nct_id=nct_id, version=version), timeout=self.timeout_s
        )
        response.raise_for_status()
        return _study_payload(response.json())


def _latest_before(
    versions: list[HistoryVersion], cutoff: Optional[pd.Timestamp]
) -> Optional[HistoryVersion]:
    if cutoff is None:
        return None
    eligible = [item for item in versions if item.version_date <= cutoff]
    return eligible[-1] if eligible else None


def _snapshot_row(
    nct_id: str,
    snapshot_type: str,
    version: str,
    version_date: str,
    endpoint: str,
    status: str,
) -> dict[str, object]:
    return {
        "nct_id": nct_id,
        "snapshot_type": snapshot_type,
        "version": version,
        "version_date": version_date,
        "primary_outcomes": endpoint,
        "history_status": status,
        "history_url": _HISTORY_PAGE.format(nct_id=nct_id),
    }


def build_registry_history(
    linked_df: pd.DataFrame,
    client: Optional[RegistryHistoryClient] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add prespecified endpoint fields and return a long snapshot audit table.

    The endpoint used for comparison is the latest version on/before recruitment
    start. If none exists, the original registry version is used. A current-only
    fallback is deliberately not scored because it could conceal outcome edits.
    """
    client = client or RegistryHistoryClient()
    enriched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for _, source_row in linked_df.iterrows():
        row = source_row.to_dict()
        nct_id = _text(row.get("nct_id")).upper()
        current_endpoint = _text(row.get("primary_outcomes"))
        history_url = _HISTORY_PAGE.format(nct_id=nct_id)
        snapshot_values = {
            "original": "",
            "pre_recruitment": "",
            "pre_primary_completion": "",
            "pre_publication": "",
            "current": current_endpoint,
        }
        status = "CURRENT_ONLY_HISTORY_UNAVAILABLE"
        versions_retrieved = 0

        try:
            versions = client.list_versions(nct_id)
            if not versions:
                raise ValueError("history endpoint returned no versions")
            versions_retrieved = len(versions)
            status = (
                "HISTORICAL_VERSIONS_AVAILABLE"
                if versions_retrieved >= 2
                else "SINGLE_REGISTERED_VERSION"
            )

            publication_date = _date(row.get("pub_date"))
            if publication_date is None:
                year = _text(row.get("pub_year"))
                publication_date = _date(f"{year}-01-01") if year.isdigit() else None

            selected = {
                "original": versions[0],
                "pre_recruitment": _latest_before(versions, _date(row.get("start_date"))),
                "pre_primary_completion": _latest_before(
                    versions, _date(row.get("primary_completion_date"))
                ),
                "pre_publication": _latest_before(versions, publication_date),
            }
            cache: dict[str, str] = {}
            for snapshot_type, item in selected.items():
                if item is None:
                    audit_rows.append(
                        _snapshot_row(nct_id, snapshot_type, "", "", "", "NO_VERSION_BEFORE_CUTOFF")
                    )
                    continue
                if item.version not in cache:
                    cache[item.version] = _format_primary_outcomes(
                        client.fetch_version(nct_id, item.version)
                    )
                endpoint = cache[item.version]
                snapshot_values[snapshot_type] = endpoint
                audit_rows.append(
                    _snapshot_row(
                        nct_id,
                        snapshot_type,
                        item.version,
                        item.version_date.date().isoformat(),
                        endpoint,
                        "AVAILABLE" if endpoint else "VERSION_HAS_NO_PRIMARY_OUTCOME",
                    )
                )
        except (
            requests.RequestException,
            RegistryHistoryUnavailable,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            status = "CURRENT_ONLY_HISTORY_UNAVAILABLE"
            logger.warning("Registry history unavailable for %s: %s", nct_id, exc)
            for snapshot_type in (
                "original",
                "pre_recruitment",
                "pre_primary_completion",
                "pre_publication",
            ):
                audit_rows.append(_snapshot_row(nct_id, snapshot_type, "", "", "", status))

        audit_rows.append(_snapshot_row(nct_id, "current", "current", "", current_endpoint, status))
        # Prefer a pre-recruitment / original snapshot; fall back to the current
        # registered endpoint (always available from the Module 1 fetch) so the
        # published-vs-registered comparison can still run when history is
        # unavailable.
        comparison_endpoint = (
            snapshot_values["pre_recruitment"]
            or snapshot_values["original"]
            or current_endpoint
        )
        comparison_source = (
            "pre_recruitment"
            if snapshot_values["pre_recruitment"]
            else (
                "original"
                if snapshot_values["original"]
                else ("current" if current_endpoint else "")
            )
        )
        # "assessable" == we hold a *pre-publication* registry snapshot, so a
        # quiet registry edit toward the published endpoint would be visible.
        # When the comparison falls back to the current endpoint this is False:
        # a "concordant" result then does not rule out registry-side editing.
        endpoint_switch_assessable = comparison_source in {"pre_recruitment", "original"}
        row.update(
            {
                "registered_primary_outcomes_original": snapshot_values["original"],
                "registered_primary_outcomes_pre_recruitment": snapshot_values["pre_recruitment"],
                "registered_primary_outcomes_pre_primary_completion": snapshot_values[
                    "pre_primary_completion"
                ],
                "registered_primary_outcomes_pre_publication": snapshot_values["pre_publication"],
                "registered_primary_outcomes_current": current_endpoint,
                "registered_primary_outcomes_for_comparison": comparison_endpoint,
                "registry_endpoint_source": comparison_source,
                "registry_history_status": status,
                "registry_versions_retrieved": versions_retrieved,
                "endpoint_switch_assessable": endpoint_switch_assessable,
                "registry_history_url": history_url,
                "registry_history_review_required": not bool(comparison_endpoint),
                "human_review_required": _boolean(
                    row.get("publication_family_human_review_required", False)
                )
                or not bool(comparison_endpoint),
            }
        )
        enriched_rows.append(row)

    enriched_df = pd.DataFrame(enriched_rows)
    _log_history_coverage(enriched_df)
    return enriched_df, pd.DataFrame(audit_rows)


def _log_history_coverage(enriched_df: pd.DataFrame) -> None:
    """Report how many trials can actually be assessed for a registry endpoint switch."""
    if enriched_df.empty or "registry_history_status" not in enriched_df.columns:
        return
    total = len(enriched_df)
    by_status = enriched_df["registry_history_status"].value_counts().to_dict()
    assessable = int(enriched_df.get("endpoint_switch_assessable", pd.Series(dtype=bool)).sum())
    logger.info(
        "Registry-history coverage — %d/%d trials assessable for an endpoint switch "
        "(%.0f%%). Status breakdown: %s",
        assessable,
        total,
        100.0 * assessable / total if total else 0.0,
        by_status,
    )
    if total and assessable / total < 0.5:
        logger.warning(
            "Registry history was retrievable for fewer than half of trials. "
            "Endpoint-switch detection from registry history is NOT broadly available "
            "on this dataset; report it as per-trial, not as a headline capability."
        )
