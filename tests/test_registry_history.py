from __future__ import annotations

import pandas as pd
import requests

from src.pipeline.registry_history import HistoryVersion, build_registry_history


class FakeHistoryClient:
    def list_versions(self, nct_id: str) -> list[HistoryVersion]:
        return [
            HistoryVersion("1", pd.Timestamp("2019-01-01")),
            HistoryVersion("2", pd.Timestamp("2020-01-01")),
            HistoryVersion("3", pd.Timestamp("2022-01-01")),
            HistoryVersion("4", pd.Timestamp("2024-01-01")),
        ]

    def fetch_version(self, nct_id: str, version: str) -> dict:
        endpoints = {
            "1": "Original primary endpoint",
            "2": "Pre-recruitment endpoint",
            "3": "Pre-completion endpoint",
            "4": "Pre-publication endpoint",
        }
        return {
            "protocolSection": {
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": endpoints[version], "timeFrame": "24 months"}]
                }
            }
        }


def test_registry_history_selects_pre_recruitment_endpoint_for_comparison() -> None:
    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT00000001",
                "start_date": "2020-06-01",
                "primary_completion_date": "2023-06-01",
                "pub_date": "2024-06-01",
                "primary_outcomes": "Current edited endpoint",
            }
        ]
    )

    enriched, audit = build_registry_history(
        linked,
        FakeHistoryClient(),  # type: ignore[arg-type]
    )
    row = enriched.iloc[0]

    assert row["registry_endpoint_source"] == "pre_recruitment"
    assert row["registry_history_status"] == "HISTORICAL_VERSIONS_AVAILABLE"
    assert bool(row["endpoint_switch_assessable"]) is True
    assert int(row["registry_versions_retrieved"]) == 4
    assert row["registered_primary_outcomes_for_comparison"] == (
        "Pre-recruitment endpoint [Time Frame: 24 months]"
    )
    assert set(audit["snapshot_type"]) == {
        "original",
        "pre_recruitment",
        "pre_primary_completion",
        "pre_publication",
        "current",
    }


class UnavailableHistoryClient:
    def list_versions(self, nct_id: str) -> list[HistoryVersion]:
        raise requests.HTTPError("history unavailable")


def test_registry_history_falls_back_to_current_endpoint_but_marks_not_assessable() -> None:
    linked = pd.DataFrame(
        [{"nct_id": "NCT00000001", "primary_outcomes": "Current registered endpoint"}]
    )

    enriched, _ = build_registry_history(
        linked,
        UnavailableHistoryClient(),  # type: ignore[arg-type]
    )
    row = enriched.iloc[0]

    # Comparison still runs — against the current endpoint — but the fact that a
    # quiet registry edit can't be ruled out is recorded, not hidden.
    assert row["registered_primary_outcomes_current"] == "Current registered endpoint"
    assert row["registered_primary_outcomes_for_comparison"] == "Current registered endpoint"
    assert row["registry_endpoint_source"] == "current"
    assert row["registry_history_status"] == "CURRENT_ONLY_HISTORY_UNAVAILABLE"
    assert bool(row["endpoint_switch_assessable"]) is False
