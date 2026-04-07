from __future__ import annotations

import pandas as pd

from src.dashboard import helpers


def test_pending_review_rows_includes_missing_published_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(helpers, "select_spot_check_pairs", lambda frame: set())
    frame = pd.DataFrame(
        [
            {
                "pair_id": "NCT001_111",
                "routing": "auto_major_switch",
                "llm_confidence": "",
                "llm_flag": "",
                "published_endpoint": "",
                "human_reviewed": "no",
                "similarity_score": "0.10",
            }
        ]
    )

    queue = helpers.pending_review_rows(frame)

    assert list(queue["pair_id"]) == ["NCT001_111"]


def test_pending_review_rows_includes_spot_check_pairs(monkeypatch) -> None:
    monkeypatch.setattr(helpers, "select_spot_check_pairs", lambda frame: {"NCT002_222"})
    frame = pd.DataFrame(
        [
            {
                "pair_id": "NCT002_222",
                "routing": "auto_concordant",
                "llm_confidence": "",
                "llm_flag": "",
                "published_endpoint": "All-cause mortality",
                "human_reviewed": "no",
                "similarity_score": "0.97",
            }
        ]
    )

    queue = helpers.pending_review_rows(frame)

    assert list(queue["pair_id"]) == ["NCT002_222"]
