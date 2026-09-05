from __future__ import annotations

import pandas as pd

from src.dashboard import helpers


def _frame(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "pair_id",
        "llm_switch_type",
        "llm_confidence",
        "llm_confidence_score",
        "llm_flag",
        "published_endpoint",
        "human_reviewed",
        "similarity_score",
    ]
    frame = pd.DataFrame(rows)
    for col in cols:
        if col not in frame.columns:
            frame[col] = ""
    return frame


def test_pending_review_rows_includes_missing_published_endpoint() -> None:
    frame = _frame(
        [
            {
                "pair_id": "NCT001_111",
                "llm_switch_type": "concordant",
                "published_endpoint": "",
                "human_reviewed": "no",
            }
        ]
    )

    queue = helpers.pending_review_rows(frame)

    assert list(queue["pair_id"]) == ["NCT001_111"]


def test_pending_review_rows_includes_every_outcome_switch(monkeypatch) -> None:
    from src.pipeline import validation

    monkeypatch.setattr(validation, "select_spot_check_pairs", lambda frame, **k: set())
    frame = _frame(
        [
            {
                "pair_id": "NCT002_222",
                "llm_switch_type": "major_switch",
                "llm_confidence": "high",
                "published_endpoint": "All-cause mortality",
                "human_reviewed": "no",
            },
            {
                "pair_id": "NCT003_333",
                "llm_switch_type": "concordant",
                "llm_confidence": "high",
                "published_endpoint": "All-cause mortality",
                "human_reviewed": "auto_accepted",
            },
        ]
    )

    queue = helpers.pending_review_rows(frame)

    assert list(queue["pair_id"]) == ["NCT002_222", "NCT003_333"]


def test_pending_review_rows_includes_spot_check_pairs(monkeypatch) -> None:
    monkeypatch.setattr(helpers, "pairs_needing_human_review", lambda frame: {"NCT004_444"})
    frame = _frame(
        [
            {
                "pair_id": "NCT004_444",
                "llm_switch_type": "concordant",
                "published_endpoint": "All-cause mortality",
                "human_reviewed": "auto_accepted",
            }
        ]
    )

    queue = helpers.pending_review_rows(frame)

    assert list(queue["pair_id"]) == ["NCT004_444"]
