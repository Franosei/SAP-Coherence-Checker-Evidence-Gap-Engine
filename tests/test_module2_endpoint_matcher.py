from __future__ import annotations

import pandas as pd

from src.models.schemas import (
    EndpointRouting,
    LLMConfidence,
    LLMEndpointClassification,
    SwitchDirection,
    SwitchType,
)
from src.pipeline import module2_endpoint_matcher as matcher


class _FakeDecisionLog:
    def __init__(self) -> None:
        self.entries = []

    def append(self, entry) -> None:
        self.entries.append(entry)

    def governance_summary(self) -> dict:
        return {"total_pairs": len(self.entries)}


def test_route_from_score_matches_proposal_thresholds() -> None:
    assert matcher._route_from_score(0.90) == EndpointRouting.AUTO_CONCORDANT
    assert matcher._route_from_score(0.89) == EndpointRouting.LLM
    assert matcher._route_from_score(0.50) == EndpointRouting.LLM
    assert matcher._route_from_score(0.49) == EndpointRouting.AUTO_MAJOR_SWITCH


def test_run_endpoint_matching_uses_published_endpoint_and_linkage_gate(monkeypatch) -> None:
    fake_log = _FakeDecisionLog()
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.72])

    llm_result = LLMEndpointClassification(
        switch_type=SwitchType.MINOR_MODIFICATION,
        direction=SwitchDirection.TIMEFRAME_CHANGED,
        step_by_step_reasoning="The endpoint concept is the same, but the reported paper changed the follow-up window.",
        confidence=LLMConfidence.HIGH,
        comparability_for_pooling=True,
        flag_for_human_review=False,
        key_differences=["follow-up window changed"],
    )
    monkeypatch.setattr(matcher, "_call_llm", lambda registered, published: llm_result)

    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT001",
                "pmid": "111",
                "linkage_confidence": "High",
                "primary_outcomes": "CV death or HF admission at 12 months",
                "published_endpoint": "CV death or HF admission at 9 months",
            },
            {
                "nct_id": "NCT002",
                "pmid": "222",
                "linkage_confidence": "Low",
                "primary_outcomes": "Should be skipped",
                "published_endpoint": "Should be skipped",
            },
            {
                "nct_id": "NCT003",
                "pmid": "",
                "linkage_confidence": "High",
                "primary_outcomes": "Missing PMID should skip",
                "published_endpoint": "Missing PMID should skip",
            },
        ]
    )

    result = matcher.run_endpoint_matching(linked)

    assert len(fake_log.entries) == 1
    entry = fake_log.entries[0]
    assert entry.pair_id == "NCT001_111"
    assert entry.published_endpoint == "CV death or HF admission at 9 months"
    assert entry.routing == EndpointRouting.LLM
    assert entry.llm_switch_type == SwitchType.MINOR_MODIFICATION

    processed = result[result["nct_id"] == "NCT001"].iloc[0]
    skipped_low = result[result["nct_id"] == "NCT002"].iloc[0]
    skipped_missing = result[result["nct_id"] == "NCT003"].iloc[0]

    assert processed["pair_id"] == "NCT001_111"
    assert processed["routing"] == "llm"
    assert pd.isna(skipped_low["pair_id"])
    assert pd.isna(skipped_missing["pair_id"])
