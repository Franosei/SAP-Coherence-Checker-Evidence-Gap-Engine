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
    monkeypatch.setattr(
        matcher,
        "_call_llm",
        lambda *a, **k: llm_result,
    )

    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT001",
                "pmid": "111",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "primary_outcomes": "Current endpoint must not be used",
                "registered_primary_outcomes_for_comparison": (
                    "CV death or HF admission at 12 months"
                ),
                "published_endpoint": "CV death or HF admission at 9 months",
            },
            {
                "nct_id": "NCT002",
                "pmid": "222",
                "linkage_confidence": "Unlinked",
                "primary_result_status": "NOT_FOUND",
                "primary_outcomes": "Should be skipped",
                "registered_primary_outcomes_for_comparison": "Should be skipped",
                "published_endpoint": "Should be skipped",
            },
            {
                "nct_id": "NCT003",
                "pmid": "",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "primary_outcomes": "Missing PMID should skip",
                "registered_primary_outcomes_for_comparison": "Missing PMID should skip",
                "published_endpoint": "Missing PMID should skip",
            },
        ]
    )

    result = matcher.run_endpoint_matching(linked)

    assert len(fake_log.entries) == 1
    entry = fake_log.entries[0]
    assert entry.pair_id == "NCT001_111"
    assert entry.registered_endpoint == "CV death or HF admission at 12 months"
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


def test_high_cosine_similarity_still_goes_to_the_llm(monkeypatch) -> None:
    """v4.1: cosine no longer auto-classifies anything — every pair is LLM-adjudicated."""
    fake_log = _FakeDecisionLog()
    calls: list[tuple] = []
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.985])

    def _fake_llm(*a, **k):
        calls.append(a)
        return LLMEndpointClassification(
            switch_type=SwitchType.CONCORDANT,
            direction=SwitchDirection.NONE,
            step_by_step_reasoning="Identical endpoint concept and timeframe in both sources.",
            confidence=LLMConfidence.HIGH,
            comparability_for_pooling=True,
            flag_for_human_review=False,
        )

    monkeypatch.setattr(matcher, "_call_llm", _fake_llm)

    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT010",
                "pmid": "999",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "registered_primary_outcomes_for_comparison": "Overall survival",
                "published_endpoint": "Overall survival",
            }
        ]
    )

    matcher.run_endpoint_matching(linked)

    assert len(calls) == 1  # LLM called despite 0.985 cosine
    assert calls[0][0] == ""  # current endpoint is not mislabelled as historical/original
    entry = fake_log.entries[0]
    assert entry.routing == EndpointRouting.LLM
    assert entry.llm_switch_type == SwitchType.CONCORDANT


def test_confident_switch_is_never_auto_accepted(monkeypatch) -> None:
    """An outcome SWITCH is the study's finding — a human must see every one,
    however confident and unflagged the LLM is."""
    from src.dashboard.helpers import pending_review_rows

    fake_log = _FakeDecisionLog()
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.3])
    monkeypatch.setattr(
        matcher,
        "_call_llm",
        lambda *a, **k: LLMEndpointClassification(
            switch_type=SwitchType.MAJOR_SWITCH,
            direction=SwitchDirection.ENDPOINT_REPLACED,
            step_by_step_reasoning="Registered primary OS was replaced by ORR; not disclosed as a change.",
            confidence=LLMConfidence.HIGH,
            confidence_score=0.92,
            comparability_for_pooling=False,
            flag_for_human_review=False,
        ),
    )
    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT030",
                "pmid": "777",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "registered_primary_outcomes_for_comparison": "Overall survival",
                "published_endpoint": "Objective response rate",
            }
        ]
    )
    result = matcher.run_endpoint_matching(linked)

    entry = fake_log.entries[0]
    assert entry.human_reviewed.value == "no"
    assert entry.human_final_class is None
    assert entry.llm_confidence_score == 0.92

    # And it DOES appear in the review queue.
    dl = pd.DataFrame([{**result.iloc[0].to_dict(), **entry.model_dump(mode="json")}])
    dl["llm_flag"] = "False"
    assert list(pending_review_rows(dl)["pair_id"]) == ["NCT030_777"]


def test_confident_non_switch_still_requires_review_during_recalibration(monkeypatch) -> None:
    from src.dashboard.helpers import pending_review_rows

    fake_log = _FakeDecisionLog()
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.95])
    monkeypatch.setattr(
        matcher,
        "_call_llm",
        lambda *a, **k: LLMEndpointClassification(
            switch_type=SwitchType.CONCORDANT,
            direction=SwitchDirection.NONE,
            step_by_step_reasoning="Same registered primary endpoint reported unchanged in the paper.",
            confidence=LLMConfidence.HIGH,
            confidence_score=0.93,
            comparability_for_pooling=True,
            flag_for_human_review=False,
        ),
    )
    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT030",
                "pmid": "777",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "registered_primary_outcomes_for_comparison": "Overall survival",
                "published_endpoint": "Overall survival",
            }
        ]
    )
    result = matcher.run_endpoint_matching(linked)

    entry = fake_log.entries[0]
    assert entry.human_reviewed.value == "no"
    assert entry.human_final_class is None
    assert entry.human_poolable is None

    dl = pd.DataFrame([{**result.iloc[0].to_dict(), **entry.model_dump(mode="json")}])
    dl["llm_flag"] = "False"
    assert list(pending_review_rows(dl)["pair_id"]) == ["NCT030_777"]


def test_low_confidence_verdict_still_goes_to_the_queue(monkeypatch) -> None:
    fake_log = _FakeDecisionLog()
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.6])
    monkeypatch.setattr(
        matcher,
        "_call_llm",
        lambda *a, **k: LLMEndpointClassification(
            switch_type=SwitchType.CONCORDANT,
            direction=SwitchDirection.NONE,
            step_by_step_reasoning="Same endpoint; the abstract merely omits the registered timeframe.",
            confidence=LLMConfidence.LOW,
            confidence_score=0.35,
            comparability_for_pooling=True,
            flag_for_human_review=True,
        ),
    )
    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT031",
                "pmid": "888",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "registered_primary_outcomes_for_comparison": "DFS at 5 years",
                "published_endpoint": "DFS",
            }
        ]
    )
    matcher.run_endpoint_matching(linked)
    entry = fake_log.entries[0]
    assert entry.human_reviewed.value == "no"
    assert entry.human_final_class is None


def test_disclosed_exploratory_addition_is_additional_outcome_not_a_switch(monkeypatch) -> None:
    fake_log = _FakeDecisionLog()
    monkeypatch.setattr(matcher, "DecisionLog", lambda: fake_log)
    monkeypatch.setattr(matcher, "_compute_similarity_scores", lambda reg, pub: [0.6])
    monkeypatch.setattr(
        matcher,
        "_call_llm",
        lambda *a, **k: LLMEndpointClassification(
            switch_type=SwitchType.ADDITIONAL_OUTCOME,
            direction=SwitchDirection.UNREGISTERED_ADDED,
            step_by_step_reasoning="Registered primary (PIS at week 6) is still reported as primary; "
            "immunoscore #2 is added but explicitly called an unplanned exploratory endpoint.",
            confidence=LLMConfidence.HIGH,
            comparability_for_pooling=True,
            flag_for_human_review=False,
            disclosed_as_exploratory=True,
            likely_results_driven=False,
            switch_forms=["unregistered_added"],
        ),
    )
    linked = pd.DataFrame(
        [
            {
                "nct_id": "NCT020",
                "pmid": "555",
                "linkage_confidence": "High",
                "primary_result_status": "SELECTED",
                "registered_primary_outcomes_for_comparison": "PIS change at week 6",
                "published_endpoint": "PIS change at week 6 (primary); immunoscore 2 (exploratory)",
            }
        ]
    )
    matcher.run_endpoint_matching(linked)
    entry = fake_log.entries[0]
    assert entry.llm_switch_type == SwitchType.ADDITIONAL_OUTCOME
    assert entry.llm_disclosed_exploratory is True
    assert entry.llm_switch_forms == "unregistered_added"


def test_adjudicator_json_is_mapped_onto_the_internal_schema(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _Msg:
        content = (
            '{"registered_primary_original":"Progression-free survival",'
            '"registered_primary_current":"Progression-free survival",'
            '"published_primary":"Progression-free survival",'
            '"primary_endpoint_match":"exact",'
            '"final_classification":"Concordant (no change)",'
            '"confidence":0.93,"secondary_outcomes_missing":["quality of life"],'
            '"additional_outcomes":[],"registry_history_relevant":false,'
            '"registry_history_interpretation":"",'
            '"treatment_comparison_changed":true,"endpoint_changed":false,'
            '"disclosure_status":"none needed",'
            '"evidence_registered":"PFS per registry",'
            '"evidence_published":"stratified HR for PFS, 0.78",'
            '"reasoning_summary":"The registered primary PFS is reported as the published primary; '
            'ORR and OS are secondary.","human_review_required":false}'
        )

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = None

    import types

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: _Resp()))
    )
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: fake_client, raising=False)

    result = matcher._call_llm("Progression-free survival", "stratified HR for PFS, 0.78")
    assert result is not None
    assert result.switch_type == SwitchType.CONCORDANT
    assert result.confidence_score == 0.93
    assert result.confidence == LLMConfidence.HIGH
    assert result.comparability_for_pooling is True
    assert result.flag_for_human_review is False
    assert "quality of life" in result.key_differences


def test_adjudicator_major_switch_requires_undisclosed_replacement(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _Msg:
        content = (
            '{"registered_primary_original":"Pathological complete response",'
            '"registered_primary_current":"Pathological complete response",'
            '"published_primary":"Event-free survival",'
            '"primary_endpoint_match":"different",'
            '"final_classification":"Outcome switch — major / undisclosed",'
            '"confidence":0.88,"secondary_outcomes_missing":[],"additional_outcomes":[],'
            '"registry_history_relevant":false,"registry_history_interpretation":"",'
            '"treatment_comparison_changed":false,"endpoint_changed":true,'
            '"actual_change_evidence":["Registry says pCR; publication explicitly names EFS primary"],'
            '"missing_detail_only":false,'
            '"disclosure_status":"undisclosed","evidence_registered":"pCR",'
            '"evidence_published":"EFS presented as primary","reasoning_summary":'
            '"pCR is absent; EFS is presented as the primary endpoint with no disclosure.",'
            '"human_review_required":false}'
        )

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = None

    import types

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: _Resp()))
    )
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: fake_client, raising=False)

    result = matcher._call_llm("Pathological complete response", "Event-free survival")
    assert result is not None
    assert result.switch_type == SwitchType.MAJOR_SWITCH
    assert result.likely_results_driven is True
    assert result.comparability_for_pooling is False


def test_missing_abstract_qualifiers_are_forced_to_concordant() -> None:
    parsed = {
        "registered_primary_original": "Confirmed overall response during monotherapy",
        "published_primary": "Objective response rate comprising CR + PR",
        "primary_endpoint_match": "clinically equivalent",
        "final_classification": "Outcome modification (timeframe/definition/population)",
        "confidence": 0.9,
        "endpoint_changed": False,
        "actual_change_evidence": [],
        "missing_detail_only": True,
        "disclosure_status": "none needed",
        "reasoning_summary": "The abstract omits confirmed and the registered timeframe.",
        "human_review_required": False,
    }

    result = matcher._adjudication_to_schema(parsed)

    assert result.switch_type == SwitchType.CONCORDANT
    assert result.flag_for_human_review is True
    assert result.adjudication_guardrail


def test_adjudicator_exposes_exactly_five_user_facing_classes() -> None:
    assert set(matcher._SWITCH_TYPE_TO_CLASSIFICATION.values()) == {
        "Concordant (no change)",
        "Additional outcome (disclosed exploratory)",
        "Outcome modification (timeframe/definition/population)",
        "Outcome switch — moderate / partly disclosed",
        "Outcome switch — major / undisclosed",
    }


def test_explicit_different_timepoint_is_outcome_modification() -> None:
    parsed = {
        "registered_primary_original": "5-year recurrence-free survival",
        "published_primary": "2-year recurrence-free survival",
        "primary_endpoint_match": "modified",
        "final_classification": "Outcome modification (timeframe/definition/population)",
        "confidence": 0.9,
        "endpoint_changed": True,
        "actual_change_evidence": ["Registry states 5 years; publication states 2 years"],
        "missing_detail_only": False,
        "disclosure_status": "undisclosed",
        "reasoning_summary": "The same endpoint has an explicitly different timepoint.",
        "human_review_required": False,
    }

    result = matcher._adjudication_to_schema(parsed)

    assert result.switch_type == SwitchType.MINOR_MODIFICATION
    assert result.adjudication_guardrail == ""


def test_structured_rules_check_construct_before_missing_timeframe() -> None:
    parsed = {
        "final_classification": "Outcome switch — major / undisclosed",
        "published_primary": "Median progression-free survival was 8.3 months",
        "confidence": 0.91,
        "disclosure_status": "undisclosed",
        "reasoning_summary": "Both sources report progression-free survival.",
        "endpoint_comparisons": [
            {
                "registered_endpoint": "Progression-free survival at 12 months",
                "published_corresponding_endpoint": "Median progression-free survival",
                "same_construct": True,
                "same_timeframe": None,
                "same_definition": None,
                "same_population": True,
                "same_measurement_method": None,
                "registered_endpoint_reported": True,
                "actual_change_evidence": [],
            }
        ],
    }

    result = matcher._adjudication_to_schema(parsed)

    assert result.switch_type == SwitchType.CONCORDANT
    assert result.primary_endpoint_match == "clinically equivalent"
    assert result.endpoint_comparisons[0].same_timeframe is None
    assert result.endpoint_comparisons[0].classification_label == "Concordant (no change)"


def test_structured_rules_identify_undisclosed_construct_switch() -> None:
    parsed = {
        "final_classification": "Concordant (no change)",
        "published_primary": "Objective response rate",
        "confidence": 0.88,
        "disclosure_status": "undisclosed",
        "reasoning_summary": "The publication presents ORR instead of registered PFS.",
        "endpoint_comparisons": [
            {
                "registered_endpoint": "Progression-free survival",
                "published_corresponding_endpoint": "Objective response rate",
                "same_construct": False,
                "same_timeframe": None,
                "same_definition": False,
                "same_population": True,
                "same_measurement_method": False,
                "registered_endpoint_reported": False,
                "change_disclosed": False,
                "actual_change_evidence": [
                    "The results identify ORR as principal efficacy and do not report PFS"
                ],
            }
        ],
    }

    result = matcher._adjudication_to_schema(parsed)

    assert result.switch_type == SwitchType.MAJOR_SWITCH
    assert result.primary_endpoint_match == "different"
    assert result.endpoint_comparisons[0].classification_label.endswith("major / undisclosed")


def test_multiple_registered_primaries_are_classified_individually() -> None:
    parsed = {
        "final_classification": "Concordant (no change)",
        "published_primary": "Objective response rate",
        "confidence": 0.82,
        "disclosure_status": "undisclosed",
        "reasoning_summary": "ORR is reported; AUC and DLT are absent.",
        "endpoint_comparisons": [
            {
                "registered_endpoint": "AUC",
                "published_corresponding_endpoint": "",
                "same_construct": None,
                "registered_endpoint_reported": False,
                "change_disclosed": False,
                "actual_change_evidence": ["The publication reports ORR but no AUC result"],
            },
            {
                "registered_endpoint": "DLT",
                "published_corresponding_endpoint": "",
                "same_construct": False,
                "registered_endpoint_reported": False,
                "change_disclosed": False,
                "actual_change_evidence": ["The publication reports ORR but no DLT result"],
            },
            {
                "registered_endpoint": "Objective response rate",
                "published_corresponding_endpoint": "Objective response rate",
                "same_construct": True,
                "same_timeframe": None,
                "same_definition": True,
                "same_population": True,
                "registered_endpoint_reported": True,
                "actual_change_evidence": [],
            },
        ],
    }

    result = matcher._adjudication_to_schema(parsed)

    assert result.switch_type == SwitchType.MAJOR_SWITCH
    assert len(result.endpoint_comparisons) == 3
    assert "prespecified_omitted" in result.switch_forms
    assert [item.classification_label for item in result.endpoint_comparisons] == [
        "Outcome switch — major / undisclosed",
        "Outcome switch — major / undisclosed",
        "Concordant (no change)",
    ]


def test_structured_explicit_timeframe_difference_is_modification() -> None:
    result = matcher._adjudication_to_schema(
        {
            "final_classification": "Concordant (no change)",
            "confidence": 0.9,
            "reasoning_summary": "The endpoint construct is retained at a different timepoint.",
            "endpoint_comparisons": [
                {
                    "registered_endpoint": "5-year recurrence-free survival",
                    "published_corresponding_endpoint": "2-year recurrence-free survival",
                    "same_construct": True,
                    "same_timeframe": False,
                    "same_definition": True,
                    "same_population": True,
                    "registered_endpoint_reported": True,
                    "actual_change_evidence": [
                        "Registry specifies 5 years; publication specifies 2 years"
                    ],
                }
            ],
        }
    )

    assert result.switch_type == SwitchType.MINOR_MODIFICATION
    assert result.endpoint_comparisons[0].classification_label.startswith("Outcome modification")


def test_structured_disclosed_construct_replacement_is_moderate() -> None:
    result = matcher._adjudication_to_schema(
        {
            "final_classification": "Outcome switch — major / undisclosed",
            "confidence": 0.9,
            "reasoning_summary": "The revised endpoint is explained in the publication.",
            "endpoint_comparisons": [
                {
                    "registered_endpoint": "Progression-free survival",
                    "published_corresponding_endpoint": "Disease-control rate",
                    "same_construct": False,
                    "registered_endpoint_reported": False,
                    "change_disclosed": True,
                    "actual_change_evidence": [
                        "Publication identifies disease-control rate as revised primary endpoint"
                    ],
                }
            ],
        }
    )

    assert result.switch_type == SwitchType.MODERATE_SWITCH
    assert result.endpoint_comparisons[0].classification_label.endswith(
        "moderate / partly disclosed"
    )


def test_structured_disclosed_exploratory_endpoint_is_additional() -> None:
    result = matcher._adjudication_to_schema(
        {
            "final_classification": "Outcome switch — major / undisclosed",
            "confidence": 0.9,
            "disclosure_status": "disclosed",
            "reasoning_summary": "ORR remains primary and PD-L1 response is exploratory.",
            "endpoint_comparisons": [
                {
                    "registered_endpoint": "Objective response rate",
                    "published_corresponding_endpoint": "Objective response rate",
                    "same_construct": True,
                    "same_timeframe": None,
                    "same_definition": True,
                    "same_population": True,
                    "registered_endpoint_reported": True,
                    "additional_endpoint_present": True,
                    "additional_endpoint_disclosed": True,
                    "actual_change_evidence": [],
                }
            ],
        }
    )

    assert result.switch_type == SwitchType.ADDITIONAL_OUTCOME
    assert result.endpoint_comparisons[0].classification_label.startswith("Additional outcome")
