"""
Unit tests for Pydantic schemas — validates that malformed inputs are rejected
loudly and that computed fields (log_hr, SSI, etc.) are consistent.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    DecisionLogEntry,
    EffectMeasure,
    EndpointRouting,
    LinkageAuditEntry,
    LinkageConfidence,
    LinkageMethod,
    LLMConfidence,
    LLMEndpointClassification,
    SwitchDirection,
    SwitchType,
)

# ---------------------------------------------------------------------------
# LinkageAuditEntry
# ---------------------------------------------------------------------------

class TestLinkageAuditEntry:
    def test_valid(self):
        entry = LinkageAuditEntry(
            nct_id="NCT01234567",
            pmid="12345678",
            linkage_method=LinkageMethod.DIRECT,
            linkage_confidence=LinkageConfidence.HIGH,
        )
        assert entry.nct_id == "NCT01234567"
        assert entry.linked_by == "pipeline_v2.0"

    def test_nct_id_normalised_to_uppercase(self):
        entry = LinkageAuditEntry(
            nct_id="nct01234567",
            linkage_method=LinkageMethod.DIRECT,
            linkage_confidence=LinkageConfidence.HIGH,
        )
        assert entry.nct_id == "NCT01234567"

    def test_invalid_nct_id_format(self):
        with pytest.raises(ValidationError, match="nct_id must start with"):
            LinkageAuditEntry(
                nct_id="12345678",
                linkage_method=LinkageMethod.DIRECT,
                linkage_confidence=LinkageConfidence.HIGH,
            )

    def test_pmid_optional(self):
        entry = LinkageAuditEntry(
            nct_id="NCT99999999",
            pmid=None,
            linkage_method=LinkageMethod.MANUAL,
            linkage_confidence=LinkageConfidence.UNLINKED,
        )
        assert entry.pmid is None


# ---------------------------------------------------------------------------
# LLMEndpointClassification
# ---------------------------------------------------------------------------

class TestLLMEndpointClassification:
    def test_valid(self):
        result = LLMEndpointClassification(
            switch_type=SwitchType.MINOR_MODIFICATION,
            direction=SwitchDirection.TIMEFRAME_CHANGED,
            step_by_step_reasoning="Both endpoints measure MACE; the published version uses 6 months vs the registered 12 months.",
            confidence=LLMConfidence.HIGH,
            comparability_for_pooling=True,
            flag_for_human_review=False,
            key_differences=["timeframe changed from 12 to 6 months"],
        )
        assert result.switch_type == SwitchType.MINOR_MODIFICATION

    def test_reasoning_too_short(self):
        with pytest.raises(ValidationError):
            LLMEndpointClassification(
                switch_type=SwitchType.CONCORDANT,
                direction=SwitchDirection.NONE,
                step_by_step_reasoning="OK",   # too short
                confidence=LLMConfidence.HIGH,
                comparability_for_pooling=True,
                flag_for_human_review=False,
            )

    def test_key_differences_defaults_empty(self):
        result = LLMEndpointClassification(
            switch_type=SwitchType.CONCORDANT,
            direction=SwitchDirection.NONE,
            step_by_step_reasoning="Endpoints are identical in wording and scope.",
            confidence=LLMConfidence.HIGH,
            comparability_for_pooling=True,
            flag_for_human_review=False,
        )
        assert result.key_differences == []


# ---------------------------------------------------------------------------
# DecisionLogEntry
# ---------------------------------------------------------------------------

class TestDecisionLogEntry:
    def test_from_layer1_auto_concordant(self):
        entry = DecisionLogEntry.from_layer1(
            pair_id="NCT01234567_12345678",
            registered_endpoint="All-cause mortality at 24 months",
            published_endpoint="All-cause mortality at 24 months",
            similarity_score=0.97,
            routing=EndpointRouting.AUTO_CONCORDANT,
        )
        assert abs(entry.ssi - (1 - 0.97) * 100) < 0.01

    def test_ssi_inconsistency_raises(self):
        with pytest.raises(ValidationError, match="SSI inconsistency"):
            DecisionLogEntry(
                pair_id="NCT01234567_12345678",
                registered_endpoint="Endpoint A",
                published_endpoint="Endpoint B",
                similarity_score=0.80,
                ssi=99.0,   # wrong: should be 20.0
                routing=EndpointRouting.LLM,
            )

    def test_override_without_reason_raises(self):
        from src.models.schemas import HumanDecision, HumanReviewStatus

        with pytest.raises(ValidationError, match="override_reason is mandatory"):
            DecisionLogEntry(
                pair_id="NCT01234567_12345678",
                registered_endpoint="A",
                published_endpoint="B",
                similarity_score=0.70,
                ssi=30.0,
                routing=EndpointRouting.LLM,
                human_reviewed=HumanReviewStatus.YES,
                human_decision=HumanDecision.OVERRIDE,
                override_reason=None,   # must be provided
            )


# ---------------------------------------------------------------------------
# EffectMeasure
# ---------------------------------------------------------------------------

class TestEffectMeasure:
    def test_from_raw_computes_correctly(self):
        em = EffectMeasure.from_raw(
            pair_id="NCT01234567_12345678",
            nct_id="NCT01234567",
            pmid="12345678",
            hr=0.75,
            hr_lci=0.60,
            hr_uci=0.94,
            extraction_method="regex",
            source_reference="PMID:12345678 Table 2",
        )
        assert abs(em.log_hr - math.log(0.75)) < 0.001
        expected_se = (math.log(0.94) - math.log(0.60)) / (2 * 1.96)
        assert abs(em.se_log_hr - expected_se) < 0.001
        assert abs(em.variance - expected_se**2) < 0.00001

    def test_hr_must_be_positive(self):
        with pytest.raises(ValidationError):
            EffectMeasure.from_raw(
                pair_id="x",
                nct_id="NCT00000000",
                pmid="99999999",
                hr=-0.5,
                hr_lci=0.3,
                hr_uci=0.8,
                extraction_method="manual",
                source_reference="test",
            )
