"""
Tests for the append-only DecisionLog manager.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.models.decision_log import DecisionLog
from src.models.schemas import (
    DecisionLogEntry,
    EndpointRouting,
    HumanDecision,
    SwitchType,
)


def _make_entry(pair_id: str, score: float, routing: EndpointRouting) -> DecisionLogEntry:
    return DecisionLogEntry.from_layer1(
        pair_id=pair_id,
        registered_endpoint="Pathologic complete response in breast and axilla at surgery",
        published_endpoint="Pathologic complete response at definitive surgery",
        similarity_score=score,
        routing=routing,
    )


class TestDecisionLog:
    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "decision_log.csv"
        self.log = DecisionLog(path=self.log_path)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_initialises_empty_csv(self):
        df = self.log.read()
        assert df.empty
        assert "pair_id" in df.columns

    def test_append_and_read(self):
        entry = _make_entry("NCT001_PMID001", 0.95, EndpointRouting.AUTO_CONCORDANT)
        self.log.append(entry)
        df = self.log.read()
        assert len(df) == 1
        assert df.iloc[0]["pair_id"] == "NCT001_PMID001"

    def test_multiple_appends(self):
        for i in range(5):
            self.log.append(_make_entry(f"NCT00{i}_PMID00{i}", 0.80, EndpointRouting.LLM))
        assert len(self.log.read()) == 5

    def test_record_human_review_confirm(self):
        entry = _make_entry("NCT001_PMID001", 0.70, EndpointRouting.LLM)
        self.log.append(entry)
        self.log.record_human_review(
            pair_id="NCT001_PMID001",
            human_decision=HumanDecision.CONFIRM,
            human_final_class=SwitchType.CONCORDANT,
            human_poolable=True,
            reviewer_initials="FO",
        )
        df = self.log.read()
        row = df[df["pair_id"] == "NCT001_PMID001"].iloc[0]
        assert row["human_decision"] == "confirm"
        assert row["human_final_class"] == "concordant"
        assert row["reviewer_initials"] == "FO"

    def test_override_requires_reason(self):
        entry = _make_entry("NCT002_PMID002", 0.60, EndpointRouting.LLM)
        self.log.append(entry)
        with pytest.raises(ValueError, match="override_reason is mandatory"):
            self.log.record_human_review(
                pair_id="NCT002_PMID002",
                human_decision=HumanDecision.OVERRIDE,
                human_final_class=SwitchType.MODERATE_SWITCH,
                human_poolable=False,
                reviewer_initials="FO",
                override_reason=None,
            )

    def test_pending_review_returns_correct_rows(self):
        # LLM-routed → pending
        self.log.append(_make_entry("NCT003_PMID003", 0.75, EndpointRouting.LLM))
        # Auto-concordant → not pending
        self.log.append(_make_entry("NCT004_PMID004", 0.97, EndpointRouting.AUTO_CONCORDANT))

        pending = self.log.pending_review()
        assert len(pending) == 1
        assert pending.iloc[0]["pair_id"] == "NCT003_PMID003"

    def test_unknown_pair_id_raises_keyerror(self):
        with pytest.raises(KeyError):
            self.log.record_human_review(
                pair_id="NONEXISTENT",
                human_decision=HumanDecision.CONFIRM,
                human_final_class=SwitchType.CONCORDANT,
                human_poolable=True,
                reviewer_initials="FO",
            )

    def test_governance_summary_keys(self):
        self.log.append(_make_entry("NCT005_PMID005", 0.65, EndpointRouting.LLM))
        summary = self.log.governance_summary()
        assert "total_pairs" in summary
        assert "human_review_rate_pct" in summary
        assert "human_override_rate_pct" in summary
