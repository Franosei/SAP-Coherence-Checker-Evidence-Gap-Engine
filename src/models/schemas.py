"""
Pydantic schemas for every data structure in the pipeline.

All models use strict typing so that malformed API responses or LLM outputs
are rejected loudly at the boundary rather than propagating silently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations — match exactly the values documented in the proposal
# ---------------------------------------------------------------------------


class LinkageMethod(str, Enum):
    PUBLICATION_FAMILY = "publication_family"
    CTGOV_REFERENCE = "ctgov_reference"
    DIRECT = "direct"
    FUZZY = "fuzzy"
    AUTHOR_DATE = "author_date"
    MANUAL = "manual"


class LinkageConfidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNLINKED = "Unlinked"


class PublicationRole(str, Enum):
    PRIMARY_RESULTS = "primary_results"
    INTERIM_RESULTS = "interim_results"
    UPDATED_RESULTS = "updated_results"
    FINAL_RESULTS = "final_results"
    SECONDARY_ENDPOINT = "secondary_endpoint"
    SUBGROUP_POSTHOC = "subgroup_posthoc"
    SAFETY = "safety"
    QOL_PRO = "qol_pro"
    BIOMARKER_TRANSLATIONAL = "biomarker_translational"
    LONG_TERM_FOLLOWUP = "long_term_followup"
    EXTENSION_STUDY = "extension_study"
    PROTOCOL_SAP = "protocol_sap"
    OTHER = "other"
    # Could not be classified because the per-trial LLM-arbiter budget was spent
    # before this paper was reached. Distinct from OTHER (a real "none of the
    # above" verdict); always carries human_review_required=True.
    UNRESOLVED = "unresolved"


class PrimaryResultStatus(str, Enum):
    # Binary by design: the LLM linker either identifies the primary-results
    # paper (SELECTED) or determines none of the candidates report it
    # (NOT_FOUND). Low-confidence SELECTED picks carry human_review_required
    # rather than a separate "ambiguous" status.
    SELECTED = "SELECTED"
    NOT_FOUND = "NOT_FOUND"


class AnalysisStage(str, Enum):
    """When in the trial's analysis timeline a publication sits.

    Deliberately distinct from :class:`PublicationRole`. A single paper can be
    ``publication_role=PRIMARY_RESULTS`` with ``analysis_stage=FINAL`` ("final
    analysis of the primary endpoint"); role and stage answer different
    questions and are derived independently.
    """

    INTERIM = "INTERIM"
    PRIMARY = "PRIMARY"
    UPDATED = "UPDATED"
    FINAL = "FINAL"
    LONG_TERM = "LONG_TERM"
    UNSPECIFIED = "UNSPECIFIED"


class TrialIdentityStatus(str, Enum):
    """Hard gate for publication-family membership.

    Publication role and trial identity are separate questions: a clean
    "primary analysis" paper can still belong to a different trial. Only
    ``CONFIRMED`` candidates are members; ``REJECTED`` candidates are retained
    for audit but never enter role rollups or primary selection.
    """

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class EndpointRouting(str, Enum):
    AUTO_CONCORDANT = "auto_concordant"
    LLM = "llm"
    AUTO_MAJOR_SWITCH = "auto_major_switch"


class SwitchType(str, Enum):
    # No material change: same outcome, incl. pure terminology change or longer
    # follow-up of the same endpoint.
    CONCORDANT = "concordant"
    # A NEW/extra analysis is reported but the paper transparently labels it
    # post-hoc / exploratory / unplanned. Not outcome switching.
    ADDITIONAL_OUTCOME = "additional_outcome"
    # Same underlying outcome, but its timeframe / definition / metric /
    # threshold / analysis population changed ("outcome modification").
    MINOR_MODIFICATION = "minor_modification"
    # A prespecified-outcome switch that is partially disclosed, or where it is
    # unclear whether the change was influenced by the observed results.
    MODERATE_SWITCH = "moderate_switch"
    # A prespecified primary replaced / demoted / omitted, or a secondary
    # promoted — not transparently disclosed and plausibly results-driven.
    MAJOR_SWITCH = "major_switch"


class SwitchDirection(str, Enum):
    NONE = "none"
    PROMOTION_OF_SECONDARY = "promotion_of_secondary"
    PRIMARY_DEMOTED = "primary_demoted"
    PRESPECIFIED_OMITTED = "prespecified_omitted"
    UNREGISTERED_ADDED = "unregistered_added"
    DEFINITION_CHANGED = "definition_changed"
    ANALYSIS_POPULATION_CHANGED = "analysis_population_changed"
    COMPOSITE_MODIFIED = "composite_modified"
    TIMEFRAME_CHANGED = "timeframe_changed"
    ENDPOINT_REPLACED = "endpoint_replaced"
    SURROGATE_SUBSTITUTED = "surrogate_substituted"  # pCR-to-survival substitution


class LLMConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HumanReviewStatus(str, Enum):
    YES = "yes"
    NO = "no"
    SPOT_CHECK = "spot_check"
    # The LLM verdict was confident (confidence_score >= threshold, not flagged)
    # and was accepted without individual human review. Still auditable and
    # still subject to the random spot-check sample.
    AUTO_ACCEPTED = "auto_accepted"


# Review states that count as "resolved" — no longer in the human queue,
# and their human_final_class / human_poolable are used downstream.
RESOLVED_REVIEW_STATES = ("yes", "spot_check", "auto_accepted")


class HumanDecision(str, Enum):
    CONFIRM = "confirm"
    OVERRIDE = "override"
    EXCLUDE = "exclude"
    DEFER = "defer"


class EvidenceStrength(str, Enum):
    STRONG = "Strong"
    MODERATE = "Moderate"
    WEAK = "Weak"
    GAP = "Gap"


# ---------------------------------------------------------------------------
# Module 1 — Linkage audit log entry (Section 3.2.2)
# ---------------------------------------------------------------------------


class LinkageAuditEntry(BaseModel):
    nct_id: str = Field(..., description="ClinicalTrials.gov NCT identifier")
    pmid: Optional[str] = Field(None, description="PubMed identifier; NULL if unlinked")
    linkage_method: LinkageMethod
    linkage_confidence: LinkageConfidence
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    linked_by: str = Field(
        default="pipeline_v4.0",
        description="'pipeline_v4.0' or 'reviewer:<initials>' if manually resolved",
    )
    notes: Optional[str] = Field(None, description="Free text — e.g. resolution rationale")

    @field_validator("nct_id")
    @classmethod
    def nct_id_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.startswith("NCT"):
            raise ValueError(f"nct_id must start with 'NCT', got: {v!r}")
        return v


class PublicationFamilyEntry(BaseModel):
    """One candidate publication and its independently assessed trial role."""

    nct_id: str
    pmid: str
    doi: str = ""
    title: str = ""
    year: str = ""
    discovery_sources: list[str] = Field(default_factory=list)
    registry_reference_type: Optional[str] = None
    registry_result_reference: bool = False
    same_trial: Optional[bool] = None
    trial_identity_status: TrialIdentityStatus = TrialIdentityStatus.UNCERTAIN
    trial_match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    trial_match_evidence: list[str] = Field(default_factory=list)
    publication_role: PublicationRole = PublicationRole.OTHER
    analysis_stage: AnalysisStage = AnalysisStage.UNSPECIFIED
    analysis_stage_raw: str = ""
    population_scope: str = "unknown"
    randomized_n: Optional[int] = None
    analyzed_n: Optional[int] = None
    arms: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    data_cutoff: str = ""
    followup_duration: str = ""
    explicit_primary: bool = False
    explicit_interim: bool = False
    explicit_final: bool = False
    explicit_updated: bool = False
    is_interim: bool = False
    nct_in_article: bool = False
    # PubMed publication-type screen: "results_candidate" | "excluded" | "neutral".
    # "excluded" (protocol, meta-analysis, editorial, ...) never reaches the LLM.
    pubtype_screen: str = "neutral"
    reports_randomized_arms: bool = False
    reports_outcome_data: bool = False
    # True on the single candidate the LLM names as the trial's primary-results paper.
    is_primary_results_pick: bool = False
    classification_confidence: str = "low"
    classification_reason: str = ""
    human_review_required: bool = True


# ---------------------------------------------------------------------------
# Module 2 — LLM structured output (Section 3.3.2)
# ---------------------------------------------------------------------------


class EndpointComparison(BaseModel):
    """One registered primary endpoint compared with its published counterpart.

    Attribute fields are tri-state: ``True`` means affirmatively the same,
    ``False`` means affirmatively different, and ``None`` means not stated or
    unclear. This prevents missing abstract detail being treated as a change.
    """

    registered_endpoint: str
    published_corresponding_endpoint: str = ""
    same_construct: Optional[bool] = None
    same_timeframe: Optional[bool] = None
    same_definition: Optional[bool] = None
    same_population: Optional[bool] = None
    same_measurement_method: Optional[bool] = None
    registered_endpoint_reported: Optional[bool] = None
    additional_endpoint_present: bool = False
    additional_endpoint_disclosed: bool = False
    change_disclosed: Optional[bool] = None
    actual_change_evidence: list[str] = Field(default_factory=list)
    classification_label: str = ""


class LLMEndpointClassification(BaseModel):
    """
    Exact JSON schema the LLM must return. Pydantic validates every field;
    a malformed LLM response raises ValidationError rather than being silently
    accepted (Section 6.1 — LLM output parsing).
    """

    switch_type: SwitchType
    classification_label: str = ""
    primary_endpoint_match: str = "unclear"
    published_primary_extracted: str = ""
    direction: SwitchDirection
    step_by_step_reasoning: str = Field(
        ...,
        min_length=20,
        description="Mandatory — LLM must show its working",
    )
    confidence: LLMConfidence
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Numeric confidence 0-1; pairs at/above the review threshold skip human review. "
        "0.0 means the model did not return one — derived from the categorical confidence.",
    )
    comparability_for_pooling: bool
    flag_for_human_review: bool
    key_differences: list[str] = Field(
        default_factory=list,
        description="Specific textual differences identified by the LLM",
    )
    disclosed_as_exploratory: bool = Field(
        default=False,
        description="Paper transparently labels the new/changed analysis as post-hoc / exploratory / unplanned",
    )
    likely_results_driven: bool = Field(
        default=False,
        description="The change plausibly favours a more positive or significant finding",
    )
    switch_forms: list[str] = Field(
        default_factory=list,
        description="Which typical outcome-switching forms apply (primary_replaced, primary_demoted, "
        "secondary_promoted, prespecified_omitted, unregistered_added, timepoint_changed, "
        "definition_changed, scale_threshold_changed, analysis_population_changed)",
    )
    actual_change_evidence: list[str] = Field(default_factory=list)
    missing_detail_only: bool = False
    adjudication_guardrail: str = ""
    endpoint_comparisons: list[EndpointComparison] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_confidence_score(self) -> LLMEndpointClassification:
        if self.confidence_score == 0.0:
            self.confidence_score = {"high": 0.9, "medium": 0.65, "low": 0.35}.get(
                self.confidence.value, 0.35
            )
        return self


# ---------------------------------------------------------------------------
# Module 2 — Full decision log entry (Section 3.3.4)
# ---------------------------------------------------------------------------


class DecisionLogEntry(BaseModel):
    pair_id: str = Field(..., description="Unique (nct_id, pmid) pair identifier")
    registered_endpoint: str
    published_endpoint: str

    # Layer 1 — Embedding similarity
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    ssi: float = Field(..., description="(1 - similarity_score) × 100")
    routing: EndpointRouting

    # Layer 2 — LLM (NULL if not called)
    llm_model: Optional[str] = None
    llm_switch_type: Optional[SwitchType] = None
    llm_reasoning: Optional[str] = None
    llm_confidence: Optional[LLMConfidence] = None
    llm_comparability: Optional[bool] = None
    llm_flag: Optional[bool] = None
    llm_disclosed_exploratory: Optional[bool] = None
    llm_results_driven: Optional[bool] = None
    llm_switch_forms: Optional[str] = None
    llm_confidence_score: Optional[float] = None
    llm_classification_label: Optional[str] = None
    llm_primary_endpoint_match: Optional[str] = None
    llm_published_primary_extracted: Optional[str] = None
    llm_actual_change_evidence: Optional[str] = None
    llm_missing_detail_only: Optional[bool] = None
    llm_adjudication_guardrail: Optional[str] = None
    llm_endpoint_comparisons: Optional[str] = None

    # Breast cancer population context (populated by Module 1 classifier)
    bc_subtype: Optional[str] = Field(
        None,
        description="her2_positive | hr_positive | tnbc | unknown_subtype",
    )
    bc_setting: Optional[str] = Field(
        None,
        description="neoadjuvant | adjuvant | metastatic | unknown_setting",
    )

    # Layer 3 — Human review
    human_reviewed: HumanReviewStatus = HumanReviewStatus.NO
    human_decision: Optional[HumanDecision] = None
    human_final_class: Optional[SwitchType] = None
    human_poolable: Optional[bool] = None
    override_reason: Optional[str] = None
    reviewer_initials: Optional[str] = None
    review_timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def override_requires_reason(self) -> DecisionLogEntry:
        if self.human_decision == HumanDecision.OVERRIDE and not self.override_reason:
            raise ValueError("override_reason is mandatory when human_decision='override'")
        return self

    @model_validator(mode="after")
    def ssi_consistency(self) -> DecisionLogEntry:
        expected_ssi = round((1.0 - self.similarity_score) * 100, 4)
        if abs(self.ssi - expected_ssi) > 0.01:
            raise ValueError(f"SSI inconsistency: expected {expected_ssi}, got {self.ssi}")
        return self

    @classmethod
    def from_layer1(
        cls,
        pair_id: str,
        registered_endpoint: str,
        published_endpoint: str,
        similarity_score: float,
        routing: EndpointRouting,
    ) -> DecisionLogEntry:
        return cls(
            pair_id=pair_id,
            registered_endpoint=registered_endpoint,
            published_endpoint=published_endpoint,
            similarity_score=round(similarity_score, 4),
            ssi=round((1.0 - similarity_score) * 100, 4),
            routing=routing,
        )


# ---------------------------------------------------------------------------
# Module 3 — Effect measure for a single trial (Section 3.4.2)
# ---------------------------------------------------------------------------


class EffectMeasure(BaseModel):
    pair_id: str
    nct_id: str
    pmid: str
    hr: float = Field(..., gt=0, description="Reported hazard ratio")
    hr_lci: float = Field(..., gt=0, description="Lower 95% CI of HR")
    hr_uci: float = Field(..., gt=0, description="Upper 95% CI of HR")
    log_hr: float
    se_log_hr: float = Field(..., gt=0)
    variance: float = Field(..., gt=0)
    extraction_method: str = Field(
        ..., description="'regex' or 'manual'; manual entries flagged for review"
    )
    source_reference: str = Field(
        ..., description="PMID + table/figure reference, e.g. 'PMID:12345 Table 2'"
    )
    registration_date: Optional[str] = Field(
        None, description="ISO 8601 date of trial registration — for sequential analysis"
    )

    @model_validator(mode="after")
    def derive_log_fields(self) -> EffectMeasure:
        import math

        expected_log_hr = round(math.log(self.hr), 6)
        expected_se = round((math.log(self.hr_uci) - math.log(self.hr_lci)) / (2 * 1.96), 6)
        expected_var = round(expected_se**2, 8)

        if abs(self.log_hr - expected_log_hr) > 0.001:
            raise ValueError(f"log_hr mismatch: expected {expected_log_hr}, got {self.log_hr}")
        if abs(self.se_log_hr - expected_se) > 0.001:
            raise ValueError(f"SE(log HR) mismatch: expected {expected_se}, got {self.se_log_hr}")
        if abs(self.variance - expected_var) > 0.0001:
            raise ValueError(f"Variance mismatch: expected {expected_var}, got {self.variance}")
        return self

    @classmethod
    def from_raw(
        cls,
        pair_id: str,
        nct_id: str,
        pmid: str,
        hr: float,
        hr_lci: float,
        hr_uci: float,
        extraction_method: str,
        source_reference: str,
        registration_date: Optional[str] = None,
    ) -> EffectMeasure:
        import math

        if hr <= 0 or hr_lci <= 0 or hr_uci <= 0:
            return cls(
                pair_id=pair_id,
                nct_id=nct_id,
                pmid=pmid,
                hr=hr,
                hr_lci=hr_lci,
                hr_uci=hr_uci,
                log_hr=0.0,
                se_log_hr=0.0,
                variance=0.0,
                extraction_method=extraction_method,
                source_reference=source_reference,
                registration_date=registration_date,
            )

        log_hr = round(math.log(hr), 6)
        se_log_hr = round((math.log(hr_uci) - math.log(hr_lci)) / (2 * 1.96), 6)
        variance = round(se_log_hr**2, 8)
        return cls(
            pair_id=pair_id,
            nct_id=nct_id,
            pmid=pmid,
            hr=hr,
            hr_lci=hr_lci,
            hr_uci=hr_uci,
            log_hr=log_hr,
            se_log_hr=se_log_hr,
            variance=variance,
            extraction_method=extraction_method,
            source_reference=source_reference,
            registration_date=registration_date,
        )


# ---------------------------------------------------------------------------
# Module 4 — Power audit entry (Section 3.5)
# ---------------------------------------------------------------------------


class PowerAuditEntry(BaseModel):
    nct_id: str
    enrollment: int = Field(..., gt=0)
    assumed_hr: float = Field(..., gt=0)
    assumed_event_rate: float = Field(..., gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    power: float = Field(default=0.80, gt=0, lt=1)
    posterior_hr_at_registration: Optional[float] = Field(
        None, description="Bayesian posterior mean available at trial registration date"
    )
    optimism_bias: Optional[float] = Field(
        None, description="assumed_hr - posterior_hr; positive = overly optimistic"
    )
    excluded_reason: Optional[str] = Field(
        None, description="If excluded from power audit, reason code"
    )
    inputs_source: str = Field(
        ...,
        description="How enrollment/event-rate were obtained: 'registry' | 'abstract' | 'manual'",
    )


# ---------------------------------------------------------------------------
# Evidence gap scorecard entry (Section 3.6)
# ---------------------------------------------------------------------------


class ScorecardEntry(BaseModel):
    endpoint_cluster: str
    trials_included: int
    trials_excluded: int
    trials_unlinked: int
    human_override_rate_pct: float
    mean_ssi: float
    human_confirmed_switch_rate_pct: float
    ai_human_agreement_rate_pct: float
    pooled_hr: Optional[float] = None
    pooled_hr_cri_lower: Optional[float] = None
    pooled_hr_cri_upper: Optional[float] = None
    between_trial_tau: Optional[float] = None
    i_squared_pct: Optional[float] = None
    mean_optimism_bias: Optional[float] = None
    evidence_strength: EvidenceStrength
