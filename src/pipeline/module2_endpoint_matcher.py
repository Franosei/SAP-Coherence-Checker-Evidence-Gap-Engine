"""
Module 2 - Outcome-switch classification with human oversight.

Compares the registered primary outcome on ClinicalTrials.gov against the
primary/main outcome presented in the linked publication and classifies whether
an outcome switch occurred (see _LLM_SYSTEM_PROMPT for the full definition).

Layer 1  Endpoint-embedding cosine similarity
    Computed for every SELECTED pair and stored as `similarity_score`, but
    INFORMATIONAL ONLY since v4.1 — it no longer routes anything.

Layer 2  LLM clinical judge
    Runs on every pair. Returns a validated switch_type
    (concordant | additional_outcome | minor_modification | moderate_switch |
    major_switch), reasoning, a numeric confidence_score, and disclosure /
    results-driven flags. Malformed output -> flagged for human review.

Layer 3  Human review gate
    Every model verdict is a suggestion and enters the human review queue by
    default. Automatic acceptance is available only through the explicit
    ENDPOINT_AUTO_ACCEPT opt-in after prospective validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import pandas as pd

from src.models.decision_log import DecisionLog
from src.models.schemas import (
    DecisionLogEntry,
    EndpointComparison,
    EndpointRouting,
    HumanDecision,
    HumanReviewStatus,
    LLMEndpointClassification,
)
from src.pipeline.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_COST_PER_1K_USD,
    EMBEDDING_MODEL,
    ENDPOINT_AUTO_ACCEPT,
    ENDPOINT_MATCHING_LLM_ONLY,
    ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD,
    LLM_BASE_URL,
    LLM_COST_CEILING_USD,
    LLM_COST_PER_1K_INPUT_USD,
    LLM_COST_PER_1K_OUTPUT_USD,
    LLM_MAX_TOKENS,
    LLM_MODEL_PRIMARY,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    SIMILARITY_AUTO_CONCORDANT,
    SIMILARITY_LLM_LOWER,
)

logger = logging.getLogger(__name__)

# Verdicts that always go to a human — never auto-accepted.
_SWITCH_VERDICTS = {"moderate_switch", "major_switch"}


_LLM_SYSTEM_PROMPT = """\
You are a clinical trial outcome-reporting adjudicator.

Your task is to compare the prospectively registered trial outcomes with the outcomes reported in the publication and classify the trial-publication pair for OUTCOME SWITCHING.

IMPORTANT:
This project defines outcome switching primarily as a discrepancy between the prospectively registered primary endpoint and the published primary endpoint.

Do NOT label a study as an outcome switch merely because:
- the publication reports many hazard ratios
- the publication reports subgroup analyses
- the publication reports adjusted and unadjusted analyses
- the publication reports additional secondary outcomes
- the publication emphasises a treatment effect in one subgroup
- the treatment comparison changes while the endpoint remains the same
- the publication uses slightly different terminology for a clinically equivalent endpoint
- the publication reports an additional exploratory analysis
- the primary endpoint result is non-significant
- another outcome has a more favourable result
- the abstract does not repeat every qualifier, confirmation rule, assessment
  schedule, or registered timeframe

CRITICAL RULE: "not explicitly restated in the abstract" is NOT evidence of an
outcome modification or switch. An abstract is a compressed report, not the full
methods section. Classify a change only when the publication provides affirmative
evidence that the actual endpoint, timeframe, definition, components, threshold,
or analysis population is different. Absence of detail is not evidence of change.

Never speculate about investigator intent. Do NOT use phrases such as "the authors may have tried to present more favourable results" unless there is explicit documentary evidence.

==================================================
1. DETERMINE WHICH REGISTRY VERSION TO USE
==================================================
When registry change history is available, prioritise the EARLIEST PROSPECTIVE VERSION that existed before trial results were known. Do NOT automatically treat the current registry entry as the original prespecified outcome. Always state: original/prospective registered outcome; later registry changes if relevant; which version you used. If the timing of registration or amendment relative to results cannot be established, state that uncertainty and avoid assigning a major switch solely on the basis of the later version.

==================================================
2. IDENTIFY THE PRIMARY OUTCOME
==================================================
Extract separately: A. Registered primary outcome(s); B. Registered secondary outcome(s); C. Published primary outcome(s); D. Other published outcomes.
Do not confuse an outcome with an analysis, a treatment comparison, a subgroup, or a statistical model. "DFS in A vs B" and "DFS in C vs B" are the SAME outcome with different comparisons. Adjusted HR, unadjusted HR, and DFS in a subgroup are analyses of the same outcome, not new outcomes.

When there are multiple registered primary endpoints, compare EACH registered
endpoint separately. Do not collapse AUC + DLT + ORR into a single semantic
comparison. Return one endpoint_comparisons item per registered primary, then
derive the overall trial-publication classification from those items.

==================================================
3. CHECK CLINICAL EQUIVALENCE
==================================================
FIRST determine whether the clinical construct changed. ONLY AFTER THAT examine
timeframe, definition, population, components, threshold, measurement method,
and disclosure. Do not classify an outcome as switched merely because
terminology differs. Assess whether the registered and published endpoints are
clinically and operationally equivalent. DFS vs relapse-free survival may be
concordant if event definitions are materially the same. ORR vs reporting CR +
PR may be concordant when ORR can be directly determined. Reporting an HR,
median, or percentage for the same clinical outcome does not change the
construct. PFS at 12 months versus median PFS is never an outcome switch; it is
Concordant unless the publication affirmatively defines a substantively
different intended assessment, in which case it is Outcome modification. PFS
versus ORR changes the construct and may be an outcome switch.

For every same_* field, use true only when affirmatively the same, false only
when the source affirmatively demonstrates a difference, and null when the
detail is not stated or unclear. A null is not evidence of a change.

==================================================
4. CLASSIFICATION OPTIONS  (choose exactly one)
==================================================
1. "Concordant (no change)" — registered primary is reported as the published primary; wording differs but endpoints are clinically equivalent; publication adds secondary analyses but the primary is intact; a treatment-comparison hierarchy changes but the outcome is unchanged; the registered outcome is reported through its components and can be unambiguously reconstructed.
2. "Additional outcome (disclosed exploratory)" — ONLY when the registered primary remains intact and reported AND the publication introduces a genuinely new ENDPOINT not prespecified AND it is transparently described as exploratory/post hoc/additional. NOT for subgroup/sensitivity/adjusted analyses or additional treatment comparisons or prespecified secondary endpoints.
3. "Outcome modification (timeframe/definition/population)" — the underlying outcome is recognisably the same but the publication AFFIRMATIVELY REPORTS a materially different timeframe, definition, composite, assessment method, analysis population, threshold, or event definition. Missing abstract detail is not a modification.
4. "Outcome switch — moderate / partly disclosed" — a meaningful change in the primary endpoint or its hierarchy, but the original endpoint is still reported or discussed, OR the change is partly acknowledged, OR there is promotion/demotion but not complete undisclosed replacement.
5. "Outcome switch — major / undisclosed" — ONLY with strong evidence that the prospectively registered primary was omitted, replaced or materially demoted AND a materially different endpoint was presented as the principal outcome AND the change was not transparently disclosed. DO NOT use this when the registered primary is actually reported, only the label differs, its components allow it to be calculated, the difference concerns only treatment comparison, there was a clearly disclosed protocol amendment, the paper is an explicitly stated interim analysis, another planned comparison was published separately, or only secondary outcomes are missing.

Operational decision order for EACH registered endpoint:
1. Is it the same clinical construct?
2. If yes, is there affirmative evidence of a changed timeframe, definition,
   population, components, threshold, or measurement method? If yes, Outcome
   modification; otherwise Concordant.
3. If no, was the registered endpoint replaced? A disclosed/partly explained
   replacement is Moderate; an omitted replacement without explanation is Major.
4. After registered endpoints are assessed, use Additional outcome only when
   they remain reported and an extra endpoint is explicitly exploratory/additional.

==================================================
5-8. OMITTED SECONDARIES / MULTIPLE PUBLICATIONS / DEMOTION / RESPONSE RATE
==================================================
A missing prespecified SECONDARY outcome does NOT by itself constitute a primary outcome switch — record it under secondary_outcomes_missing. Before calling an outcome omitted, check whether the publication says another comparison was previously/separately reported, is an interim analysis, or an arm is still under follow-up ("not reported in this publication" vs "not reported by the trial"). Do not infer demotion from scientific emphasis, effect size, or number of paragraphs — it requires evidence of a formal change to secondary, omission with promotion of another endpoint, or explicit replacement. If the registry specifies ORR (= CR + PR) and the publication reports CR and PR separately, ORR is NOT omitted if it can be unambiguously calculated. If the registry gives "ORR at 3 months" and the abstract reports ORR without restating the timeframe, classify Concordant; use Outcome modification only if the paper explicitly reports a different timepoint.

Examples that MUST be Concordant:
- Registered confirmed overall/objective response; publication reports ORR as
  CR + PR but does not repeat "confirmed" or the full assessment schedule.
- Registered pCR; publication reports pCR results by randomized treatment arm.
- Registered survival percentage; publication reports an HR for the same survival endpoint.

Example that IS Outcome modification:
- Registered 5-year recurrence-free survival; publication explicitly presents
  2-year recurrence-free survival as the primary endpoint.

==================================================
9. OUTPUT FORMAT  (return valid JSON only, this exact structure)
==================================================
{
  "registered_primary_original": "",
  "registered_primary_current": "",
  "published_primary": "",
  "primary_endpoint_match": "exact | clinically equivalent | modified | different | unclear",
  "endpoint_comparisons": [
    {
      "registered_endpoint": "",
      "published_corresponding_endpoint": "",
      "same_construct": true,
      "same_timeframe": null,
      "same_definition": true,
      "same_population": true,
      "same_measurement_method": null,
      "registered_endpoint_reported": true,
      "additional_endpoint_present": false,
      "additional_endpoint_disclosed": false,
      "change_disclosed": null,
      "actual_change_evidence": []
    }
  ],
  "final_classification": "Concordant (no change) | Additional outcome (disclosed exploratory) | Outcome modification (timeframe/definition/population) | Outcome switch — moderate / partly disclosed | Outcome switch — major / undisclosed",
  "confidence": 0.00,
  "secondary_outcomes_missing": [],
  "additional_outcomes": [],
  "registry_history_relevant": true,
  "registry_history_interpretation": "",
  "treatment_comparison_changed": false,
  "endpoint_changed": false,
  "actual_change_evidence": [],
  "missing_detail_only": false,
  "disclosure_status": "none needed | disclosed | partly disclosed | undisclosed | unclear",
  "evidence_registered": "",
  "evidence_published": "",
  "reasoning_summary": "",
  "human_review_required": false
}

==================================================
10. CONSERVATIVE ADJUDICATION RULE
==================================================
When evidence is ambiguous, choose the LESS severe classification and set "human_review_required": true. `actual_change_evidence` must contain short, source-grounded evidence of an explicit conflict; leave it empty when details are merely absent. Set `missing_detail_only=true` when the apparent discrepancy is only non-restatement in the abstract. Do not infer a major switch unless the evidence clearly demonstrates replacement, omission or demotion of the prospectively registered primary endpoint. A low semantic similarity score between registry and publication text is NOT sufficient evidence of outcome switching. Clinical equivalence and registry chronology take precedence over textual similarity.
"""

_LLM_USER_TEMPLATE = """\
Trial subtype: {bc_subtype}
Treatment setting: {bc_setting}

REGISTERED OUTCOMES (ClinicalTrials.gov)
Original / prospective primary outcome(s):
{registered_original}

Current registry primary outcome(s) (may equal the original if never amended):
{registered_current}

Registered secondary outcome(s):
{registered_secondary}

Registry change history retrievable for this trial: {registry_history_note}

PUBLICATION (PubMed / PMC) — identify the PUBLISHED PRIMARY outcome from this text; \
ignore secondary outcomes, subgroups, adjusted/unadjusted models, and treatment-comparison changes.
Extracted primary-endpoint sentence(s):
{published_endpoint}

Results section:
{published_results}

Conclusions:
{published_conclusion}

Adjudicate outcome switching between the prospectively registered PRIMARY outcome and \
the publication's PRIMARY outcome, following every rule above. Return the JSON object only.
"""


def _normalise(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return "" if cleaned.lower() in {"", "none", "nan"} else cleaned


def _split_registered_candidates(text: str) -> list[str]:
    norm = _normalise(text)
    if not norm:
        return []
    parts = [_normalise(part) for part in re.split(r"\s*\|\s*|\n+", norm)]
    return [part for part in parts if part] or [norm]


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two pre-normalised embedding vectors.

    The OpenAI Embeddings API returns unit vectors (L2-normalised), so the
    cosine similarity is equal to the dot product — no magnitude division
    required.  Clamped to [0, 1] to guard against floating-point drift.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot))


def _embed_batch(texts: list[str], client: Any) -> list[list[float]]:
    """
    Embed a batch of texts via the OpenAI Embeddings API.

    Sends texts in chunks of ``EMBEDDING_BATCH_SIZE`` to stay within the
    API's per-request limit.  Returns one embedding vector per input text,
    preserving order.

    Parameters
    ----------
    texts:
        Non-empty strings to embed.
    client:
        An authenticated ``openai.OpenAI`` instance.

    Returns
    -------
    list[list[float]]
        One 1 536-dimensional unit vector per input text.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        chunk = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        # Response items are ordered to match the input order.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

        tokens_used = response.usage.total_tokens
        cost = tokens_used / 1_000 * EMBEDDING_COST_PER_1K_USD
        logger.debug(
            "Embedding batch [%d–%d]: %d tokens, $%.6f",
            start,
            start + len(chunk) - 1,
            tokens_used,
            cost,
        )
    return vectors


def _compute_similarity_scores(
    registered_endpoints: list[str],
    published_endpoints: list[str],
) -> list[float]:
    """
    Compute cosine similarity scores between registered and published endpoints.

    Uses the OpenAI Embeddings API (``text-embedding-3-small``).  All unique
    texts are embedded in a single batched call to minimise API
    round-trips.  Registered endpoints may contain multiple pipe-separated
    candidate measures from CT.gov; the best cosine score against the
    published endpoint is used for routing.

    Parameters
    ----------
    registered_endpoints:
        List of registered primary endpoint strings (one per trial).
    published_endpoints:
        Corresponding list of published primary endpoint strings from PubMed.

    Returns
    -------
    list[float]
        Cosine similarity score in [0.0, 1.0] for each trial pair.
        Returns 0.0 for pairs where either endpoint string is empty.
    """
    import openai  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    client = openai.OpenAI(api_key=api_key, base_url=LLM_BASE_URL)

    # Collect every unique text that needs embedding in one pass so we can
    # batch all requests together rather than calling the API per-pair.
    pair_candidates: list[tuple[list[str], str]] = []
    all_texts: list[str] = []
    text_index: dict[str, int] = {}  # text → index in all_texts (deduplication)

    for registered, published in zip(registered_endpoints, published_endpoints):
        candidates = _split_registered_candidates(registered)
        pub_text = _normalise(published)
        pair_candidates.append((candidates, pub_text))

        for text in candidates + ([pub_text] if pub_text else []):
            if text and text not in text_index:
                text_index[text] = len(all_texts)
                all_texts.append(text)

    if not all_texts:
        return [0.0] * len(registered_endpoints)

    logger.info(
        "Layer 1 — embedding %d unique endpoint strings via %s...",
        len(all_texts),
        EMBEDDING_MODEL,
    )
    all_vectors = _embed_batch(all_texts, client)

    scores: list[float] = []
    for candidates, pub_text in pair_candidates:
        if not candidates or not pub_text:
            scores.append(0.0)
            continue

        pub_vec = all_vectors[text_index[pub_text]]
        best = max(
            _cosine(all_vectors[text_index[c]], pub_vec) for c in candidates if c in text_index
        )
        scores.append(round(best, 4))

    return scores


def _route_from_score(score: float) -> EndpointRouting:
    if score >= SIMILARITY_AUTO_CONCORDANT:
        return EndpointRouting.AUTO_CONCORDANT
    if score >= SIMILARITY_LLM_LOWER:
        return EndpointRouting.LLM
    return EndpointRouting.AUTO_MAJOR_SWITCH


class _LLMCostTracker:
    """In-process LLM cost accumulator with a hard ceiling."""

    def __init__(self, ceiling: float = LLM_COST_CEILING_USD) -> None:
        self._ceiling = ceiling
        self._total_usd = 0.0
        self._total_calls = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost = (
            prompt_tokens / 1_000 * LLM_COST_PER_1K_INPUT_USD
            + completion_tokens / 1_000 * LLM_COST_PER_1K_OUTPUT_USD
        )
        self._total_usd += cost
        self._total_calls += 1
        logger.debug(
            "LLM call #%d - %d input / %d output tokens - $%.6f this call - $%.4f total",
            self._total_calls,
            prompt_tokens,
            completion_tokens,
            cost,
            self._total_usd,
        )
        if self._total_usd > self._ceiling:
            raise RuntimeError(
                f"LLM cost ceiling of ${self._ceiling:.2f} exceeded "
                f"(running total: ${self._total_usd:.4f} after {self._total_calls} calls)."
            )

    @property
    def total_usd(self) -> float:
        return round(self._total_usd, 6)

    @property
    def total_calls(self) -> int:
        return self._total_calls


_cost_tracker = _LLMCostTracker()


# The adjudicator prompt's free-text classification -> internal SwitchType.
_CLASSIFICATION_TO_SWITCH_TYPE = {
    "concordant (no change)": "concordant",
    "additional outcome (disclosed exploratory)": "additional_outcome",
    "outcome modification (timeframe/definition/population)": "minor_modification",
    "outcome switch — moderate / partly disclosed": "moderate_switch",
    "outcome switch - moderate / partly disclosed": "moderate_switch",
    "outcome switch — major / undisclosed": "major_switch",
    "outcome switch - major / undisclosed": "major_switch",
}

_SWITCH_TYPE_TO_CLASSIFICATION = {
    "concordant": "Concordant (no change)",
    "additional_outcome": "Additional outcome (disclosed exploratory)",
    "minor_modification": "Outcome modification (timeframe/definition/population)",
    "moderate_switch": "Outcome switch — moderate / partly disclosed",
    "major_switch": "Outcome switch — major / undisclosed",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_optional_bool(value: object) -> Optional[bool]:
    """Parse a tri-state model field without converting missing detail to False."""
    if value is None or str(value).strip().lower() in {
        "",
        "null",
        "none",
        "not stated",
        "not_stated",
        "unclear",
        "unknown",
    }:
        return None
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "same", "reported", "disclosed"}:
        return True
    if normalised in {"false", "0", "no", "different", "not reported", "undisclosed"}:
        return False
    return None


_CLASSIFICATION_SEVERITY = {
    "concordant": 0,
    "additional_outcome": 1,
    "minor_modification": 2,
    "moderate_switch": 3,
    "major_switch": 4,
}


def _classify_endpoint_comparisons(
    raw_comparisons: object,
) -> tuple[list[EndpointComparison], Optional[str], str, list[str]]:
    """Classify registered primary endpoints individually, then aggregate.

    The model supplies evidence extraction only. The five-class result follows
    the user's fixed decision tree. ``None`` attribute values mean "not stated"
    and therefore never trigger modification.
    """
    if not isinstance(raw_comparisons, list):
        return [], None, "unclear", []

    comparisons: list[EndpointComparison] = []
    internal_classes: list[str] = []
    guardrails: list[str] = []
    attribute_fields = (
        "same_timeframe",
        "same_definition",
        "same_population",
        "same_measurement_method",
    )

    for position, raw in enumerate(raw_comparisons, start=1):
        if not isinstance(raw, dict):
            guardrails.append(f"Endpoint {position}: malformed comparison was ignored.")
            continue

        evidence_raw = raw.get("actual_change_evidence") or []
        if not isinstance(evidence_raw, list):
            evidence_raw = [evidence_raw]
        evidence = [str(value).strip() for value in evidence_raw if str(value).strip()]
        values = {name: _as_optional_bool(raw.get(name)) for name in attribute_fields}
        same_construct = _as_optional_bool(raw.get("same_construct"))
        registered_reported = _as_optional_bool(raw.get("registered_endpoint_reported"))
        change_disclosed = _as_optional_bool(raw.get("change_disclosed"))
        additional_present = _as_bool(raw.get("additional_endpoint_present"))
        additional_disclosed = _as_bool(raw.get("additional_endpoint_disclosed"))
        different_attributes = [
            name.removeprefix("same_") for name, value in values.items() if value is False
        ]

        if registered_reported is False and evidence:
            if change_disclosed is True:
                endpoint_class = "moderate_switch"
            elif change_disclosed is False:
                endpoint_class = "major_switch"
            else:
                endpoint_class = "moderate_switch"
                guardrails.append(
                    f"Endpoint {position}: omission is evidenced but disclosure is unclear; major switch was not assigned."
                )
        elif same_construct is True:
            if different_attributes and evidence:
                endpoint_class = "minor_modification"
            elif different_attributes:
                endpoint_class = "concordant"
                guardrails.append(
                    f"Endpoint {position}: an attribute was marked different without affirmative evidence."
                )
            elif additional_present and additional_disclosed and registered_reported is not False:
                endpoint_class = "additional_outcome"
            else:
                endpoint_class = "concordant"
        elif same_construct is False:
            if additional_present and additional_disclosed and registered_reported is True:
                endpoint_class = "additional_outcome"
            elif not evidence:
                endpoint_class = "concordant"
                guardrails.append(
                    f"Endpoint {position}: a construct change was asserted without affirmative evidence."
                )
            elif change_disclosed is True or registered_reported is True:
                endpoint_class = "moderate_switch"
            elif registered_reported is False and change_disclosed is False:
                endpoint_class = "major_switch"
            else:
                endpoint_class = "moderate_switch"
                guardrails.append(
                    f"Endpoint {position}: omission or disclosure was unclear; major switch was not assigned."
                )
        else:
            endpoint_class = "concordant"
            guardrails.append(
                f"Endpoint {position}: clinical-construct equivalence was not established."
            )

        comparisons.append(
            EndpointComparison(
                registered_endpoint=str(raw.get("registered_endpoint", "")).strip(),
                published_corresponding_endpoint=str(
                    raw.get("published_corresponding_endpoint", "")
                ).strip(),
                same_construct=same_construct,
                **values,
                registered_endpoint_reported=registered_reported,
                additional_endpoint_present=additional_present,
                additional_endpoint_disclosed=additional_disclosed,
                change_disclosed=change_disclosed,
                actual_change_evidence=evidence,
                classification_label=_SWITCH_TYPE_TO_CLASSIFICATION[endpoint_class],
            )
        )
        internal_classes.append(endpoint_class)

    if not internal_classes:
        return comparisons, None, "unclear", guardrails

    overall = max(internal_classes, key=_CLASSIFICATION_SEVERITY.__getitem__)
    if any(
        item.same_construct is False
        or (item.registered_endpoint_reported is False and item.actual_change_evidence)
        for item in comparisons
    ):
        match = "different"
    elif any(
        any(getattr(item, field) is False for field in attribute_fields) for item in comparisons
    ):
        match = "modified"
    elif all(item.same_construct is True for item in comparisons):
        match = "clinically equivalent"
    else:
        match = "unclear"
    return comparisons, overall, match, guardrails


def _adjudication_to_schema(parsed: dict) -> LLMEndpointClassification:
    """Map the adjudicator prompt's JSON (Section 9) onto ``LLMEndpointClassification``.

    Every derived field follows deterministically from the adjudicator's own
    output — we never re-judge here, only translate.
    """
    raw_class = str(parsed.get("final_classification", "")).strip().lower()
    # tolerate hyphen/en-dash and trailing punctuation differences
    raw_class = raw_class.replace("–", "—").rstrip(".")
    switch_type = _CLASSIFICATION_TO_SWITCH_TYPE.get(raw_class)
    if switch_type is None:
        for key, value in _CLASSIFICATION_TO_SWITCH_TYPE.items():
            if raw_class and (raw_class in key or key in raw_class):
                switch_type = value
                break
    if switch_type is None:
        raise ValueError(
            f"Unrecognised final_classification: {parsed.get('final_classification')!r}"
        )

    endpoint_comparisons, structured_class, structured_match, structured_guardrails = (
        _classify_endpoint_comparisons(parsed.get("endpoint_comparisons"))
    )
    if structured_class is not None:
        switch_type = structured_class
    match = (
        structured_match
        if endpoint_comparisons
        else str(parsed.get("primary_endpoint_match", "")).strip().lower()
    )
    disclosure = str(parsed.get("disclosure_status", "")).strip().lower()
    endpoint_changed = _as_bool(parsed.get("endpoint_changed"))
    missing_detail_only = _as_bool(parsed.get("missing_detail_only"))
    raw_change_evidence = parsed.get("actual_change_evidence") or []
    if not isinstance(raw_change_evidence, list):
        raw_change_evidence = [raw_change_evidence]
    actual_change_evidence = [
        str(item).strip() for item in raw_change_evidence if str(item).strip()
    ]
    if endpoint_comparisons:
        actual_change_evidence = [
            evidence
            for comparison in endpoint_comparisons
            for evidence in comparison.actual_change_evidence
        ]
        endpoint_changed = switch_type in {
            "minor_modification",
            "moderate_switch",
            "major_switch",
        }
        if switch_type == "major_switch":
            disclosure = "undisclosed"
        elif switch_type == "moderate_switch" and any(
            comparison.change_disclosed is True for comparison in endpoint_comparisons
        ):
            disclosure = "partly disclosed"
        missing_detail_only = all(
            comparison.same_construct is not False
            and not comparison.actual_change_evidence
            and any(
                getattr(comparison, field) is None
                for field in (
                    "same_timeframe",
                    "same_definition",
                    "same_population",
                    "same_measurement_method",
                )
            )
            for comparison in endpoint_comparisons
        )
    guardrails: list[str] = list(structured_guardrails)

    # Deterministic safeguards implement the study's adjudication rule. The
    # model cannot convert missing abstract detail into a factual endpoint
    # change, and cannot call an equivalent endpoint a switch.
    if switch_type in {"minor_modification", "moderate_switch", "major_switch"}:
        if match in {"exact", "clinically equivalent"}:
            switch_type = "concordant"
            guardrails.append("Equivalent endpoint cannot be classified as changed.")
        elif missing_detail_only:
            switch_type = "concordant"
            guardrails.append("Missing abstract detail is not evidence of an endpoint change.")
        elif not actual_change_evidence:
            switch_type = "concordant"
            guardrails.append("No affirmative, source-grounded change evidence was supplied.")

    if switch_type == "major_switch" and not (
        endpoint_changed and match == "different" and disclosure == "undisclosed"
    ):
        switch_type = "moderate_switch"
        guardrails.append(
            "Major switch requires a different endpoint, explicit change evidence, and undisclosed replacement."
        )

    if switch_type == "moderate_switch" and match == "modified":
        switch_type = "minor_modification"
        guardrails.append("Same endpoint with an explicit feature change is outcome modification.")

    if switch_type == "additional_outcome" and match not in {"exact", "clinically equivalent"}:
        switch_type = "concordant"
        guardrails.append(
            "Additional-outcome classification requires the registered primary to remain intact."
        )

    is_switch = switch_type in _SWITCH_VERDICTS

    # confidence: the prompt returns a 0-1 number; be forgiving if it sent a word.
    try:
        conf_score = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        conf_score = {"high": 0.9, "medium": 0.65, "low": 0.35}.get(
            str(parsed.get("confidence", "")).strip().lower(), 0.5
        )
    conf_score = max(0.0, min(1.0, conf_score))
    if guardrails:
        conf_score = min(conf_score, 0.65)
    confidence = "high" if conf_score >= 0.8 else "medium" if conf_score >= 0.5 else "low"

    if switch_type == "concordant":
        direction = "none"
    elif switch_type == "additional_outcome":
        direction = "unregistered_added"
    elif switch_type == "minor_modification":
        direction = "definition_changed"
    elif "promot" in str(parsed.get("reasoning_summary", "")).lower():
        direction = "promotion_of_secondary"
    else:
        direction = "endpoint_replaced"

    reasoning = " ".join(
        s
        for s in (
            str(parsed.get("reasoning_summary", "")).strip(),
            str(parsed.get("registry_history_interpretation", "")).strip(),
        )
        if s
    ).strip()
    if guardrails:
        reasoning = " ".join([reasoning, "Guardrail applied: " + " ".join(guardrails)]).strip()
    if len(reasoning) < 20:
        reasoning = (
            f"Registered primary: {parsed.get('registered_primary_original') or parsed.get('registered_primary_current') or 'n/a'}. "
            f"Published primary: {parsed.get('published_primary') or 'n/a'}. "
            f"Match: {match or 'unclear'}. Classification: {switch_type}."
        )

    key_differences = [
        str(x) for x in (parsed.get("secondary_outcomes_missing") or []) if str(x).strip()
    ] + [str(x) for x in (parsed.get("additional_outcomes") or []) if str(x).strip()]
    for ev in (parsed.get("evidence_registered"), parsed.get("evidence_published")):
        if str(ev or "").strip():
            key_differences.append(str(ev).strip())

    forms: list[str] = []
    if switch_type in {"moderate_switch", "major_switch"}:
        forms.append(
            "primary_demoted"
            if disclosure in {"partly disclosed", "disclosed"}
            else "primary_replaced"
        )
        if any(
            comparison.registered_endpoint_reported is False for comparison in endpoint_comparisons
        ):
            forms.append("prespecified_omitted")
    if switch_type == "minor_modification":
        forms.append("definition_or_timeframe_changed")
    if switch_type == "additional_outcome":
        forms.append("unregistered_added_disclosed")

    return LLMEndpointClassification(
        switch_type=switch_type,
        classification_label=_SWITCH_TYPE_TO_CLASSIFICATION[switch_type],
        primary_endpoint_match=match or "unclear",
        published_primary_extracted=str(parsed.get("published_primary", "")).strip(),
        direction=direction,
        step_by_step_reasoning=reasoning,
        confidence=confidence,
        confidence_score=round(conf_score, 3),
        comparability_for_pooling=match in {"exact", "clinically equivalent"},
        flag_for_human_review=_as_bool(parsed.get("human_review_required")) or bool(guardrails),
        key_differences=key_differences[:6],
        disclosed_as_exploratory=(disclosure == "disclosed" or switch_type == "additional_outcome"),
        likely_results_driven=(
            is_switch and endpoint_changed and disclosure not in {"disclosed", "none needed"}
        ),
        switch_forms=forms,
        actual_change_evidence=actual_change_evidence,
        missing_detail_only=missing_detail_only,
        adjudication_guardrail=" ".join(guardrails),
        endpoint_comparisons=endpoint_comparisons,
    )


def _call_llm(
    registered_original: str,
    published_endpoint: str,
    bc_subtype: str = "unknown_subtype",
    bc_setting: str = "unknown_setting",
    *,
    registered_current: str = "",
    registered_secondary: str = "",
    published_results: str = "",
    published_conclusion: str = "",
    registry_history_note: str = "not available",
) -> Optional[LLMEndpointClassification]:
    """
    Submit one trial-publication pair to the configured OpenAI-compatible model
    for outcome-switching adjudication.

    The adjudicator is given the registered primary (original + current), the
    registered secondary outcomes, and the publication's Results + Conclusion
    text so it can identify the PUBLISHED primary itself rather than being handed
    a noisy pre-extracted snippet. See ``_LLM_SYSTEM_PROMPT``.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    if api_key.lower().startswith("your_") or "api_key_here" in api_key.lower():
        logger.error(
            "OPENAI_API_KEY appears to still be a placeholder value. "
            "Replace it in .env to enable LLM adjudication."
        )
        return None

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    reg_current = _normalise(registered_current)
    reg_original = _normalise(registered_original)
    user_content = _LLM_USER_TEMPLATE.format(
        bc_subtype=bc_subtype or "unknown_subtype",
        bc_setting=bc_setting or "unknown_setting",
        registered_original=reg_original or "(not available)",
        registered_current=reg_current or reg_original or "(not available)",
        registered_secondary=_normalise(registered_secondary) or "(none listed)",
        registry_history_note=registry_history_note or "not available",
        published_endpoint=_normalise(published_endpoint) or "(not separately extracted)",
        published_results=_normalise(published_results) or "(no Results section text)",
        published_conclusion=_normalise(published_conclusion) or "(no Conclusion text)",
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_PRIMARY,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as exc:
        logger.error("%s API error: %s", LLM_PROVIDER, exc)
        return None

    if response.usage:
        _cost_tracker.record(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    raw_text = (response.choices[0].message.content or "").strip()
    raw_text = re.sub(r"^```[a-z]*\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        parsed = json.loads(raw_text)
        return _adjudication_to_schema(parsed)
    except Exception as exc:
        logger.warning(
            "Malformed LLM response for endpoint pair; flagging for human review. "
            "Error: %s | Raw response (first 400 chars): %s",
            exc,
            raw_text[:400],
        )
        return None


def run_endpoint_matching(linked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Layers 1-2 for every High/Medium-confidence linked trial-publication pair.

    Required columns
    ----------------
    nct_id, pmid, linkage_confidence, primary_result_status, published_endpoint,
    and a registered endpoint: ``registered_primary_outcomes_for_comparison``
    (pre-recruitment/original snapshot) when available, otherwise
    ``primary_outcomes`` (current). ``endpoint_switch_assessable`` is optional
    and only affects logging.

    Notes
    -----
    Low-confidence and unlinked publication pairs are intentionally excluded
    here. They must be resolved in the linkage review workflow before they can
    enter endpoint matching or any downstream analysis.
    """
    decision_log = DecisionLog()

    # Registered endpoint for comparison: the pre-recruitment/original registry
    # snapshot when registry_history could retrieve it, otherwise the current
    # registered endpoint. `endpoint_switch_assessable` (carried into the
    # decision log) records which basis was used — a "concordant" verdict on a
    # current-only basis does not rule out a quiet registry edit.
    if "registered_primary_outcomes_for_comparison" in linked_df.columns:
        registered = linked_df["registered_primary_outcomes_for_comparison"].where(
            linked_df["registered_primary_outcomes_for_comparison"].apply(_normalise).astype(bool),
            linked_df.get("primary_outcomes", ""),
        )
    else:
        registered = linked_df.get("primary_outcomes", pd.Series("", index=linked_df.index))
    linked_df = linked_df.assign(_registered_endpoint_for_match=registered)
    endpoint_column = "_registered_endpoint_for_match"

    gate = (
        linked_df["pmid"].notna()
        & ~linked_df["pmid"].isin(["", "None"])
        & ~linked_df["linkage_confidence"].isin(["Unlinked"])
        & linked_df[endpoint_column].apply(_normalise).astype(bool)
        & linked_df["published_endpoint"].apply(_normalise).astype(bool)
    )
    if "primary_result_status" in linked_df.columns:
        # The linker is binary now: a SELECTED row has a committed primary paper.
        # Low-confidence picks still enter matching but carry human_review.
        gate &= linked_df["primary_result_status"].eq("SELECTED")

    if "endpoint_switch_assessable" in linked_df.columns:
        n_current_only = int(
            (
                gate
                & ~linked_df["endpoint_switch_assessable"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            ).sum()
        )
        if n_current_only:
            logger.warning(
                "%d pair(s) compared against the CURRENT registered endpoint only "
                "(no pre-recruitment registry snapshot available). A concordant result "
                "for these does not rule out registry-side endpoint editing.",
                n_current_only,
            )
    processable = linked_df[gate].copy()

    skipped_due_to_linkage = len(linked_df) - len(processable)
    if skipped_due_to_linkage:
        logger.info(
            "Endpoint matching skipped %d row(s) because linkage was unresolved, "
            "low-confidence, or missing a registered endpoint.",
            skipped_due_to_linkage,
        )

    if processable.empty:
        logger.warning(
            "run_endpoint_matching: no processable linked rows after linkage gate. "
            "Resolve Low/Unlinked publication links before running Module 2."
        )
        return linked_df

    registered_endpoints = processable[endpoint_column].fillna("").tolist()
    published_endpoints = processable["published_endpoint"].fillna("").tolist()

    logger.info(
        "Layer 1 - computing endpoint-embedding similarity for %d pairs "
        "(informational only in v4.1)...",
        len(processable),
    )
    scores = _compute_similarity_scores(registered_endpoints, published_endpoints)

    pair_ids: list[str] = []
    routings: list[str] = []
    n_llm_ok = n_llm_fail = n_auto = 0

    for idx, (_, row) in enumerate(processable.iterrows()):
        nct_id = str(row["nct_id"]).strip()
        pmid = str(row["pmid"]).strip()
        pair_id = f"{nct_id}_{pmid}"
        pair_ids.append(pair_id)

        registered = registered_endpoints[idx]
        published = published_endpoints[idx]
        score = scores[idx]
        # v4.1: every pair goes to the LLM; cosine no longer routes.
        routing = EndpointRouting.LLM if ENDPOINT_MATCHING_LLM_ONLY else _route_from_score(score)
        routings.append(routing.value)

        bc_subtype_val = str(row.get("bc_subtype", "unknown_subtype") or "unknown_subtype")
        bc_setting_val = str(row.get("bc_setting", "unknown_setting") or "unknown_setting")

        entry = DecisionLogEntry.from_layer1(
            pair_id=pair_id,
            registered_endpoint=registered,
            published_endpoint=published,
            similarity_score=score,
            routing=routing,
        )
        entry.bc_subtype = bc_subtype_val
        entry.bc_setting = bc_setting_val

        if routing == EndpointRouting.LLM:
            assessable = str(row.get("endpoint_switch_assessable", "")).strip().lower() in {
                "true",
                "1",
                "yes",
            }
            registry_source = str(row.get("registry_endpoint_source", "")).strip().lower()
            has_snapshot = assessable and registry_source in {"pre_recruitment", "original"}
            if has_snapshot:
                registry_note = (
                    "yes — a pre-recruitment registry snapshot was retrieved and is shown as "
                    "the original primary outcome"
                )
            elif assessable:
                registry_note = (
                    "partial — registry history retrieved but no distinct pre-results version"
                )
            else:
                registry_note = (
                    "no — only the CURRENT registry entry is available; treat it cautiously and "
                    "do not assign a major switch on the later wording alone"
                )
            llm_result = _call_llm(
                registered if has_snapshot else "",
                published,
                bc_subtype_val,
                bc_setting_val,
                registered_current=str(row.get("primary_outcomes", "") or ""),
                registered_secondary=str(row.get("secondary_outcomes", "") or ""),
                published_results=str(row.get("published_results", "") or ""),
                published_conclusion=str(row.get("published_conclusion", "") or ""),
                registry_history_note=registry_note,
            )
            entry.llm_model = LLM_MODEL_PRIMARY

            if llm_result is not None:
                entry.llm_switch_type = llm_result.switch_type
                entry.llm_reasoning = llm_result.step_by_step_reasoning
                entry.llm_confidence = llm_result.confidence
                entry.llm_confidence_score = round(llm_result.confidence_score, 3)
                entry.llm_comparability = llm_result.comparability_for_pooling
                entry.llm_flag = llm_result.flag_for_human_review
                entry.llm_disclosed_exploratory = llm_result.disclosed_as_exploratory
                entry.llm_results_driven = llm_result.likely_results_driven
                entry.llm_switch_forms = (
                    " | ".join(llm_result.switch_forms) if llm_result.switch_forms else ""
                )
                entry.llm_classification_label = llm_result.classification_label
                entry.llm_primary_endpoint_match = llm_result.primary_endpoint_match
                entry.llm_published_primary_extracted = llm_result.published_primary_extracted
                entry.llm_actual_change_evidence = " | ".join(llm_result.actual_change_evidence)
                entry.llm_missing_detail_only = llm_result.missing_detail_only
                entry.llm_adjudication_guardrail = llm_result.adjudication_guardrail
                entry.llm_endpoint_comparisons = json.dumps(
                    [item.model_dump(mode="json") for item in llm_result.endpoint_comparisons],
                    ensure_ascii=False,
                )
                n_llm_ok += 1

                # Human-in-the-loop policy: an outcome-SWITCH verdict is the
                # study's finding, so it always goes to a human. By default all
                # other model verdicts also require human review; auto-acceptance
                # is an explicit post-validation opt-in.
                is_switch = llm_result.switch_type in _SWITCH_VERDICTS
                if (
                    ENDPOINT_AUTO_ACCEPT
                    and not is_switch
                    and not llm_result.flag_for_human_review
                    and llm_result.confidence_score >= ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD
                    and _normalise(published)
                ):
                    entry.human_reviewed = HumanReviewStatus.AUTO_ACCEPTED
                    entry.human_decision = HumanDecision.CONFIRM
                    entry.human_final_class = llm_result.switch_type
                    entry.human_poolable = llm_result.comparability_for_pooling
                    entry.reviewer_initials = "AUTO"
                    n_auto += 1
            else:
                entry.llm_flag = True
                n_llm_fail += 1
                logger.warning(
                    "Layer 2 - invalid LLM response for %s. Pair flagged for human review.",
                    pair_id,
                )
            if (idx + 1) % 25 == 0:
                logger.info("  ...LLM-adjudicated %d/%d pairs", idx + 1, len(processable))

        decision_log.append(entry)

    logger.info(
        "Layer 2 - LLM adjudication: %d ok | %d auto-accepted | %d to human review | %d failed",
        n_llm_ok,
        n_auto,
        n_llm_ok - n_auto,
        n_llm_fail,
    )

    processable["pair_id"] = pair_ids
    processable["similarity_score"] = scores
    processable["routing"] = routings

    logger.info(
        "Endpoint matching complete - %d processed | %d LLM calls | $%.6f total LLM cost",
        len(processable),
        sum(1 for routing in routings if routing == EndpointRouting.LLM.value),
        _cost_tracker.total_usd,
    )
    logger.info("Governance snapshot: %s", decision_log.governance_summary())

    return linked_df.merge(
        processable[["nct_id", "pmid", "pair_id", "similarity_score", "routing"]],
        on=["nct_id", "pmid"],
        how="left",
    )
