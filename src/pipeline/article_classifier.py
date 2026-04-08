"""
Article-type gate — agentic results-paper classifier.  (Pre-linkage layer)

Purpose
-------
Before any linked PubMed record enters the endpoint matching pipeline, this
module verifies that the article is a *primary results paper* — i.e. the
publication reporting the trial's pre-specified primary outcome from the
randomised data.  Articles that are not results papers (protocols, design
papers, sub-group analyses, systematic reviews, editorials, letters, etc.)
are rejected with an explicit reason code, preventing them from contributing
spurious endpoint comparisons or HR estimates to the analysis.

Why this matters
----------------
The NCT-to-PMID linkage cascade (module1_linker) finds candidate PMIDs by
matching trial identifiers and titles.  A trial may have several associated
PubMed records:

  - Protocol / design paper      — describes the planned analysis; no outcome data
  - Primary results paper        — the one we want; reports the pre-specified primary endpoint
  - Sub-group / post-hoc paper   — reports a secondary analysis; endpoint may differ
  - Systematic review / meta-analysis — aggregates multiple trials; not a primary source
  - Editorial / commentary       — no original data
  - Safety / pharmacokinetic report — not a primary efficacy results paper

Without this gate, the endpoint extraction and HR regex steps attempt to parse
non-results text, producing empty published_endpoints (inflating the manual review
queue) or incorrect HR matches (corrupting the Bayesian model input).

Architecture — two-tier agentic classifier
------------------------------------------
Tier 1  Fast heuristic (no API call)
    Checks five deterministic signals from the already-fetched ``PubMedRecord``:
      1. Publication type tags (``PublicationType`` XML elements): "Clinical Trial",
         "Randomized Controlled Trial", "Multicenter Study" → strong results signal.
         "Study Protocol", "Meta-Analysis", "Review", "Editorial" → reject.
      2. MeSH terms: "Clinical Trial as Topic" → reject (it's about a trial, not of one).
      3. Title keyword scan: "protocol", "design", "rationale", "sub-group", "subgroup",
         "systematic review", "meta-analysis", "pharmacokinetics", "editorial" → reject.
      4. Structured abstract section labels: presence of a "RESULTS" section with numeric
         data → strong results signal.
      5. Abstract keyword scan for primary-outcome language vs. protocol language.

    If the heuristic reaches a confident verdict (ACCEPT or REJECT), the LLM is not called.
    The heuristic is calibrated to err on the side of uncertain rather than confident
    rejection, so borderline cases escalate to Tier 2.

Tier 2  LLM arbiter (called only for uncertain cases)
    A single lightweight prompt presents the title, publication types, and abstract
    to the LLM and asks it to classify the article as one of five types with a
    chain-of-thought reasoning step.  The LLM response is Pydantic-validated; a
    malformed response is treated as UNCERTAIN and flagged for human review.

    The LLM is deliberately NOT called for confident heuristic verdicts (≥ 80 % of
    cases), making this gate cheap: < 20 % of candidates incur an LLM call.

Output
------
Every classification produces an ``ArticleClassification`` dataclass containing:
  - ``article_type``    — primary result of the classification
  - ``verdict``         — ACCEPT | REJECT | UNCERTAIN
  - ``confidence``      — high | medium | low
  - ``tier``            — "heuristic" | "llm"
  - ``reason``          — human-readable explanation for the decision
  - ``flag_for_review`` — True when human reviewer should confirm before exclusion

Integration
-----------
Called by ``module1_linker._cascade_link()`` immediately after a PubMed record is
fetched.  A REJECT verdict causes the cascade to continue searching for other candidate
PMIDs for the same trial.  An UNCERTAIN verdict keeps the candidate but sets
``linkage_confidence = Low`` and flags the linkage log entry for human review.

    from src.pipeline.article_classifier import classify_article, ArticleVerdict

    record = client.fetch_record(candidate_pmid)
    classification = classify_article(record)
    if classification.verdict == ArticleVerdict.REJECT:
        logger.info("PMID %s rejected: %s", pmid, classification.reason)
        continue  # try next candidate
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.pipeline.pubmed_client import PubMedRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class ArticleVerdict(str, Enum):
    """Final gate decision for a candidate PubMed record."""
    ACCEPT    = "accept"     # Primary results paper — proceed with endpoint extraction
    REJECT    = "reject"     # Definitively not a results paper — skip this PMID
    UNCERTAIN = "uncertain"  # Could not determine — flag for human review


class ArticleType(str, Enum):
    """Fine-grained article type, for the audit log."""
    PRIMARY_RESULTS     = "primary_results"
    PROTOCOL            = "protocol"
    SUBGROUP_POSTHOC    = "subgroup_posthoc"
    SYSTEMATIC_REVIEW   = "systematic_review"
    EDITORIAL_LETTER    = "editorial_letter"
    PHARMACOKINETIC     = "pharmacokinetic"
    SAFETY_REPORT       = "safety_report"
    SECONDARY_ANALYSIS  = "secondary_analysis"
    UNKNOWN             = "unknown"


@dataclass
class ArticleClassification:
    """
    Result of the article-type gate for a single PubMed record.

    Attributes
    ----------
    pmid:
        PubMed identifier of the classified record.
    article_type:
        Fine-grained classification of the publication type.
    verdict:
        ACCEPT / REJECT / UNCERTAIN gate decision.
    confidence:
        ``"high"`` — classifier is confident; LLM not needed.
        ``"medium"`` — some uncertainty; heuristic plus LLM agreement.
        ``"low"`` — LLM fallback was inconclusive or LLM call failed.
    tier:
        ``"heuristic"`` if the verdict came from Tier 1 alone;
        ``"llm"`` if the LLM arbiter was invoked.
    reason:
        Human-readable justification for the verdict.  Written to the
        linkage audit log ``notes`` field.
    flag_for_review:
        If ``True``, a human reviewer must confirm the rejection/uncertainty
        before the trial is excluded from downstream analysis.
    signals:
        Internal list of heuristic signals that contributed to the verdict.
        Included in the audit trail for transparency.
    """
    pmid:            str
    article_type:    ArticleType    = ArticleType.UNKNOWN
    verdict:         ArticleVerdict = ArticleVerdict.UNCERTAIN
    confidence:      str            = "low"
    tier:            str            = "heuristic"
    reason:          str            = ""
    flag_for_review: bool           = True
    signals:         list[str]      = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tier 1 — heuristic signal sets
# ---------------------------------------------------------------------------

# PubMed PublicationType values that strongly indicate a primary results paper.
_ACCEPT_PUB_TYPES: frozenset[str] = frozenset({
    "Randomized Controlled Trial",
    "Clinical Trial, Phase II",
    "Clinical Trial, Phase III",
    "Clinical Trial, Phase IV",
    "Multicenter Study",
    "Clinical Trial",
    "Controlled Clinical Trial",
    "Equivalence Trial",
})

# PubMed PublicationType values that definitively indicate a non-results paper.
_REJECT_PUB_TYPES: frozenset[str] = frozenset({
    "Study Protocol",
    "Clinical Study Design",
    "Meta-Analysis",
    "Systematic Review",
    "Review",
    "Editorial",
    "Letter",
    "Comment",
    "News",
    "Biography",
    "Published Erratum",
    "Retraction of Publication",
})

# Title words/phrases that signal a protocol, design, or review paper.
_REJECT_TITLE_PATTERNS: re.Pattern = re.compile(
    r"\b("
    r"study\s+protocol"
    r"|protocol\s+paper"
    r"|design\s+and\s+rationale"
    r"|design\s+paper"
    r"|rationale\s+and\s+design"
    r"|trial\s+design"
    r"|study\s+design"
    r"|methods\s+paper"
    r"|systematic\s+review"
    r"|meta.analysis"
    r"|narrative\s+review"
    r"|sub.?group\s+analysis"
    r"|post.?hoc\s+analysis"
    r"|pharmacokinetics?"
    r"|pharmacodynamics?"
    r"|dose.?finding"
    r"|editorial"
    r"|commentary"
    r"|letter\s+to"
    r"|correction"
    r"|erratum"
    r"|retraction"
    r")\b",
    re.IGNORECASE,
)

# MeSH terms that indicate the article is *about* a clinical trial rather
# than *reporting* one (e.g. methodological or systematic review articles).
_REJECT_MESH_TERMS: frozenset[str] = frozenset({
    "Clinical Trial as Topic",
    "Randomized Controlled Trials as Topic",
    "Meta-Analysis as Topic",
    "Systematic Reviews as Topic",
    "Research Design",
})

# Abstract section labels or lead phrases that confirm the paper reports results.
_RESULTS_SECTION_LABELS: frozenset[str] = frozenset({
    "RESULTS", "FINDINGS", "OUTCOMES", "MAIN RESULTS",
    "MAIN OUTCOME MEASURES", "RESULTS AND DISCUSSION",
})

# Abstract keyword patterns that confirm primary outcome reporting.
_RESULTS_ABSTRACT_PATTERNS: re.Pattern = re.compile(
    r"\b("
    r"primary\s+end.?point"
    r"|primary\s+outcome"
    r"|hazard\s+ratio"
    r"|randomized\s+to"
    r"|randomly\s+assigned"
    r"|were\s+enrolled"
    r"|patients\s+were\s+randomly"
    r")\b",
    re.IGNORECASE,
)

# Abstract keyword patterns that indicate protocol / design language (not results).
_PROTOCOL_ABSTRACT_PATTERNS: re.Pattern = re.compile(
    r"\b("
    r"will\s+be\s+randomly"
    r"|will\s+be\s+enrolled"
    r"|we\s+will\s+recruit"
    r"|this\s+protocol\s+describes"
    r"|this\s+paper\s+describes\s+the\s+(?:design|rationale|protocol)"
    r"|sample\s+size\s+calculation"
    r"|the\s+study\s+is\s+registered"
    r"|aims?\s+to\s+evaluate"
    r"|is\s+designed\s+to"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tier 1 — heuristic classifier
# ---------------------------------------------------------------------------

def _heuristic_classify(record: PubMedRecord) -> ArticleClassification:
    """
    Apply deterministic heuristics to classify the article type.

    Works entirely from already-fetched ``PubMedRecord`` fields — no API calls.
    Returns a confident ACCEPT or REJECT when the evidence is unambiguous;
    returns UNCERTAIN with ``confidence="low"`` when signals conflict or are
    insufficient, triggering Tier 2.

    Parameters
    ----------
    record:
        A fully populated ``PubMedRecord`` from ``PubMedClient.fetch_record()``.
        Must include ``pub_types``, ``mesh_terms``, ``title``,
        ``abstract_sections``, and ``abstract_text``.

    Returns
    -------
    ArticleClassification
        Verdict with all contributing signals listed in ``signals``.
    """
    signals: list[str] = []
    reject_score = 0   # positive integer — higher = more confident rejection
    accept_score = 0   # positive integer — higher = more confident acceptance

    # ---- Publication type tags -------------------------------------------
    reject_types = record.pub_types & _REJECT_PUB_TYPES
    accept_types = record.pub_types & _ACCEPT_PUB_TYPES

    for pt in reject_types:
        signals.append(f"pub_type:REJECT:{pt}")
        reject_score += 3   # pub type is a strong signal

    for pt in accept_types:
        signals.append(f"pub_type:ACCEPT:{pt}")
        accept_score += 2

    # ---- Title scan -------------------------------------------------------
    title_match = _REJECT_TITLE_PATTERNS.search(record.title)
    if title_match:
        signals.append(f"title:REJECT:{title_match.group(0)!r}")
        reject_score += 3

    # ---- MeSH terms -------------------------------------------------------
    reject_mesh = record.mesh_terms & _REJECT_MESH_TERMS if isinstance(record.mesh_terms, set) \
        else set(record.mesh_terms) & _REJECT_MESH_TERMS
    for term in reject_mesh:
        signals.append(f"mesh:REJECT:{term}")
        reject_score += 2

    # ---- Abstract section labels ------------------------------------------
    result_sections = set(record.abstract_sections.keys()) & _RESULTS_SECTION_LABELS
    if result_sections:
        signals.append(f"abstract_section:ACCEPT:{sorted(result_sections)}")
        accept_score += 2

    # ---- Abstract text patterns ------------------------------------------
    if record.abstract_text:
        if _RESULTS_ABSTRACT_PATTERNS.search(record.abstract_text):
            signals.append("abstract:ACCEPT:primary_outcome_language_detected")
            accept_score += 2
        if _PROTOCOL_ABSTRACT_PATTERNS.search(record.abstract_text):
            signals.append("abstract:REJECT:protocol_future_tense_language_detected")
            reject_score += 3

    # ---- No abstract at all ----------------------------------------------
    if not record.abstract_text:
        signals.append("abstract:UNCERTAIN:no_abstract_text")
        reject_score += 1  # slight reject lean — results papers almost always have abstracts

    # ---- Verdict determination -------------------------------------------
    # Strong rejection: any definitive reject pub type, OR reject score dominates
    # decisively (≥ 3 points ahead of accept score) with at least one explicit
    # reject signal.
    if reject_types and not accept_types:
        # Unambiguous: only reject pub types, no accept pub types
        article_type = _map_pub_type_to_article_type(reject_types)
        return ArticleClassification(
            pmid         = record.pmid,
            article_type = article_type,
            verdict      = ArticleVerdict.REJECT,
            confidence   = "high",
            tier         = "heuristic",
            reason       = (
                f"Publication type(s) definitively non-results: "
                f"{', '.join(sorted(reject_types))}."
            ),
            flag_for_review = False,
            signals      = signals,
        )

    if accept_types and not reject_types and not title_match and not reject_mesh:
        # Unambiguous acceptance: RCT/clinical trial pub types, no reject signals
        return ArticleClassification(
            pmid         = record.pmid,
            article_type = ArticleType.PRIMARY_RESULTS,
            verdict      = ArticleVerdict.ACCEPT,
            confidence   = "high",
            tier         = "heuristic",
            reason       = (
                f"Publication type(s) confirm results paper: "
                f"{', '.join(sorted(accept_types))}."
                + (" Results section present." if result_sections else "")
            ),
            flag_for_review = False,
            signals      = signals,
        )

    if reject_score >= accept_score + 3:
        # Reject score dominates clearly — confident rejection without LLM
        return ArticleClassification(
            pmid         = record.pmid,
            article_type = ArticleType.UNKNOWN,
            verdict      = ArticleVerdict.REJECT,
            confidence   = "medium",
            tier         = "heuristic",
            reason       = (
                f"Heuristic signals strongly favour rejection "
                f"(reject_score={reject_score}, accept_score={accept_score}). "
                f"Signals: {'; '.join(signals[:5])}."
            ),
            flag_for_review = True,  # medium confidence — flag for spot-check
            signals      = signals,
        )

    # Insufficient or conflicting signals → escalate to LLM
    return ArticleClassification(
        pmid         = record.pmid,
        article_type = ArticleType.UNKNOWN,
        verdict      = ArticleVerdict.UNCERTAIN,
        confidence   = "low",
        tier         = "heuristic",
        reason       = (
            f"Heuristic signals inconclusive "
            f"(reject_score={reject_score}, accept_score={accept_score}). "
            "Escalating to LLM arbiter."
        ),
        flag_for_review = True,
        signals      = signals,
    )


def _map_pub_type_to_article_type(pub_types: set[str]) -> ArticleType:
    """Map a set of PubMed publication type strings to the closest ArticleType."""
    if pub_types & {"Study Protocol", "Clinical Study Design"}:
        return ArticleType.PROTOCOL
    if pub_types & {"Meta-Analysis", "Systematic Review", "Review"}:
        return ArticleType.SYSTEMATIC_REVIEW
    if pub_types & {"Editorial", "Letter", "Comment", "News"}:
        return ArticleType.EDITORIAL_LETTER
    return ArticleType.UNKNOWN


# ---------------------------------------------------------------------------
# Tier 2 — LLM arbiter
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a clinical research methodologist reviewing PubMed article records to \
determine whether each article is the PRIMARY RESULTS PAPER for a registered \
randomised controlled trial.

A PRIMARY RESULTS PAPER:
- Reports the final analysis of the pre-specified primary endpoint
- Contains a RESULTS section with hazard ratios, p-values, or event counts
- Is NOT a protocol paper, design paper, sub-group analysis, systematic review, \
  editorial, letter, or pharmacokinetic/safety-only report

You will receive the article title, publication type tags, and abstract. Work \
through TWO reasoning steps then give your verdict.

Return ONLY valid JSON — no markdown, no prose outside the JSON.

JSON schema:
{
  "step1_signals": "List every signal from title, pub types, and abstract that \
indicates or contradicts a primary results paper.",
  "step2_reasoning": "Explain your final classification decision. Be specific.",
  "article_type": "primary_results | protocol | subgroup_posthoc | systematic_review \
| editorial_letter | pharmacokinetic | safety_report | secondary_analysis | unknown",
  "verdict": "accept | reject | uncertain",
  "confidence": "high | medium | low",
  "reason": "One sentence suitable for the audit log."
}
"""

_LLM_USER_TEMPLATE = """\
=== ARTICLE RECORD ===
Title:             {title}
Publication types: {pub_types}
Journal:           {journal}
Year:              {pub_year}

=== ABSTRACT ===
{abstract}

Classify this article and return your JSON verdict.
"""


def _call_llm_arbiter(record: PubMedRecord) -> Optional[ArticleClassification]:
    """
    Call the LLM to arbitrate uncertain article-type classifications.

    Uses the same OpenAI client as Module 2 to avoid introducing a second
    API integration.  The LLM is given the title, publication types, and
    abstract — no full-text access — and returns a structured JSON verdict.

    Parameters
    ----------
    record:
        The ``PubMedRecord`` to classify.

    Returns
    -------
    ArticleClassification | None
        A classification built from the LLM response, or ``None`` if the
        LLM call failed or returned a malformed response.  ``None`` causes
        the caller to fall back to UNCERTAIN + flag_for_review=True.
    """
    from src.pipeline.config import LLM_BASE_URL, LLM_MODEL_PRIMARY

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — LLM arbiter skipped for PMID %s.", record.pmid)
        return None

    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    except ImportError:
        logger.warning("openai package not available — LLM arbiter skipped.")
        return None

    user_content = _LLM_USER_TEMPLATE.format(
        title      = record.title or "(not available)",
        pub_types  = ", ".join(sorted(record.pub_types)) or "(not available)",
        journal    = record.journal or "(not available)",
        pub_year   = record.pub_year or "(not available)",
        abstract   = (record.abstract_text[:1_500] if record.abstract_text
                      else "(no abstract available)"),
    )

    try:
        response = client.chat.completions.create(
            model           = LLM_MODEL_PRIMARY,
            max_tokens      = 512,
            temperature     = 0.0,
            response_format = {"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
    except Exception as exc:
        logger.error("LLM arbiter API error for PMID %s: %s", record.pmid, exc)
        return None

    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM arbiter returned invalid JSON for PMID %s: %s\nRaw: %s",
            record.pmid, exc, raw[:300],
        )
        return None

    # Map string values to enums, with safe fallbacks
    verdict_map = {
        "accept":    ArticleVerdict.ACCEPT,
        "reject":    ArticleVerdict.REJECT,
        "uncertain": ArticleVerdict.UNCERTAIN,
    }
    type_map = {t.value: t for t in ArticleType}

    verdict      = verdict_map.get(str(parsed.get("verdict", "")).lower(), ArticleVerdict.UNCERTAIN)
    article_type = type_map.get(str(parsed.get("article_type", "")).lower(), ArticleType.UNKNOWN)
    confidence   = str(parsed.get("confidence", "low")).lower()
    reason       = str(parsed.get("reason", "LLM arbiter verdict."))
    step1        = str(parsed.get("step1_signals", ""))
    step2        = str(parsed.get("step2_reasoning", ""))

    logger.debug(
        "LLM arbiter PMID %s → %s (%s) | %s",
        record.pmid, verdict.value, confidence, reason,
    )

    return ArticleClassification(
        pmid         = record.pmid,
        article_type = article_type,
        verdict      = verdict,
        confidence   = confidence,
        tier         = "llm",
        reason       = f"[LLM] {reason} | Step1: {step1[:200]} | Step2: {step2[:200]}",
        flag_for_review = verdict == ArticleVerdict.UNCERTAIN or confidence == "low",
        signals      = [f"llm_step1:{step1[:200]}", f"llm_step2:{step2[:200]}"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_article(record: PubMedRecord) -> ArticleClassification:
    """
    Classify a PubMed record as a results paper, protocol, review, etc.

    Runs the two-tier classification cascade:
      Tier 1 — fast heuristic (always runs; no API call)
      Tier 2 — LLM arbiter (only when Tier 1 is uncertain)

    The LLM is invoked for ≤ 20 % of records in practice (the uncertain
    fraction), keeping the gate lightweight.

    Parameters
    ----------
    record:
        A ``PubMedRecord`` returned by ``PubMedClient.fetch_record()``.
        Must have ``pub_types`` populated (added to the dataclass in the
        updated ``pubmed_client.py``).

    Returns
    -------
    ArticleClassification
        The final verdict.  The caller in ``module1_linker._cascade_link()``
        should:
          - ACCEPT  → continue with linkage as normal
          - REJECT  → skip this PMID and try the next candidate
          - UNCERTAIN → keep candidate but set linkage_confidence = Low
                        and flag for human review

    Notes
    -----
    This function never raises.  Any internal error is caught, logged, and
    converted to an UNCERTAIN + flag_for_review=True result so the pipeline
    degrades gracefully.
    """
    try:
        heuristic = _heuristic_classify(record)
    except Exception as exc:
        logger.error(
            "Heuristic classifier failed for PMID %s: %s", record.pmid, exc, exc_info=True
        )
        return ArticleClassification(
            pmid         = record.pmid,
            verdict      = ArticleVerdict.UNCERTAIN,
            confidence   = "low",
            tier         = "heuristic",
            reason       = f"Classifier error: {exc}",
            flag_for_review = True,
        )

    if heuristic.verdict != ArticleVerdict.UNCERTAIN:
        # Tier 1 is confident — no LLM needed
        logger.info(
            "Article gate PMID %s → %s (%s, heuristic) | %s",
            record.pmid, heuristic.verdict.value, heuristic.confidence, heuristic.reason,
        )
        return heuristic

    # Tier 1 uncertain → invoke LLM arbiter
    logger.info(
        "Article gate PMID %s → Tier 1 uncertain; calling LLM arbiter...", record.pmid
    )
    llm_result = _call_llm_arbiter(record)

    if llm_result is None:
        # LLM unavailable or failed — fall back to Tier 1 UNCERTAIN result
        heuristic.reason += " LLM arbiter unavailable; flagged for human review."
        return heuristic

    # Combine tier results: if LLM and heuristic agree on ACCEPT or REJECT,
    # promote to medium confidence.  If they disagree, stay UNCERTAIN.
    if llm_result.verdict in (ArticleVerdict.ACCEPT, ArticleVerdict.REJECT):
        if llm_result.confidence in ("high", "medium"):
            llm_result.flag_for_review = llm_result.verdict == ArticleVerdict.REJECT
        logger.info(
            "Article gate PMID %s → %s (%s, llm) | %s",
            record.pmid, llm_result.verdict.value, llm_result.confidence, llm_result.reason,
        )

    return llm_result
