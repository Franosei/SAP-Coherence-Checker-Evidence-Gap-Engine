"""
PubMed / NCBI E-utilities client.

Provides a thin, typed interface over the NCBI E-utilities REST API for the
two operations the pipeline requires:

  esearch  — find PubMed article identifiers (PMIDs) by NCT ID secondary
             identifier, by free-text title query, or by author + year.
  efetch   — retrieve a full PubMed article record (XML), parse it into a
             structured ``PubMedRecord`` dataclass.

Rate-limiting
-------------
NCBI enforces a hard ceiling on unauthenticated requests (≤ 3 req/s).
Supplying ``NCBI_API_KEY`` in ``.env`` raises this to ≤ 10 req/s.  The
sleep interval is read from ``config.PUBMED_RATE_LIMIT_S`` which is set
dynamically based on whether the key is present.

Error handling
--------------
All public functions raise ``requests.HTTPError`` on non-2xx responses and
``xml.etree.ElementTree.ParseError`` on malformed XML.  Callers that need
graceful degradation (e.g. the linkage cascade) should catch these explicitly
rather than catching bare ``Exception``.

Usage
-----
    from src.pipeline.pubmed_client import PubMedClient, jaccard_token_similarity

    client = PubMedClient()
    pmids  = client.search_by_nct_id("NCT01520558")
    record = client.fetch_record(pmids[0])
    score  = jaccard_token_similarity(trial_title, record.title)
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.pipeline.config import (
    NCBI_API_KEY,
    PUBMED_BASE_URL,
    PUBMED_RATE_LIMIT_S,
    PUBMED_REQUEST_TIMEOUT_S,
    PUBMED_SEARCH_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PubMedRecord:
    """
    Structured representation of a single PubMed article.

    Attributes
    ----------
    pmid:
        PubMed unique identifier.
    title:
        Article title, with XML tags stripped.
    abstract_text:
        Full abstract as a single string.  For structured abstracts (common
        in RCT publications) the labelled sections are joined with newlines
        so that downstream regex patterns can search across section boundaries.
    abstract_sections:
        Mapping of section label (e.g. ``"RESULTS"``, ``"METHODS"``) to the
        corresponding abstract text.  Empty for unstructured abstracts.
    authors:
        List of author last names in publication order.
    pub_year:
        Four-digit publication year as a string, or ``""`` if absent.
    journal:
        Journal name (MedlineTA abbreviation preferred, full title fallback).
    mesh_terms:
        MeSH descriptor names, useful for endpoint domain classification.
    """

    pmid: str
    title: str
    abstract_text: str
    abstract_sections: dict[str, str] = field(default_factory=dict)
    authors: list[str] = field(default_factory=list)
    pub_year: str = ""
    journal: str = ""
    mesh_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------

def jaccard_token_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between the token sets of two strings.

    Tokens are lower-cased alphabetic words (punctuation stripped).  This
    mirrors the approach specified in Section 3.2.2 of the proposal for
    title-based fuzzy matching in the NCT-to-PMID linkage cascade.

    Parameters
    ----------
    text_a, text_b:
        The two strings to compare.

    Returns
    -------
    float
        Value in [0.0, 1.0].  Returns 0.0 when either string is empty.

    Examples
    --------
    >>> jaccard_token_similarity(
    ...     "PARADIGM-HF: LCZ696 versus Enalapril in HFrEF",
    ...     "Angiotensin receptor neprilysin inhibition versus enalapril HFrEF",
    ... )
    0.2857...
    """
    def _tokenise(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 1}

    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


# ---------------------------------------------------------------------------
# PubMed API client
# ---------------------------------------------------------------------------

class PubMedClient:
    """
    Stateful NCBI E-utilities client with integrated rate limiting.

    A single instance is intended to be shared across the entire NCT-to-PMID
    linkage run so that the rate-limit sleep is applied consistently.

    Parameters
    ----------
    api_key:
        NCBI API key.  Defaults to the value of ``NCBI_API_KEY`` in config.
    rate_limit_s:
        Minimum seconds between successive API requests.  Defaults to
        ``PUBMED_RATE_LIMIT_S`` (0.11 s with key, 0.34 s without).
    """

    def __init__(
        self,
        api_key: str = NCBI_API_KEY,
        rate_limit_s: float = PUBMED_RATE_LIMIT_S,
    ) -> None:
        self._api_key      = api_key
        self._rate_limit_s = rate_limit_s
        self._last_call_ts: float = 0.0
        self._session      = requests.Session()
        self._session.headers.update({"User-Agent": "SAP-Coherence-Checker/2.0 (research)"})

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_by_nct_id(self, nct_id: str) -> list[str]:
        """
        Search PubMed for articles whose secondary identifier matches *nct_id*.

        This is Stage 1 of the NCT-to-PMID linkage cascade (Section 3.2.2).
        The ``[si]`` field tag targets the secondary identifier field, which is
        populated by investigators when they submit trial results to PubMed.

        Parameters
        ----------
        nct_id:
            ClinicalTrials.gov NCT identifier, e.g. ``"NCT01520558"``.

        Returns
        -------
        list[str]
            PMIDs found, ordered by PubMed relevance score.  Empty list if
            no match.
        """
        term = f"{nct_id}[si]"
        pmids = self._esearch(term, max_results=PUBMED_SEARCH_MAX_RESULTS)
        logger.debug("search_by_nct_id(%s) → %d result(s): %s", nct_id, len(pmids), pmids)
        return pmids

    def search_by_title(self, title: str, max_results: int = PUBMED_SEARCH_MAX_RESULTS) -> list[str]:
        """
        Search PubMed using title words to find candidate articles.

        This is Stage 2 of the NCT-to-PMID linkage cascade.  The query uses
        the ``[title]`` field tag to restrict matches to article titles, which
        reduces false positives from abstracts containing similar language.

        Callers should subsequently call :func:`jaccard_token_similarity` to
        score each candidate against the registered trial title before
        accepting a match.

        Parameters
        ----------
        title:
            Trial official or brief title from ClinicalTrials.gov.
        max_results:
            Maximum number of PMIDs to return.

        Returns
        -------
        list[str]
            Candidate PMIDs, ordered by PubMed relevance.
        """
        # Strip non-alphanumeric characters that would break the esearch query
        clean = re.sub(r"[^\w\s]", " ", title).strip()
        if not clean:
            return []

        # Use the first 12 significant words to keep the query focused and
        # avoid hitting the URL length limit for very long titles.
        words = [w for w in clean.split() if len(w) > 2][:12]
        term = " ".join(words) + "[title]"

        pmids = self._esearch(term, max_results=max_results)
        logger.debug("search_by_title(%r…) → %d result(s)", title[:60], len(pmids))
        return pmids

    def search_by_author_and_year(self, last_name: str, pub_year: str) -> list[str]:
        """
        Search PubMed for articles by a specific first author in a given year.

        This is Stage 3 of the NCT-to-PMID linkage cascade, applied to
        disambiguate medium-confidence title matches.

        Parameters
        ----------
        last_name:
            First author's last (family) name from a candidate PubMed record.
        pub_year:
            Four-digit publication year string, e.g. ``"2021"``.

        Returns
        -------
        list[str]
            Candidate PMIDs.
        """
        term = f"{last_name}[1au] AND {pub_year}[dp]"
        pmids = self._esearch(term, max_results=PUBMED_SEARCH_MAX_RESULTS)
        logger.debug(
            "search_by_author_and_year(%s, %s) → %d result(s)", last_name, pub_year, len(pmids)
        )
        return pmids

    # ------------------------------------------------------------------
    # Record fetching
    # ------------------------------------------------------------------

    def fetch_record(self, pmid: str) -> PubMedRecord:
        """
        Retrieve and parse a full PubMed article record by PMID.

        Fetches the article XML from NCBI efetch and returns a
        :class:`PubMedRecord` with all fields populated.

        Parameters
        ----------
        pmid:
            The PubMed identifier to fetch.

        Returns
        -------
        PubMedRecord
            Parsed article data.

        Raises
        ------
        requests.HTTPError
            If the NCBI server returns a non-2xx status.
        xml.etree.ElementTree.ParseError
            If the response body is not valid XML.
        """
        url    = f"{PUBMED_BASE_URL}/efetch.fcgi"
        params = {
            "db":      "pubmed",
            "id":      pmid,
            "rettype": "xml",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        response = self._get(url, params)
        root     = ET.fromstring(response.text)
        record   = self._parse_article(root, pmid)
        logger.debug("fetch_record(%s) → title=%r", pmid, record.title[:60])
        return record

    # ------------------------------------------------------------------
    # Private helpers — NCBI HTTP layer
    # ------------------------------------------------------------------

    def _esearch(self, term: str, max_results: int) -> list[str]:
        """Execute an esearch query and return a list of PMIDs."""
        url    = f"{PUBMED_BASE_URL}/esearch.fcgi"
        params = {
            "db":       "pubmed",
            "term":     term,
            "retmax":   max_results,
            "retmode":  "json",
            "usehistory": "n",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        response = self._get(url, params)
        data     = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _get(self, url: str, params: dict) -> requests.Response:
        """
        Issue a rate-limited GET request.

        Sleeps for ``self._rate_limit_s`` seconds since the last call before
        dispatching the request, ensuring the pipeline stays within NCBI's
        stated rate limits regardless of how quickly the caller invokes this
        method.
        """
        elapsed   = time.monotonic() - self._last_call_ts
        remaining = self._rate_limit_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

        response          = self._session.get(url, params=params, timeout=PUBMED_REQUEST_TIMEOUT_S)
        self._last_call_ts = time.monotonic()
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Private helpers — XML parsing
    # ------------------------------------------------------------------

    def _parse_article(self, root: ET.Element, pmid: str) -> PubMedRecord:
        """
        Parse a PubMed XML document into a :class:`PubMedRecord`.

        Handles both structured abstracts (labelled ``AbstractText`` elements)
        and unstructured abstracts (a single ``AbstractText`` element with no
        label attribute).
        """
        article = (
            root.find(".//MedlineCitation/Article")
            or root.find(".//Article")
        )
        if article is None:
            logger.warning("fetch_record(%s): no <Article> element found in XML", pmid)
            return PubMedRecord(pmid=pmid, title="", abstract_text="")

        title   = self._text(article.find("ArticleTitle"))
        authors = self._parse_authors(article)
        journal = self._parse_journal(root)
        pub_year = self._parse_pub_year(article, root)
        abstract_sections, abstract_text = self._parse_abstract(article)
        mesh_terms = self._parse_mesh(root)

        return PubMedRecord(
            pmid              = pmid,
            title             = title,
            abstract_text     = abstract_text,
            abstract_sections = abstract_sections,
            authors           = authors,
            pub_year          = pub_year,
            journal           = journal,
            mesh_terms        = mesh_terms,
        )

    @staticmethod
    def _text(element: Optional[ET.Element]) -> str:
        """Return the full text content of an XML element, stripping inner tags."""
        if element is None:
            return ""
        # itertext() yields text from the element and all descendants,
        # including text inside child tags (e.g. <i>, <b>, <sup>).
        return " ".join("".join(element.itertext()).split()).strip()

    def _parse_abstract(self, article: ET.Element) -> tuple[dict[str, str], str]:
        """
        Extract abstract sections and the combined abstract string.

        Returns
        -------
        tuple[dict[str, str], str]
            ``(abstract_sections, abstract_text)`` where ``abstract_sections``
            maps section labels to text and ``abstract_text`` is the full
            abstract as a single string (sections joined by newlines).
        """
        abstract_elem = article.find("Abstract")
        if abstract_elem is None:
            return {}, ""

        sections: dict[str, str] = {}
        parts: list[str] = []

        for elem in abstract_elem.findall("AbstractText"):
            label = (elem.get("Label") or "").strip().upper()
            text  = self._text(elem)
            if not text:
                continue
            if label:
                sections[label] = text
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        return sections, "\n".join(parts)

    def _parse_authors(self, article: ET.Element) -> list[str]:
        """Return a list of author last names in publication order."""
        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            last = self._text(author.find("LastName"))
            if last:
                authors.append(last)
        return authors

    def _parse_journal(self, root: ET.Element) -> str:
        """Return the journal abbreviation, falling back to the full title."""
        medline_ta = root.find(".//MedlineCitation/MedlineJournalInfo/MedlineTA")
        if medline_ta is not None:
            return self._text(medline_ta)
        title_elem = root.find(".//Article/Journal/Title")
        return self._text(title_elem)

    def _parse_pub_year(self, article: ET.Element, root: ET.Element) -> str:
        """
        Extract the four-digit publication year.

        Attempts, in order:
          1. ``<PubDate><Year>`` (electronic publication date)
          2. ``<PubDate><MedlineDate>`` (free-text fallback, first 4 digits)
          3. ``<ArticleDate><Year>`` (article-level date)
        """
        year_elem = article.find(".//Journal/JournalIssue/PubDate/Year")
        if year_elem is not None:
            return self._text(year_elem)

        medline_date = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date is not None:
            match = re.search(r"\b(19|20)\d{2}\b", self._text(medline_date))
            if match:
                return match.group(0)

        article_date = article.find(".//ArticleDate/Year")
        return self._text(article_date) if article_date is not None else ""

    def _parse_mesh(self, root: ET.Element) -> list[str]:
        """Return a list of MeSH descriptor names for this article."""
        terms: list[str] = []
        for heading in root.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            name = self._text(heading)
            if name:
                terms.append(name)
        return terms
