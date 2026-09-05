"""
PubMed / NCBI E-utilities client.

Provides a thin, typed interface over NCBI EFetch. It retrieves one or more
PMIDs supplied by ClinicalTrials.gov and parses them into ``PubMedRecord``
instances.

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
graceful degradation should catch these explicitly
rather than catching bare ``Exception``.

Usage
-----
    from src.pipeline.pubmed_client import PubMedClient

    client = PubMedClient()
    records = client.fetch_records_batch(["26030518", "26947331"])
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
)

logger = logging.getLogger(__name__)

# Transient-failure retry (DNS/connection drops, NCBI 429/5xx).
_MAX_RETRIES = 4
_RETRY_BACKOFF_S = 2.0


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
    results_text:
        Text from structured abstract sections that report trial results, if
        PubMed exposes labelled sections.
    conclusion_text:
        Text from structured abstract conclusion / interpretation sections, if
        available.
    authors:
        List of author last names in publication order.
    pub_year:
        Four-digit publication year as a string, or ``""`` if absent.
    pub_date:
        ISO publication date when PubMed supplies year, month, and day.
    journal:
        Journal name (MedlineTA abbreviation preferred, full title fallback).
    mesh_terms:
        MeSH descriptor names, useful for endpoint domain classification.
    pub_types:
        Set of PubMed ``PublicationType`` strings for this article, e.g.
        ``{"Randomized Controlled Trial", "Multicenter Study"}``.
        Used by the article-type gate (``article_classifier.py``) to detect
        protocol papers, systematic reviews, and editorials before they enter
        the endpoint matching pipeline.
    doi:
        Digital Object Identifier when supplied in the PubMed record.
    nct_ids:
        ClinicalTrials.gov identifiers found in PubMed secondary identifiers,
        the title, or the abstract.
    """

    pmid: str
    title: str
    abstract_text: str
    abstract_sections: dict[str, str] = field(default_factory=dict)
    results_text: str = ""
    conclusion_text: str = ""
    authors: list[str] = field(default_factory=list)
    pub_year: str = ""
    pub_date: str = ""
    journal: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    pub_types: set[str] = field(default_factory=set)
    doi: str = ""
    nct_ids: set[str] = field(default_factory=set)


_RESULT_SECTION_LABELS: tuple[str, ...] = (
    "RESULTS",
    "RESULT",
    "FINDINGS",
    "MAIN RESULTS",
    "OUTCOMES",
    "MAIN OUTCOME MEASURE",
    "MAIN OUTCOME MEASURES",
    "RESULTS AND DISCUSSION",
)

_CONCLUSION_SECTION_LABELS: tuple[str, ...] = (
    "CONCLUSION",
    "CONCLUSIONS",
    "CONCLUSIONS AND RELEVANCE",
    "INTERPRETATION",
    "INTERPRETATIONS",
    "DISCUSSION",
)


# ---------------------------------------------------------------------------
# PubMed API client
# ---------------------------------------------------------------------------


class PubMedClient:
    """
    Stateful NCBI E-utilities client with integrated rate limiting.

    A single instance is intended to be shared across a publication-selection
    run so that the rate-limit sleep is applied consistently.

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
        self._api_key = api_key
        self._rate_limit_s = rate_limit_s
        self._last_call_ts: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SAP-Coherence-Checker/3.0 (research)"})
        # PMID -> parsed record. A PMID reached through several trials' citation
        # chains is fetched and parsed once.
        self._record_cache: dict[str, PubMedRecord] = {}

    # ------------------------------------------------------------------
    # Candidate discovery
    # ------------------------------------------------------------------

    def search(self, term: str, max_results: int = 50) -> list[str]:
        """Return PubMed IDs matching an Entrez query."""
        if not term.strip():
            return []
        url = f"{PUBMED_BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        data = self._get(url, params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def search_by_trial_id(self, nct_id: str, max_results: int = 100) -> list[str]:
        """Find every PubMed record that indexes or mentions an exact NCT ID."""
        nct_id = nct_id.strip().upper()
        return self.search(f'({nct_id}[si] OR "{nct_id}"[tiab])', max_results=max_results)

    def citation_neighbors(self, pmids: list[str], max_results: int = 100) -> list[str]:
        """Return PubMed references and citing papers for confirmed seed PMIDs."""
        unique = list(dict.fromkeys(str(pmid).strip() for pmid in pmids if str(pmid).strip()))
        if not unique:
            return []

        neighbors: list[str] = []
        for link_name in ("pubmed_pubmed_refs", "pubmed_pubmed_citedin"):
            url = f"{PUBMED_BASE_URL}/elink.fcgi"
            params: list[tuple[str, str]] = [
                ("dbfrom", "pubmed"),
                ("db", "pubmed"),
                ("linkname", link_name),
                ("retmode", "xml"),
            ]
            params.extend(("id", pmid) for pmid in unique)
            if self._api_key:
                params.append(("api_key", self._api_key))
            root = ET.fromstring(self._get(url, params).text)
            for element in root.findall(".//LinkSetDb/Link/Id"):
                value = (element.text or "").strip()
                if value and value not in unique and value not in neighbors:
                    neighbors.append(value)
                    if len(neighbors) >= max_results:
                        return neighbors
        return neighbors

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
        key = str(pmid).strip()
        if key in self._record_cache:
            return self._record_cache[key]

        url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "xml",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        response = self._get(url, params)
        root = ET.fromstring(response.text)
        record = self._parse_article(root, pmid)
        self._record_cache[key] = record
        logger.debug("fetch_record(%s) → title=%r", pmid, record.title[:60])
        return record

    def fetch_records_batch(self, pmids: list[str]) -> dict[str, "PubMedRecord"]:
        """
        Retrieve multiple PubMed records in a single efetch call.

        NCBI efetch accepts a comma-separated ``id`` list, so this replaces
        N sequential ``fetch_record`` calls with one API round-trip.  Rate
        limiting still applies (one sleep before the single request).

        Parameters
        ----------
        pmids:
            List of PubMed identifiers to fetch.  Duplicates are de-duplicated
            before the request is issued.

        Returns
        -------
        dict[str, PubMedRecord]
            Mapping of PMID → PubMedRecord for every record successfully
            parsed from the response.  PMIDs absent from the PubMed response
            (e.g. retracted records) are silently omitted.
        """
        unique = list(dict.fromkeys(str(p).strip() for p in pmids if str(p).strip()))
        if not unique:
            return {}

        records: dict[str, PubMedRecord] = {p: self._record_cache[p] for p in unique if p in self._record_cache}
        to_fetch = [p for p in unique if p not in records]
        if not to_fetch:
            return records

        url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        params: dict = {
            "db": "pubmed",
            "id": ",".join(to_fetch),
            "rettype": "xml",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        response = self._get(url, params)
        root = ET.fromstring(response.text)

        for article_elem in root.findall(".//PubmedArticle"):
            # Prefer Version="1" to avoid duplicate PMID entries for erratum notices.
            pmid_elem = article_elem.find(".//PMID[@Version='1']")
            if pmid_elem is None:
                pmid_elem = article_elem.find(".//PMID")
            if pmid_elem is None or not (pmid_elem.text or "").strip():
                continue
            p = pmid_elem.text.strip()
            record = self._parse_article(article_elem, p)
            records[p] = record
            self._record_cache[p] = record

        logger.debug(
            "fetch_records_batch(%d pmids) → %d cached, %d fetched",
            len(unique),
            len(unique) - len(to_fetch),
            len(to_fetch),
        )
        return records

    # ------------------------------------------------------------------
    # Private helpers — NCBI HTTP layer
    # ------------------------------------------------------------------

    def _get(
        self,
        url: str,
        params: dict[str, object] | list[tuple[str, str]],
    ) -> requests.Response:
        """
        Issue a rate-limited GET request.

        Sleeps for ``self._rate_limit_s`` seconds since the last call before
        dispatching the request, ensuring the pipeline stays within NCBI's
        stated rate limits regardless of how quickly the caller invokes this
        method.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            elapsed = time.monotonic() - self._last_call_ts
            remaining = self._rate_limit_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            try:
                response = self._session.get(
                    url, params=params, timeout=PUBMED_REQUEST_TIMEOUT_S
                )
                self._last_call_ts = time.monotonic()
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{response.status_code} from NCBI", response=response)
                response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                self._last_call_ts = time.monotonic()
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = _RETRY_BACKOFF_S * (2**attempt)
                    logger.warning(
                        "NCBI request failed (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
        raise last_exc  # type: ignore[misc]

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
        article = root.find(".//MedlineCitation/Article")
        if article is None:
            article = root.find(".//Article")
        if article is None:
            logger.warning("fetch_record(%s): no <Article> element found in XML", pmid)
            return PubMedRecord(pmid=pmid, title="", abstract_text="")

        title = self._text(article.find("ArticleTitle"))
        authors = self._parse_authors(article)
        journal = self._parse_journal(root)
        pub_year = self._parse_pub_year(article, root)
        pub_date = self._parse_pub_date(article)
        abstract_sections, abstract_text = self._parse_abstract(article)
        results_text = self._section_text(abstract_sections, _RESULT_SECTION_LABELS)
        conclusion_text = self._section_text(abstract_sections, _CONCLUSION_SECTION_LABELS)
        mesh_terms = self._parse_mesh(root)
        pub_types = self._parse_pub_types(article)
        doi = self._parse_doi(root)
        nct_ids = self._parse_nct_ids(root, title, abstract_text)

        return PubMedRecord(
            pmid=pmid,
            title=title,
            abstract_text=abstract_text,
            abstract_sections=abstract_sections,
            results_text=results_text,
            conclusion_text=conclusion_text,
            authors=authors,
            pub_year=pub_year,
            pub_date=pub_date,
            journal=journal,
            mesh_terms=mesh_terms,
            pub_types=pub_types,
            doi=doi,
            nct_ids=nct_ids,
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
            text = self._text(elem)
            if not text:
                continue
            if label:
                sections[label] = " ".join(part for part in (sections.get(label, ""), text) if part)
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        return sections, "\n".join(parts)

    @staticmethod
    def _section_text(sections: dict[str, str], labels: tuple[str, ...]) -> str:
        """Return labelled abstract text for the first matching section group."""
        parts = [sections[label] for label in labels if sections.get(label)]
        return "\n".join(parts)

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

    def _parse_pub_date(self, article: ET.Element) -> str:
        """Return an exact ISO publication date, or blank when incomplete."""
        month_names = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        for date_element in (
            article.find(".//ArticleDate"),
            article.find(".//Journal/JournalIssue/PubDate"),
        ):
            if date_element is None:
                continue
            year = self._text(date_element.find("Year"))
            month = self._text(date_element.find("Month"))
            day = self._text(date_element.find("Day"))
            if not (year.isdigit() and day.isdigit() and month):
                continue
            month_number = int(month) if month.isdigit() else month_names.get(month[:3].lower())
            if month_number is None:
                continue
            try:
                return f"{int(year):04d}-{month_number:02d}-{int(day):02d}"
            except ValueError:
                continue
        return ""

    def _parse_mesh(self, root: ET.Element) -> list[str]:
        """Return a list of MeSH descriptor names for this article."""
        terms: list[str] = []
        for heading in root.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            name = self._text(heading)
            if name:
                terms.append(name)
        return terms

    def _parse_pub_types(self, article: ET.Element) -> set[str]:
        """
        Parse PubMed ``PublicationType`` tags into a set of strings.

        These tags are the most reliable signal for the article-type gate.
        Typical values for results papers: ``"Randomized Controlled Trial"``,
        ``"Clinical Trial, Phase III"``, ``"Multicenter Study"``.
        Typical values for non-results articles: ``"Study Protocol"``,
        ``"Meta-Analysis"``, ``"Editorial"``, ``"Letter"``.

        Parameters
        ----------
        article:
            The ``<Article>`` XML element from a parsed PubMed record.

        Returns
        -------
        set[str]
            Publication type strings, stripped of whitespace.  Empty set if
            the ``<PublicationTypeList>`` element is absent.
        """
        types: set[str] = set()
        for pt_elem in article.findall(".//PublicationTypeList/PublicationType"):
            pt = self._text(pt_elem)
            if pt:
                types.add(pt)
        return types

    def _parse_doi(self, root: ET.Element) -> str:
        """Return the DOI from PubMed article identifiers, when available."""
        for element in root.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if (element.get("IdType") or "").lower() == "doi":
                return self._text(element).lower()
        return ""

    def _parse_nct_ids(self, root: ET.Element, title: str, abstract: str) -> set[str]:
        """Collect NCT identifiers from indexed IDs and citation text."""
        text_parts = [title, abstract]
        text_parts.extend(
            self._text(element) for element in root.findall(".//DataBankList//AccessionNumber")
        )
        text_parts.extend(
            self._text(element) for element in root.findall(".//MedlineCitation/OtherID")
        )
        return {value.upper() for value in re.findall(r"\bNCT\d{8}\b", " ".join(text_parts), re.I)}
