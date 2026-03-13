"""
PubMed fetch module — retrieves abstracts via NCBI Entrez API (Biopython).
"""
import logging
import time
from typing import Optional
from Bio import Entrez

logger = logging.getLogger(__name__)


def fetch_abstracts(
    query: str,
    email: str,
    max_results: int = 50,
    api_key: Optional[str] = None,
) -> list:
    """
    Search PubMed and fetch abstracts for a given query string.

    Parameters
    ----------
    query : str
        Free-text PubMed query.
    email : str
        Required by NCBI to identify the caller.
    max_results : int
        Maximum number of papers to retrieve.
    api_key : str, optional
        NCBI API key for higher rate limits.

    Returns
    -------
    list of dicts: pmid, title, abstract, authors, journal, year, keywords, url
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    logger.info(f"Searching PubMed: '{query}' (max={max_results})")

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]

    if not pmids:
        logger.warning("No results found for query.")
        return []

    logger.info(f"Found {len(pmids)} papers. Fetching abstracts...")

    papers = []
    batch_size = 20
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records["PubmedArticle"]:
            paper = _parse_article(article)
            if paper:
                papers.append(paper)
        time.sleep(0.4)

    logger.info(f"Successfully fetched {len(papers)} abstracts.")
    return papers


def _parse_article(article: dict) -> Optional[dict]:
    """Parse a single PubMed XML article record into a flat dict."""
    try:
        medline = article["MedlineCitation"]
        art     = medline["Article"]

        pmid    = str(medline["PMID"])
        title   = str(art.get("ArticleTitle", ""))
        journal = str(art["Journal"]["Title"]) if "Journal" in art else ""

        abstract_obj   = art.get("Abstract", {})
        abstract_texts = abstract_obj.get("AbstractText", [])
        if isinstance(abstract_texts, list):
            abstract = " ".join(str(t) for t in abstract_texts)
        else:
            abstract = str(abstract_texts)

        author_list = art.get("AuthorList", [])
        authors = []
        for a in author_list:
            last  = a.get("LastName", "")
            first = a.get("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())
        authors_str = ", ".join(authors[:6])
        if len(authors) > 6:
            authors_str += " et al."

        pub_date = art["Journal"]["JournalIssue"].get("PubDate", {})
        year     = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

        mesh_list = medline.get("MeshHeadingList", [])
        keywords  = [str(m["DescriptorName"]) for m in mesh_list]

        return {
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "authors":  authors_str,
            "journal":  journal,
            "year":     year,
            "keywords": keywords,
            "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except Exception as e:
        logger.warning(f"Could not parse article: {e}")
        return None
