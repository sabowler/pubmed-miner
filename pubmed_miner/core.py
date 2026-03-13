"""
Core orchestration — ties fetch, entity extraction, scoring, clustering,
and export into a single search() call.
"""

import logging
import os
from typing import Optional

from pubmed_miner.fetch import fetch_abstracts
from pubmed_miner.entities import extract_entities
from pubmed_miner.scoring import score_relevance
from pubmed_miner.clustering import cluster_papers
from pubmed_miner.export import export_excel

logger = logging.getLogger(__name__)


def search(
    query: str,
    email: str,
    max_results: int = 50,
    n_clusters: Optional[int] = None,
    output_dir: str = "results",
    api_key: Optional[str] = None,
    export: bool = True,
) -> dict:
    """
    Run a full PubMed literature mining pipeline for a given query.

    Parameters
    ----------
    query : str
        Free-text search query. Supports full PubMed syntax, e.g.:
          "HERV-K immunotherapy melanoma"
          "DNA methylation COVID-19 severity"
          "opioid use disorder single cell RNA sequencing"
    email : str
        Your email address — required by NCBI to identify API callers.
    max_results : int
        Number of abstracts to retrieve. Default 50.
    n_clusters : int, optional
        Number of topic clusters. Auto-selected if not specified.
    output_dir : str
        Directory to write the Excel report. Default "results".
    api_key : str, optional
        NCBI API key for higher rate limits (10 req/s vs 3 req/s).
        Get one free at https://www.ncbi.nlm.nih.gov/account/
    export : bool
        Whether to write the Excel report. Default True.

    Returns
    -------
    dict with keys:
        query          : original query string
        papers         : list of paper dicts, sorted by relevance, each containing:
                           pmid, title, abstract, authors, journal, year,
                           keywords, url, relevance_score, cluster_id,
                           cluster_label, entities
        clusters       : list of cluster summary dicts
        top_entities   : dict of most frequent genes, diseases, drugs, viral_immune
        excel_path     : path to Excel report (None if export=False)
        n_retrieved    : number of papers successfully fetched

    Example
    -------
    >>> import pubmed_miner
    >>> results = pubmed_miner.search(
    ...     query="HERV-K expression PD-1 immunotherapy response",
    ...     email="sbowler49601@gmail.com",
    ...     max_results=50,
    ...     output_dir="grant_review"
    ... )
    >>> print(f"Retrieved {results['n_retrieved']} papers")
    >>> print(f"Top paper: {results['papers'][0]['title']}")
    >>> print(f"Report saved to: {results['excel_path']}")
    """
    _setup_logging()
    logger.info(f"=== pubmed-miner | Query: '{query}' ===")

    # 1. Fetch abstracts from PubMed
    papers = fetch_abstracts(query, email=email, max_results=max_results, api_key=api_key)
    if not papers:
        logger.warning("No papers retrieved. Check your query or internet connection.")
        return {"query": query, "papers": [], "clusters": [], "top_entities": {}, "excel_path": None, "n_retrieved": 0}

    # 2. Score relevance
    papers = score_relevance(query, papers)

    # 3. Extract biomedical entities
    papers, top_entities = extract_entities(papers)

    # 4. Cluster by topic
    papers, cluster_summary = cluster_papers(papers, n_clusters=n_clusters)

    # 5. Export to Excel
    excel_path = None
    if export:
        excel_path = export_excel(query, papers, cluster_summary, top_entities, output_dir)

    logger.info(
        f"=== Complete | {len(papers)} papers | "
        f"{len(cluster_summary)} clusters | "
        f"Report: {excel_path} ==="
    )

    return {
        "query":        query,
        "papers":       papers,
        "clusters":     cluster_summary,
        "top_entities": top_entities,
        "excel_path":   excel_path,
        "n_retrieved":  len(papers),
    }


def _setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
