"""
pubmed-miner — Biomedical literature mining toolkit.

Fetches PubMed abstracts, scores relevance, extracts biomedical entities,
clusters by topic, summarizes findings, and exports to Excel.

Example usage:
    import pubmed_miner

    results = pubmed_miner.search(
        query="HERV-K expression melanoma immunotherapy PD-1",
        max_results=50,
    )

    print(results.summary)
    results.to_excel("grant_review.xlsx")
"""

from pubmed_miner.core import search

__version__ = "0.1.0"
__author__ = "Scott A. Bowler"
__all__ = ["search"]
