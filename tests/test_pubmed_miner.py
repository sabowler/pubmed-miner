"""Tests for pubmed-miner — no network calls required."""

import pytest
from pubmed_miner.scoring import score_relevance
from pubmed_miner.clustering import cluster_papers
from pubmed_miner.entities import extract_entities


MOCK_PAPERS = [
    {
        "pmid": "1001",
        "title": "HERV-K expression predicts immunotherapy response in melanoma",
        "abstract": "We investigated HERV-K endogenous retrovirus expression in melanoma patients treated with PD-1 checkpoint inhibitors pembrolizumab and nivolumab. BRCA1 and TP53 mutations were also assessed.",
        "authors": "Bowler SA, Ndhlovu LC",
        "journal": "Cancer Research",
        "year": "2024",
        "keywords": ["Melanoma", "Immunotherapy", "Endogenous Retroviruses"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/1001/",
    },
    {
        "pmid": "1002",
        "title": "DNA methylation patterns in COVID-19 severity classification",
        "abstract": "Machine learning classifiers using CpG methylation features predicted COVID-19 disease severity with AUC greater than 0.90. EGFR and KRAS methylation were among top features.",
        "authors": "Bowler SA et al.",
        "journal": "Scientific Reports",
        "year": "2022",
        "keywords": ["COVID-19", "DNA Methylation", "Machine Learning"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/1002/",
    },
    {
        "pmid": "1003",
        "title": "Single-cell RNA sequencing of opioid-exposed neurons",
        "abstract": "scRNA-seq analysis of opioid-exposed neuronal populations revealed differential expression of HERV elements and immune checkpoint genes including PD-L1 and CTLA4.",
        "authors": "Smith J et al.",
        "journal": "Nature Neuroscience",
        "year": "2023",
        "keywords": ["Opioids", "Single-Cell Analysis"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/1003/",
    },
    {
        "pmid": "1004",
        "title": "Tumor mutational burden predicts response to anti-CTLA4 therapy",
        "abstract": "High TMB was associated with improved response to ipilimumab in NSCLC patients. TP53 and KRAS mutations correlated with TMB scores.",
        "authors": "Jones K et al.",
        "journal": "Journal of Clinical Oncology",
        "year": "2023",
        "keywords": ["Lung Cancer", "Immunotherapy", "CTLA-4 Antigen"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/1004/",
    },
    {
        "pmid": "1005",
        "title": "MSI status and immune infiltration in colorectal cancer",
        "abstract": "Microsatellite instability high colorectal cancers showed increased CD8 T cell infiltration and response to PD-1 blockade with pembrolizumab.",
        "authors": "Lee M et al.",
        "journal": "Gut",
        "year": "2024",
        "keywords": ["Colorectal Cancer", "Microsatellite Instability"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/1005/",
    },
]


# --- Scoring tests -----------------------------------------------------------

def test_score_relevance_returns_scores():
    query = "HERV-K immunotherapy melanoma"
    papers = [p.copy() for p in MOCK_PAPERS]
    scored = score_relevance(query, papers)
    for p in scored:
        assert "relevance_score" in p
        assert 0.0 <= p["relevance_score"] <= 1.0


def test_score_relevance_sorted_descending():
    query = "HERV-K immunotherapy melanoma"
    papers = [p.copy() for p in MOCK_PAPERS]
    scored = score_relevance(query, papers)
    scores = [p["relevance_score"] for p in scored]
    assert scores == sorted(scores, reverse=True)


def test_most_relevant_paper_matches_query():
    query = "HERV-K immunotherapy melanoma"
    papers = [p.copy() for p in MOCK_PAPERS]
    scored = score_relevance(query, papers)
    # The HERV-K melanoma paper should rank first
    assert scored[0]["pmid"] == "1001"


def test_score_relevance_empty():
    result = score_relevance("some query", [])
    assert result == []


# --- Clustering tests --------------------------------------------------------

def test_cluster_assigns_all_papers():
    papers = [p.copy() for p in MOCK_PAPERS]
    papers = score_relevance("immunotherapy cancer", papers)
    clustered, summary = cluster_papers(papers)
    assert all("cluster_id" in p for p in clustered)
    assert all("cluster_label" in p for p in clustered)


def test_cluster_summary_has_all_clusters():
    papers = [p.copy() for p in MOCK_PAPERS]
    papers = score_relevance("immunotherapy cancer", papers)
    clustered, summary = cluster_papers(papers, n_clusters=2)
    cluster_ids_in_papers = {p["cluster_id"] for p in clustered}
    cluster_ids_in_summary = {c["cluster_id"] for c in summary}
    assert cluster_ids_in_papers == cluster_ids_in_summary


def test_cluster_paper_counts_sum_correctly():
    papers = [p.copy() for p in MOCK_PAPERS]
    papers = score_relevance("immunotherapy", papers)
    clustered, summary = cluster_papers(papers, n_clusters=2)
    assert sum(c["paper_count"] for c in summary) == len(papers)


# --- Entity extraction tests -------------------------------------------------

def test_extract_entities_adds_entities_key():
    papers = [p.copy() for p in MOCK_PAPERS]
    papers, top = extract_entities(papers)
    for p in papers:
        assert "entities" in p
        assert "genes" in p["entities"]
        assert "diseases" in p["entities"]


def test_extracts_known_gene():
    papers = [MOCK_PAPERS[0].copy()]  # Contains BRCA1, TP53
    papers, top = extract_entities(papers)
    genes = papers[0]["entities"]["genes"]
    assert any(g in {"BRCA1", "TP53"} for g in genes)


def test_extracts_known_disease():
    papers = [MOCK_PAPERS[0].copy()]  # Contains melanoma
    papers, top = extract_entities(papers)
    diseases = papers[0]["entities"]["diseases"]
    assert "melanoma" in diseases


def test_extracts_known_drug():
    papers = [MOCK_PAPERS[0].copy()]  # Contains pembrolizumab
    papers, top = extract_entities(papers)
    drugs = papers[0]["entities"]["drugs"]
    assert "pembrolizumab" in drugs


def test_top_entities_structure():
    papers = [p.copy() for p in MOCK_PAPERS]
    _, top = extract_entities(papers)
    assert "genes" in top
    assert "diseases" in top
    assert "drugs" in top
    assert isinstance(top["genes"], list)
