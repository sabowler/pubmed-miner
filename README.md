# pubmed-miner

**Biomedical literature mining toolkit for grant hypothesis generation.**

Built and maintained by [Scott A. Bowler](https://github.com/sabowler) — Ndhlovu Lab, Weill Cornell Medicine.

Takes a free-text query string and returns ranked, clustered, entity-annotated PubMed abstracts — exported to a structured Excel report. Designed to accelerate literature reviews during grant hypothesis development.

---

## Quickstart

```python
import pubmed_miner

results = pubmed_miner.search(
    query="HERV-K expression immunotherapy response melanoma",
    email="your@email.com",
    max_results=50,
    output_dir="grant_review"
)

print(f"Retrieved {results['n_retrieved']} papers")
print(f"Top paper: {results['papers'][0]['title']}")
print(f"Top genes: {results['top_entities']['genes'][:5]}")
print(f"Report: {results['excel_path']}")
```

---

## What It Does

For a given query string, the pipeline:

1. **Fetches** the top 50 most relevant PubMed abstracts via the NCBI Entrez API
2. **Scores** each paper by relevance to the query using TF-IDF cosine similarity
3. **Extracts** biomedical entities — genes, diseases, drugs, viral/immune terms
4. **Clusters** papers into topic groups with human-readable labels
5. **Exports** a multi-sheet Excel report

---

## Output

### Results dict

```python
results["papers"]         # List of paper dicts, sorted by relevance score
results["clusters"]       # Topic cluster summaries with top terms
results["top_entities"]   # Most frequent genes, diseases, drugs across corpus
results["excel_path"]     # Path to the exported Excel report
results["n_retrieved"]    # Number of papers fetched
```

### Excel report (5 sheets)

| Sheet | Contents |
|---|---|
| Ranked Papers | All papers sorted by relevance, with entities and cluster label |
| Topic Clusters | Cluster summaries with top terms and representative titles |
| Top Entities | Most frequent genes, diseases, drugs across the corpus |
| Entity Detail | Per-paper entity breakdown |
| Query Info | Query string, date, paper count — for reproducibility |

---

## Installation

```bash
git clone https://github.com/sabowler/pubmed-miner.git
cd pubmed-miner
pip install -e .

# Development (includes pytest)
pip install -e ".[dev]"
```

---

## Parameters

```python
pubmed_miner.search(
    query       = "opioid use disorder neuroinflammation scRNA-seq",
    email       = "your@email.com",      # Required by NCBI
    max_results = 50,                    # Papers to retrieve (default 50)
    n_clusters  = None,                  # Auto-selected if None
    output_dir  = "results",             # Where to write the Excel report
    api_key     = None,                  # NCBI API key (optional, increases rate limit)
    export      = True,                  # Set False to skip Excel export
)
```

**NCBI API key** — free to register at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).
Increases rate limit from 3 to 10 requests/second.

---

## Example Queries

```python
# Immunotherapy biomarker discovery
pubmed_miner.search("PD-1 PD-L1 biomarker response solid tumors", email="...")

# HIV neurocognitive research
pubmed_miner.search("HIV associated neurocognitive disorder neuroinflammation", email="...")

# Opioid epigenomics (SCORCH)
pubmed_miner.search("opioid use disorder DNA methylation epigenomics", email="...")

# Cancer microbiome
pubmed_miner.search("tumor microbiome immunotherapy response colorectal cancer", email="...")
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 12 tests run without a network connection using mock paper data.

