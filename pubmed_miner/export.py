"""
Excel export — writes a multi-sheet report from pubmed-miner results.

Sheet 1: Ranked Papers      — all papers sorted by relevance score
Sheet 2: Topic Clusters     — cluster summaries with top terms
Sheet 3: Top Entities       — most frequent genes, diseases, drugs across corpus
Sheet 4: Entity Detail      — per-paper entity breakdown
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def export_excel(
    query: str,
    papers: list[dict],
    cluster_summary: list[dict],
    top_entities: dict,
    output_dir: str = "results",
) -> str:
    """
    Export full results to a multi-sheet Excel workbook.

    Parameters
    ----------
    query : str
    papers : list of paper dicts with relevance_score, entities, cluster_label
    cluster_summary : from clustering.cluster_papers()
    top_entities : from entities.extract_entities()
    output_dir : directory to write the file

    Returns
    -------
    str : path to the written Excel file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "_".join(query.split()[:5]).replace("/", "-")
    filename = f"pubmed_miner_{safe_query}_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        _write_papers_sheet(papers, writer)
        _write_clusters_sheet(cluster_summary, writer)
        _write_entities_sheet(top_entities, writer)
        _write_entity_detail_sheet(papers, writer)
        _write_query_sheet(query, papers, writer)

    logger.info(f"Excel report written to: {filepath}")
    return filepath


def _write_papers_sheet(papers, writer):
    rows = []
    for p in papers:
        rows.append({
            "Rank":            papers.index(p) + 1,
            "PMID":            p["pmid"],
            "Title":           p["title"],
            "Authors":         p["authors"],
            "Journal":         p["journal"],
            "Year":            p["year"],
            "Relevance Score": p.get("relevance_score", ""),
            "Cluster":         p.get("cluster_label", ""),
            "Genes":           ", ".join(p.get("entities", {}).get("genes", [])),
            "Diseases":        ", ".join(p.get("entities", {}).get("diseases", [])),
            "Drugs":           ", ".join(p.get("entities", {}).get("drugs", [])),
            "Abstract":        p["abstract"],
            "URL":             p["url"],
        })
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Ranked Papers", index=False)
    _autofit(writer, "Ranked Papers", df)


def _write_clusters_sheet(cluster_summary, writer):
    rows = []
    for c in cluster_summary:
        rows.append({
            "Cluster ID":             c["cluster_id"],
            "Label":                  c["label"],
            "Paper Count":            c["paper_count"],
            "Top Terms":              ", ".join(c["top_terms"]),
            "Representative Title":   c["representative_title"],
            "Representative PMID":    c["representative_pmid"],
        })
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Topic Clusters", index=False)
    _autofit(writer, "Topic Clusters", df)


def _write_entities_sheet(top_entities, writer):
    max_len = max(len(v) for v in top_entities.values()) if top_entities else 0
    data = {}
    for category, items in top_entities.items():
        terms  = [i[0] for i in items] + [""] * (max_len - len(items))
        counts = [i[1] for i in items] + [""] * (max_len - len(items))
        data[f"{category.title()} Term"]  = terms
        data[f"{category.title()} Count"] = counts
    df = pd.DataFrame(data)
    df.to_excel(writer, sheet_name="Top Entities", index=False)
    _autofit(writer, "Top Entities", df)


def _write_entity_detail_sheet(papers, writer):
    rows = []
    for p in papers:
        ents = p.get("entities", {})
        rows.append({
            "PMID":         p["pmid"],
            "Title":        p["title"],
            "Genes":        ", ".join(ents.get("genes", [])),
            "Diseases":     ", ".join(ents.get("diseases", [])),
            "Drugs":        ", ".join(ents.get("drugs", [])),
            "Viral/Immune": ", ".join(ents.get("viral_immune", [])),
            "MeSH Keywords": ", ".join(p.get("keywords", [])),
        })
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Entity Detail", index=False)
    _autofit(writer, "Entity Detail", df)


def _write_query_sheet(query, papers, writer):
    """Summary sheet — query metadata for reproducibility."""
    rows = [
        {"Field": "Query",           "Value": query},
        {"Field": "Papers Retrieved","Value": len(papers)},
        {"Field": "Date Run",        "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Tool",            "Value": "pubmed-miner v0.1.0"},
        {"Field": "Source",          "Value": "NCBI PubMed E-utilities API"},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Query Info", index=False)


def _autofit(writer, sheet_name, df):
    """Set reasonable column widths."""
    ws = writer.sheets[sheet_name]
    for col_idx, col in enumerate(df.columns, 1):
        max_len = max(
            len(str(col)),
            df[col].astype(str).str.len().max() if not df.empty else 0,
        )
        ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 60)
