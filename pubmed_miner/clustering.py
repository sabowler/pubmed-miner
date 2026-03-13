"""
Topic clustering — groups papers into thematic clusters using TF-IDF + KMeans.

Each cluster is labelled by its top TF-IDF terms, giving a human-readable
summary of what each topic group covers. Useful for spotting sub-themes
within a broad literature search.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


def cluster_papers(papers: list[dict], n_clusters: int = None) -> tuple[list[dict], list[dict]]:
    """
    Cluster papers into topic groups.

    Automatically selects number of clusters if not specified:
      - <= 10 papers  → 2 clusters
      - 11–30 papers  → 3 clusters
      - 31–50 papers  → 5 clusters
      - > 50 papers   → 7 clusters

    Parameters
    ----------
    papers : list of paper dicts (must have 'title' and 'abstract')
    n_clusters : int, optional — override automatic cluster count

    Returns
    -------
    papers : same list with 'cluster_id' and 'cluster_label' added
    cluster_summary : list of dicts describing each cluster:
        {cluster_id, label, top_terms, paper_count, representative_title}
    """
    if len(papers) < 4:
        logger.warning("Too few papers to cluster meaningfully. Assigning all to cluster 0.")
        for p in papers:
            p["cluster_id"]    = 0
            p["cluster_label"] = "All Papers"
        return papers, [{"cluster_id": 0, "label": "All Papers", "top_terms": [], "paper_count": len(papers)}]

    if n_clusters is None:
        n = len(papers)
        if n <= 10:   n_clusters = 2
        elif n <= 30: n_clusters = 3
        elif n <= 50: n_clusters = 5
        else:         n_clusters = 7

    corpus = [f"{p['title']} {p['abstract']}" for p in papers]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(corpus)

    # Dimensionality reduction for better clustering
    n_components = min(50, X.shape[1] - 1, len(papers) - 1)
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X)
    else:
        X_reduced = X.toarray()

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_reduced)

    # Get top terms per cluster from original TF-IDF space
    feature_names = vectorizer.get_feature_names_out()
    cluster_top_terms = _get_cluster_terms(km, svd if n_components > 1 else None, feature_names, X, n_clusters)

    # Assign cluster info to papers
    for paper, cluster_id in zip(papers, labels):
        paper["cluster_id"]    = int(cluster_id)
        paper["cluster_label"] = _make_label(cluster_top_terms[int(cluster_id)])

    # Build cluster summary
    cluster_summary = []
    for cid in range(n_clusters):
        cluster_papers = [p for p in papers if p["cluster_id"] == cid]
        if not cluster_papers:
            continue
        # Representative title = highest relevance score in cluster
        rep = max(cluster_papers, key=lambda p: p.get("relevance_score", 0))
        cluster_summary.append({
            "cluster_id":           cid,
            "label":                _make_label(cluster_top_terms[cid]),
            "top_terms":            cluster_top_terms[cid],
            "paper_count":          len(cluster_papers),
            "representative_title": rep["title"],
            "representative_pmid":  rep["pmid"],
        })

    cluster_summary.sort(key=lambda c: c["paper_count"], reverse=True)
    logger.info(f"Clustered {len(papers)} papers into {n_clusters} topic groups.")
    return papers, cluster_summary


def _get_cluster_terms(km, svd, feature_names, X, n_clusters, top_n=6):
    """Extract top TF-IDF terms for each cluster centroid."""
    terms_per_cluster = {}
    if svd is not None:
        # Project centroids back to original feature space
        centroids = km.cluster_centers_.dot(svd.components_)
    else:
        centroids = km.cluster_centers_

    for cid in range(n_clusters):
        if cid < len(centroids):
            top_indices = np.argsort(centroids[cid])[::-1][:top_n]
            terms_per_cluster[cid] = [feature_names[i] for i in top_indices]
        else:
            terms_per_cluster[cid] = []

    return terms_per_cluster


def _make_label(terms: list[str]) -> str:
    """Create a short human-readable cluster label from top terms."""
    if not terms:
        return "Uncategorized"
    # Use first 3 terms, title-cased
    return " / ".join(t.title() for t in terms[:3])
