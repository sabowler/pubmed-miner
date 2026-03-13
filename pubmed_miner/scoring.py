"""
Relevance scoring — ranks papers by semantic similarity to the input query.

Uses TF-IDF cosine similarity, which is:
  - Fast and dependency-light (scikit-learn only)
  - Robust for keyword-rich biomedical queries
  - Interpretable (score reflects term overlap with query)

For semantic/embedding-based scoring, swap in a SentenceTransformer
(e.g. pritamdeka/S-PubMedBert-MS-MARCO) if GPU is available.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def score_relevance(query: str, papers: list[dict]) -> list[dict]:
    """
    Score and rank papers by relevance to the query using TF-IDF cosine similarity.

    Parameters
    ----------
    query : str
        The original search query.
    papers : list of dicts
        Papers from fetch_abstracts(), each with 'title' and 'abstract'.

    Returns
    -------
    papers : same list with 'relevance_score' added, sorted descending by score.
    """
    if not papers:
        return papers

    # Build corpus: title + abstract for each paper
    corpus = [f"{p['title']} {p['abstract']}" for p in papers]

    # Fit TF-IDF on corpus + query together so query terms are in vocabulary
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # Unigrams + bigrams for biomedical phrases
        max_features=10000,
        sublinear_tf=True,    # Log normalization reduces impact of high-freq terms
    )
    all_texts  = [query] + corpus
    tfidf      = vectorizer.fit_transform(all_texts)
    query_vec  = tfidf[0]
    corpus_vec = tfidf[1:]

    scores = cosine_similarity(query_vec, corpus_vec).flatten()

    for paper, score in zip(papers, scores):
        paper["relevance_score"] = round(float(score), 4)

    # Sort by relevance descending
    papers = sorted(papers, key=lambda p: p["relevance_score"], reverse=True)

    logger.info(
        f"Relevance scoring complete. "
        f"Top score={papers[0]['relevance_score']:.4f}, "
        f"Mean={np.mean(scores):.4f}"
    )
    return papers
