"""
Biomedical entity extraction.

Uses a hybrid approach:
  1. Curated gene/protein symbol regex (human HGNC-style patterns)
  2. MeSH keyword passthrough from PubMed records
  3. Dictionary-based disease and drug term matching

This approach requires no GPU or large model downloads, making it
portable across HPC and local environments. For higher-precision NER,
swap in a HuggingFace biomedical NER model (e.g. d4data/biomedical-ner-all).
"""

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated term lists (extend as needed for your research domain)
# ---------------------------------------------------------------------------

IMMUNE_CHECKPOINT_DRUGS = {
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "avelumab",
    "cemiplimab", "ipilimumab", "tremelimumab", "relatlimab",
    "pd-1", "pd-l1", "ctla-4", "ctla4", "lag-3", "tim-3", "tigit",
}

COMMON_CANCER_TERMS = {
    "melanoma", "nsclc", "lung cancer", "breast cancer", "colorectal cancer",
    "glioblastoma", "leukemia", "lymphoma", "hepatocellular carcinoma",
    "pancreatic cancer", "ovarian cancer", "prostate cancer", "bladder cancer",
    "renal cell carcinoma", "head and neck cancer", "cervical cancer",
    "endometrial cancer", "esophageal cancer", "gastric cancer",
    "non-small cell lung cancer", "small cell lung cancer",
    "diffuse large b-cell lymphoma", "multiple myeloma",
}

VIRAL_IMMUNE_TERMS = {
    "herv", "herv-k", "endogenous retrovirus", "hiv", "sars-cov-2", "covid-19",
    "ebv", "cmv", "hpv", "htlv", "hepatitis b", "hepatitis c",
}

COMMON_BIOMARKER_GENES = {
    "tp53", "brca1", "brca2", "egfr", "kras", "braf", "pik3ca", "pten",
    "akt1", "mtor", "myc", "vegf", "vegfa", "her2", "erbb2", "met",
    "alk", "ros1", "ret", "ntrk", "fgfr1", "fgfr2", "fgfr3",
    "cdkn2a", "rb1", "apc", "vhl", "nf1", "nf2", "stk11",
    "dnmt3a", "tet2", "asxl1", "jak2", "idh1", "idh2", "npm1",
    "flt3", "kit", "pdgfra", "pdgfrb", "cdk4", "cdk6",
    "pd1", "pdcd1", "cd274", "ctla4", "foxp3", "cd8a", "cd4",
    "ifng", "tnf", "il6", "il10", "il2", "tgfb1",
}

# Gene symbol pattern: 2-10 uppercase letters optionally followed by digits
GENE_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

# Remove common false positives (English words that match gene pattern)
GENE_FALSE_POSITIVES = {
    "CI", "HR", "OR", "SD", "SE", "BMI", "DNA", "RNA", "PCR", "MRI", "CT",
    "PET", "OS", "PFS", "AUC", "ROC", "IC", "WT", "KO", "IHC", "FISH",
    "USA", "FDA", "WHO", "NCI", "NIH", "US", "UK", "EU", "AND", "THE",
    "FOR", "WITH", "FROM", "WERE", "HAVE", "BEEN", "THIS", "THAT", "THEY",
    "THEIR", "WHICH", "THESE", "THOSE", "THAN", "INTO", "SOME", "BOTH",
    "ALSO", "SUCH", "EACH", "ONLY", "WHEN", "AFTER", "BEFORE", "MOST",
    "MORE", "LESS", "HIGH", "LOW", "NEW", "TWO", "THREE", "ONE",
}


def extract_entities(papers: list[dict]) -> list[dict]:
    """
    Extract biomedical entities from a list of paper dicts.

    Adds an 'entities' key to each paper containing:
        genes, diseases, drugs, viral_immune

    Also returns corpus-level entity frequency counts.

    Parameters
    ----------
    papers : list of dicts from fetch.fetch_abstracts()

    Returns
    -------
    papers : same list with 'entities' added to each paper
    """
    all_genes    = Counter()
    all_diseases = Counter()
    all_drugs    = Counter()
    all_viral    = Counter()

    for paper in papers:
        text = f"{paper['title']} {paper['abstract']}".lower()
        text_upper = f"{paper['title']} {paper['abstract']}"

        entities = {
            "genes":    _extract_genes(text, text_upper),
            "diseases": _extract_terms(text, COMMON_CANCER_TERMS),
            "drugs":    _extract_terms(text, IMMUNE_CHECKPOINT_DRUGS),
            "viral_immune": _extract_terms(text, VIRAL_IMMUNE_TERMS),
        }

        # Add MeSH keywords as supplementary disease terms
        for kw in paper.get("keywords", []):
            kw_lower = kw.lower()
            if any(d in kw_lower for d in ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "melanoma"]):
                entities["diseases"].add(kw)

        paper["entities"] = {k: sorted(v) for k, v in entities.items()}

        all_genes.update(entities["genes"])
        all_diseases.update(entities["diseases"])
        all_drugs.update(entities["drugs"])
        all_viral.update(entities["viral_immune"])

    logger.info(
        f"Entity extraction complete. "
        f"Unique genes={len(all_genes)}, diseases={len(all_diseases)}, "
        f"drugs={len(all_drugs)}"
    )

    return papers, {
        "genes":       all_genes.most_common(20),
        "diseases":    all_diseases.most_common(20),
        "drugs":       all_drugs.most_common(20),
        "viral_immune": all_viral.most_common(20),
    }


def _extract_genes(text_lower: str, text_original: str) -> set:
    """Extract likely gene symbols from text."""
    found = set()

    # Dictionary-based (known biomarker genes)
    for gene in COMMON_BIOMARKER_GENES:
        if re.search(r"\b" + re.escape(gene) + r"\b", text_lower):
            found.add(gene.upper())

    # Pattern-based (uppercase gene symbols)
    for match in GENE_SYMBOL_RE.finditer(text_original):
        symbol = match.group(1)
        if symbol not in GENE_FALSE_POSITIVES and len(symbol) <= 8:
            found.add(symbol)

    return found


def _extract_terms(text: str, term_set: set) -> set:
    """Find which terms from a set appear in text."""
    return {term for term in term_set if re.search(r"\b" + re.escape(term) + r"\b", text)}
