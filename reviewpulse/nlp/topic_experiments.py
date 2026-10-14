"""Compare TF-IDF + NMF settings on the real reviews, so the topic configuration is chosen from evidence.

    python -m reviewpulse.nlp.topic_experiments          # about 8 minutes on a laptop CPU

Every setting is fitted with three random seeds. The results are saved to reports/topic_experiments.json and are
used by `python -m reviewpulse.nlp.topic_report`.

Measures (all computed from the data):
  biggest_topic_pct  share of the tagged reviews whose strongest topic is the largest topic. A very large value
                     means one topic is a catch-all.
  npmi               coherence of each topic's top-10 single words, from how often they appear in the same
                     review (-1 to +1, higher is better). NPMI is the usual way to judge topic quality.
  distinct_words     share of different words among all topics' top-10 words (1.0 = no word is repeated).
  weak_pct           share of reviews whose strongest topic holds less than half of their topic weight. Reviews
                     that talk about several things (price AND size) are weakly assigned by nature.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from reviewpulse.config import DATA_PROCESSED_DIR, REPORTS_DIR
from reviewpulse.nlp.topic_model import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_DF,
    DEFAULT_N_TOPICS,
    DOMAIN_STOP_WORDS,
    MIN_WORDS_TO_FIT,
    PRAISE_STOP_WORDS,
)

logger = logging.getLogger(__name__)
RESULTS_PATH = REPORTS_DIR / "topic_experiments.json"
SEEDS = (0, 1, 2)

# Everyday filler that was tried as extra stop words and rejected (it made the biggest topic larger).
FILLER_WORDS = frozenset(
    "using long far set new box worked small light look looks looking seems feel feels right enough getting recommend "
    "highly definitely awesome excellent perfect amazing happy satisfied fine super loves loved easy old year years month "
    "months week weeks first second back still even ever never always though although however especially sure know think "
    "thought put take took see seen kind".split()
)
BASE_STOPS = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)

# name -> settings. "v1" is what the project used first; "adopted" is what topic_model.py uses now.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "v1: 14 topics, 20,000 features, min_df 5 (first version)": dict(stops=BASE_STOPS, k=14, max_features=20000, min_df=5),
    "20 topics, same words as v1": dict(stops=BASE_STOPS, k=20, max_features=20000, min_df=5),
    "30 topics, same words as v1": dict(stops=BASE_STOPS, k=30, max_features=20000, min_df=5),
    "14 topics + extra filler stop words (rejected)": dict(stops=sorted(set(BASE_STOPS) | FILLER_WORDS), k=14, max_features=20000, min_df=5),
    "20 topics + praise stop words + min_df 20 (adopted)": dict(
        stops=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS | PRAISE_STOP_WORDS), k=DEFAULT_N_TOPICS, max_features=DEFAULT_MAX_FEATURES, min_df=DEFAULT_MIN_DF
    ),
}


def npmi_coherence(binary_docs, term_columns: List[List[int]]) -> float:
    """Mean NPMI over all word pairs of every topic, using co-occurrence in the same review."""
    n_docs = binary_docs.shape[0]
    per_topic = []
    for columns in term_columns:
        sub = binary_docs[:, columns].astype(np.float32)
        doc_freq = np.asarray(sub.sum(axis=0)).ravel()
        both = (sub.T @ sub).toarray()
        pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                p_i, p_j, p_ij = doc_freq[i] / n_docs, doc_freq[j] / n_docs, both[i, j] / n_docs
                if p_ij <= 0:
                    pairs.append(-1.0)
                else:
                    pairs.append(float(np.log(p_ij / (p_i * p_j)) / -np.log(p_ij)) if p_ij < 1 else 1.0)
        per_topic.append(np.mean(pairs))
    return float(np.mean(per_topic))


def evaluate(texts: pd.Series, word_counts: pd.Series, cfg: Dict[str, Any], seed: int) -> Dict[str, float]:
    vectorizer = TfidfVectorizer(max_features=cfg["max_features"], ngram_range=(1, 2), stop_words=cfg["stops"], min_df=cfg["min_df"], max_df=0.5)
    fit_mask = (word_counts >= MIN_WORDS_TO_FIT).to_numpy()
    vectorizer.fit(texts[fit_mask])
    everything = vectorizer.transform(texts)
    nmf = NMF(n_components=cfg["k"], random_state=seed, init="nndsvda", max_iter=400).fit(everything[fit_mask])
    weights = nmf.transform(everything)
    names = vectorizer.get_feature_names_out()

    has_signal = weights.sum(axis=1) > 0
    strongest = weights.argmax(axis=1)
    shares = np.divide(weights, weights.sum(axis=1)[:, None], out=np.zeros_like(weights), where=has_signal[:, None])
    sizes = np.bincount(strongest[has_signal], minlength=cfg["k"]) / has_signal.sum()

    top_unigrams = [[i for i in np.argsort(row)[::-1] if " " not in names[i]][:10] for row in nmf.components_]
    flat = [names[i] for cols in top_unigrams for i in cols]
    return {
        "biggest_topic_pct": round(float(sizes.max() * 100), 1),
        "npmi": round(npmi_coherence((everything[fit_mask] > 0).tocsc(), top_unigrams), 3),
        "distinct_words": round(len(set(flat)) / len(flat), 3),
        "weak_pct": round(float((shares.max(axis=1)[has_signal] < 0.5).mean() * 100), 1),
    }


def run(parquet_path: Path = DATA_PROCESSED_DIR / "reviews_all.parquet", output_path: Path = RESULTS_PATH) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    texts = df["combined_text"].fillna("")
    word_counts = texts.str.split().str.len()
    results: Dict[str, Any] = {"reviews": int(len(df)), "seeds": list(SEEDS), "configs": {}}
    for name, cfg in CONFIGS.items():
        start = time.time()
        runs = [evaluate(texts, word_counts, cfg, seed) for seed in SEEDS]
        summary = {"topics": cfg["k"], "runs": runs}
        for key in runs[0]:
            values = [r[key] for r in runs]
            summary[key] = {"mean": round(float(np.mean(values)), 3), "min": min(values), "max": max(values)}
        results["configs"][name] = summary
        logger.info("%-58s biggest %5.1f%%  npmi %+.3f  distinct %.2f  weak %4.1f%%  (%.0fs)", name, summary["biggest_topic_pct"]["mean"], summary["npmi"]["mean"], summary["distinct_words"]["mean"], summary["weak_pct"]["mean"], time.time() - start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved %s", output_path)
    return results


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Compare TF-IDF + NMF settings (about 8 minutes) and save reports/topic_experiments.json.").parse_args()  # no options; this makes --help print help instead of running the job
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
