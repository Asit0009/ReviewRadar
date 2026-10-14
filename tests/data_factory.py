"""Builds a small, fully deterministic ReviewPulse data set for tests.

It uses the real cleaning code and the real Parquet schema, so tests exercise the same paths as production
but with data whose correct answers are known in advance. Nothing here touches the real data or models.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from reviewpulse.config import Settings
from reviewpulse.nlp.semantic_search import build_faiss_index
from reviewpulse.pipeline.cleaner import clean_and_normalize_record
from reviewpulse.pipeline.run_pipeline import REVIEW_PYARROW_SCHEMA

CATEGORIES = ["Electronics", "Pet_Supplies", "Automotive", "Office_Products"]
REVIEWS_PER_CATEGORY = 60

# (positive, neutral, negative) sentences per category. Words are chosen so similarity search has an obvious answer.
TEXTS: Dict[str, Dict[str, List[str]]] = {
    "Electronics": {
        "pos": ["The battery lasts all day and the screen is bright and sharp.", "Great headphones, excellent sound quality and comfortable fit."],
        "neu": ["The charger works but the cable feels a bit flimsy."],
        "neg": ["The battery drains quickly and the phone stopped working after a week.", "Screen cracked easily, terrible build quality."],
    },
    "Pet_Supplies": {
        "pos": ["My dog loves this food and his coat looks shiny.", "The cat tree is sturdy and my cats play on it every day."],
        "neu": ["The dog bed is okay, a little smaller than expected."],
        "neg": ["My dog refused to eat this food and got an upset stomach.", "The leash snapped on the first walk, very poor quality."],
    },
    "Automotive": {
        "pos": ["Easy to install and fits my truck perfectly, wipers work great.", "The floor mats fit exactly and look good."],
        "neu": ["The oil filter fits but the packaging was damaged."],
        "neg": ["The brake pads squeal loudly and wore out in two months.", "Car phone mount broke on the first bumpy road."],
    },
    "Office_Products": {
        "pos": ["The printer paper is bright white and never jams.", "Comfortable office chair with good back support."],
        "neu": ["Pens write fine but the ink runs out sooner than I hoped."],
        "neg": ["The stapler jammed constantly and the stapler broke within days.", "Printer ink cartridge leaked all over the box."],
    },
}

# A review that contains characters which are special inside SQL LIKE patterns, to test literal matching.
SPECIAL_TEXT = "Saved 100% of my time, great_value for money"


def _rating_for(i: int) -> float:
    # 60% five stars, 20% four, 8% three, 12% one or two. Same pattern in every category.
    slot = i % 25
    if slot < 15:
        return 5.0
    if slot < 20:
        return 4.0
    if slot < 22:
        return 3.0
    return 1.0 if slot % 2 else 2.0


def build_reviews() -> pd.DataFrame:
    rows = []
    base = datetime(2021, 1, 15, tzinfo=timezone.utc)
    for c_idx, category in enumerate(CATEGORIES):
        for i in range(REVIEWS_PER_CATEGORY):
            rating = _rating_for(i)
            kind = "pos" if rating >= 4 else "neu" if rating == 3 else "neg"
            options = TEXTS[category][kind]
            text = options[i % len(options)] + f" Unit {i}."
            if category == "Electronics" and i == 0:
                text = SPECIAL_TEXT
            timestamp_ms = int((base.timestamp() + (i * 13 + c_idx) * 86400) * 1000)  # one review every ~13 days
            record = {
                "rating": rating,
                "title": f"{category} review {i}",
                "text": text + (" <br />Second line." if i % 7 == 0 else ""),
                "asin": f"A{c_idx}{i:03d}",
                "parent_asin": f"P{c_idx}{i % 20:02d}",
                "user_id": f"U{c_idx}{i:03d}",
                "timestamp": timestamp_ms,
                "helpful_vote": i % 5,
                "verified_purchase": i % 3 != 0,
            }
            rows.append(clean_and_normalize_record(record, category))
    return pd.DataFrame(rows)


def fake_encoder(texts) -> np.ndarray:
    """Deterministic bag-of-words embedding: texts that share words get a high cosine similarity.

    2048 dimensions keep accidental hash collisions rare, so unrelated sentences score near zero.
    """
    dim = 2048
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for r, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", str(text).lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            out[r, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def write_dataset(root: Path) -> Settings:
    """Write parquet, topic tags, metrics, tiny trained models and a vector store under `root`."""
    processed, models, nlp, vectors = root / "processed", root / "models", root / "nlp", root / "vectors"
    for d in (processed, models, nlp, vectors):
        d.mkdir(parents=True, exist_ok=True)

    df = build_reviews()
    pq.write_table(pa.Table.from_pandas(df, schema=REVIEW_PYARROW_SCHEMA, preserve_index=False), processed / "reviews_all.parquet")

    # Topic tags: topic = category index (0..3) for most reviews; every 10th review is left untagged.
    tagged = df[~df.index.isin(df.index[::10])].copy()
    tagged["topic_id"] = tagged["category"].map({c: i for i, c in enumerate(CATEGORIES)}).astype("int32")
    tagged["topic_score"] = np.float32(0.8)
    tagged[["review_id", "topic_id", "topic_score"]].to_parquet(nlp / "review_topics.parquet", index=False)
    topics = [
        {
            "topic_id": i,
            "label": f"{c.split('_')[0]} Stuff",
            "top_words": [c.lower(), "quality", "price"],
            "sample_reviews": [f"Example review for {c}."],
            "prevalence_count": int((tagged["topic_id"] == i).sum()),
            "prevalence_pct": 25.0,
        }
        for i, c in enumerate(CATEGORIES)
    ]
    (nlp / "topic_labels.json").write_text(json.dumps({"n_topics": 4, "topics": topics}), encoding="utf-8")

    # Tiny real models trained on the fixture text, saved with the same file names as production.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(df["combined_text"])
    y = df["sentiment_encoded"].to_numpy()
    joblib.dump(vectorizer, models / "tfidf_vectorizer.joblib")
    joblib.dump(LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y), models / "logistic_regression.joblib")
    joblib.dump(LinearSVC(class_weight="balanced").fit(X, y), models / "linear_svc.joblib")

    def metric_block(acc: float) -> dict:
        return {"accuracy": acc, "macro_f1": acc - 0.2, "weighted_f1": acc, "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    per_cat = {c: {"test_samples": 40, "reliable": i % 2 == 0, "accuracy": 0.8, "macro_f1": 0.6} for i, c in enumerate(CATEGORIES)}
    (models / "metrics.json").write_text(
        json.dumps(
            {
                "split_info": {"train_samples": 1, "test_samples": 1},
                "models": {
                    "logistic_regression": {"validation": metric_block(0.8), "test": metric_block(0.82)},
                    "linear_svc": {"validation": metric_block(0.85), "test": metric_block(0.86)},
                },
                "per_category": {"logistic_regression": per_cat, "linear_svc": per_cat},
            }
        ),
        encoding="utf-8",
    )

    build_faiss_index(parquet_path=processed / "reviews_all.parquet", output_dir=vectors, encoder=fake_encoder, embedding_model="fake-bow-128")

    return Settings(
        processed_dir=processed,
        models_dir=models,
        nlp_dir=nlp,
        vector_store_dir=vectors,
        frontend_dist_dir=root / "no-dist",
        categories=tuple(CATEGORIES),
        cors_origins=("http://localhost:5173",),
        gemini_api_key="",
        gemini_model="gemini-test-model",
        gemini_timeout_seconds=2.0,
        retrieval_min_score=0.25,
    )
