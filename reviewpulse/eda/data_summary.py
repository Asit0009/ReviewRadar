"""Per-category data accounting: what was downloaded, what was kept, what was used for training and search.

Everything is read from files the pipeline already wrote (the manifest, the Parquet file, metrics.json, the topic
tags and the search index description), so the numbers cannot drift from the real artifacts.

    python -m reviewpulse.eda.data_summary      ->  reports/data_summary.md and reports/data_summary.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reviewpulse.config import MANIFEST_PATH, MODELS_DIR, NLP_DIR, OFFICIAL_CATEGORIES, REPORTS_DIR, VECTOR_STORE_DIR, DATA_PROCESSED_DIR
from reviewpulse.modeling.dataset import load_and_split_data

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_summary() -> Dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH) or {}
    metrics = _read_json(MODELS_DIR / "metrics.json") or {}
    index_info = _read_json(VECTOR_STORE_DIR / "index_info.json") or {}
    parquet = DATA_PROCESSED_DIR / "reviews_all.parquet"
    df = pd.read_parquet(parquet, columns=["review_id", "category", "user_id", "rating", "review_date"]) if parquet.exists() else pd.DataFrame()

    topic_tags = NLP_DIR / "review_topics.parquet"
    tagged = pd.read_parquet(topic_tags, columns=["review_id"]) if topic_tags.exists() else pd.DataFrame(columns=["review_id"])
    tagged_by_cat = df[df["review_id"].isin(set(tagged["review_id"]))].groupby("category").size().to_dict() if len(df) else {}

    split_counts: Dict[str, Dict[str, int]] = {}
    if parquet.exists():
        train, val, test, _ = load_and_split_data()
        for name, part in (("train", train), ("validation", val), ("test", test)):
            for cat, n in part.groupby("category").size().items():
                split_counts.setdefault(cat, {})[name] = int(n)

    reports = manifest.get("pipeline_processing", {}).get("category_reports", {})
    rows: List[Dict[str, Any]] = []
    for cat in OFFICIAL_CATEGORIES:
        m = manifest.get("categories", {}).get(cat, {})
        r = reports.get(cat, {})
        sub = df[df["category"] == cat] if len(df) else df
        sp = split_counts.get(cat, {})
        rows.append(
            {
                "category": cat,
                "raw_reviews_downloaded": m.get("downloaded_raw_reviews"),
                "sampling": m.get("sampling", {}).get("strategy"),
                "distinct_reviewers_raw": m.get("distinct_reviewers"),
                "rejected_by_validation": (r.get("total_raw_read", 0) - r.get("retained_after_cleaning", 0)) if r else None,
                "reviews_in_processed_dataset": int(len(sub)) if len(df) else None,
                "distinct_reviewers_processed": int(sub["user_id"].nunique()) if len(sub) else None,
                "date_min": str(sub["review_date"].min()) if len(sub) else None,
                "date_max": str(sub["review_date"].max()) if len(sub) else None,
                "model_train": sp.get("train"),
                "model_validation": sp.get("validation"),
                "model_test": sp.get("test"),
                "topic_tagged": int(tagged_by_cat.get(cat, 0)) if len(df) else None,
                "semantic_index_eligible": index_info.get("per_category", {}).get(cat, {}).get("eligible"),
                "semantic_index_indexed": index_info.get("per_category", {}).get(cat, {}).get("indexed"),
            }
        )
    total = {k: (sum(r[k] for r in rows if isinstance(r[k], (int, float))) if any(isinstance(r[k], (int, float)) for r in rows) else None) for k in rows[0] if k not in ("category", "sampling", "date_min", "date_max")}
    return {
        "dataset": manifest.get("dataset_name"),
        "split": {k: metrics.get("split_info", {}).get(k) for k in ("source_rows", "total_deduplicated_samples", "train_samples", "val_samples", "test_samples", "split_strategy", "reviewers_shared_between_splits")},
        "index": {k: index_info.get(k) for k in ("eligible_reviews", "indexed_reviews", "coverage_pct", "limited_by", "embedding_model", "built_at")},
        "categories": rows,
        "totals": total,
    }


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:,}" if isinstance(v, int) else str(v)


def write_reports() -> Dict[str, Any]:
    summary = build_summary()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cols = [
        ("category", "Category"), ("raw_reviews_downloaded", "Downloaded"), ("distinct_reviewers_raw", "Reviewers"),
        ("reviews_in_processed_dataset", "Kept after cleaning"), ("model_train", "Train"), ("model_validation", "Val"),
        ("model_test", "Test"), ("topic_tagged", "Topic-tagged"), ("semantic_index_indexed", "Search-indexed"),
    ]
    lines = ["# Data Summary by Category", "", "Generated by `python -m reviewpulse.eda.data_summary` from the manifest, the processed Parquet file, `metrics.json`, the topic tags and `index_info.json`.", ""]
    lines.append("| " + " | ".join(h for _, h in cols) + " |")
    lines.append("| :--- " + "| ---: " * (len(cols) - 1) + "|")
    for r in summary["categories"]:
        lines.append("| " + " | ".join(f"`{r[k]}`" if k == "category" else _fmt(r[k]) for k, _ in cols) + " |")
    t = summary["totals"]
    lines.append("| **Total** | " + " | ".join(f"**{_fmt(t.get(k))}**" for k, _ in cols[1:]) + " |")
    s, i = summary["split"], summary["index"]
    lines += [
        "",
        "## How to read this",
        "",
        "- **Downloaded**: reviews saved by the acquisition step (sampled from many places in each source file).",
        "- **Reviewers**: distinct reviewer accounts among the downloaded reviews. This is used to check the sample is not dominated by a few people.",
        "- **Kept after cleaning**: reviews that passed validation (valid rating, text, timestamp) and were not duplicates.",
        f"- **Train / Val / Test**: reviews used for modeling after also removing identical review texts ({_fmt(s.get('total_deduplicated_samples'))} in total). Split strategy: {s.get('split_strategy')}. Reviewers shared between splits: {s.get('reviewers_shared_between_splits')}.",
        "- **Topic-tagged**: reviews that received a topic from the NMF topic model (reviews with no known words get no tag).",
        f"- **Search-indexed**: reviews in the semantic search index. Index covers {_fmt(i.get('indexed_reviews'))} of {_fmt(i.get('eligible_reviews'))} eligible reviews ({i.get('coverage_pct')}%); limited by: {i.get('limited_by') or 'nothing'}; embedding model: {i.get('embedding_model')}.",
        "",
    ]
    (REPORTS_DIR / "data_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Write reports/data_summary.md and .json: reviews per category at every stage, from the saved files.").parse_args()  # no options; this makes --help print help instead of running the job
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    write_reports()
    print((REPORTS_DIR / "data_summary.md").read_text(encoding="utf-8"))
