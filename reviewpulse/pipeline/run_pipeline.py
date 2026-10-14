"""End-to-end data pipeline runner for ReviewPulse.

Validates raw records, cleans text, enriches with metadata, deduplicates,
and writes optimized Parquet datasets using PyArrow and DuckDB.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reviewpulse.acquisition.downloader import load_manifest, save_manifest
from reviewpulse.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    OFFICIAL_CATEGORIES,
    ensure_directories,
)
from reviewpulse.pipeline.cleaner import clean_and_normalize_record, contains_amazon_marker, is_default_rating_title
from reviewpulse.pipeline.validator import ValidationReport, validate_review_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# PyArrow Schema for standardized Parquet files
REVIEW_PYARROW_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("category", pa.string()),
        ("rating", pa.float32()),
        ("sentiment", pa.string()),
        ("sentiment_encoded", pa.int8()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("combined_text", pa.string()),
        ("text_length", pa.int32()),
        ("word_count", pa.int32()),
        ("user_id", pa.string()),
        ("asin", pa.string()),
        ("parent_asin", pa.string()),
        ("timestamp", pa.int64()),
        ("timestamp_utc", pa.string()),
        ("review_date", pa.string()),
        ("helpful_vote", pa.int32()),
        ("verified_purchase", pa.bool_()),
        ("product_title", pa.string()),
        ("product_price", pa.string()),
    ]
)


def load_meta_lookup(category: str) -> Dict[str, Dict]:
    """Load product metadata into in-memory dictionary keyed by parent_asin."""
    meta_path = DATA_RAW_DIR / f"{category}_meta.jsonl"
    lookup = {}
    if not meta_path.exists():
        logger.warning(f"Meta file {meta_path} not found. Proceeding without meta enrichment.")
        return lookup

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parent_asin = obj.get("parent_asin")
                    if parent_asin:
                        lookup[parent_asin] = {
                            "title": obj.get("title", ""),
                            "price": obj.get("price"),
                            "main_category": obj.get("main_category", ""),
                        }
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(lookup):,} product metadata entries for {category}")
    except Exception as e:
        logger.warning(f"Error loading metadata for {category}: {e}")
    return lookup


def process_category(category: str) -> Tuple[pd.DataFrame, ValidationReport]:
    """Run validation, cleaning, and Parquet export for a single category."""
    raw_file = DATA_RAW_DIR / f"{category}_reviews.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw reviews file {raw_file} does not exist. Run acquisition first."
        )

    report = ValidationReport(category)
    meta_lookup = load_meta_lookup(category)

    seen_keys = set()
    cleaned_rows = []

    with open(raw_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError:
                report.malformed_json += 1
                continue

            is_valid, reason = validate_review_record(raw_record, report)
            if not is_valid:
                continue

            # Deduplication key: (user_id, asin, timestamp)
            dedup_key = (
                str(raw_record.get("user_id")),
                str(raw_record.get("asin")),
                int(raw_record.get("timestamp")),
            )
            if dedup_key in seen_keys:
                report.duplicate_records += 1
                continue
            seen_keys.add(dedup_key)

            # Clean and normalize
            cleaned_record = clean_and_normalize_record(raw_record, category, meta_lookup)
            cleaned_rows.append(cleaned_record)
            if is_default_rating_title(cleaned_record["title"]):
                report.automatic_titles_left_out += 1
            if contains_amazon_marker(raw_record.get("text")) or contains_amazon_marker(raw_record.get("title")):
                report.reviews_with_amazon_markers += 1

    report.retained_after_cleaning = len(cleaned_rows)
    df = pd.DataFrame(cleaned_rows)

    # Save category Parquet
    out_parquet = DATA_PROCESSED_DIR / f"{category}.parquet"
    if not df.empty:
        table = pa.Table.from_pandas(df, schema=REVIEW_PYARROW_SCHEMA, preserve_index=False)
        pq.write_table(table, out_parquet, compression="snappy")
        logger.info(
            f"Saved {category}.parquet: {len(df):,} rows ({out_parquet.stat().st_size / 1024:.1f} KB)"
        )
    else:
        logger.warning(f"No valid rows retained for category {category}!")

    return df, report


CLEANING_RULES = [
    "invalid records are dropped and counted by reason (see category_reports)",
    "records with the same (user_id, asin, timestamp) are kept once",
    "HTML entities are unescaped and HTML tags removed",
    "Amazon [[VIDEOID:...]] and [[ASIN:...]] markers are removed",
    "whitespace is normalised",
    "an automatic star-rating title (\"Five Stars\") is kept out of combined_text (the title column keeps it)",
    "sentiment is derived from the rating: 1-2 negative, 3 neutral, 4-5 positive",
]


def describe_scope(manifest: Dict, reports: Dict[str, Dict], combined_df: pd.DataFrame) -> Dict:
    """A plain summary of what was processed, so the manifest alone says what the processed data covers."""
    acquired = {name: entry for name, entry in manifest.get("categories", {}).items() if name in reports}
    return {
        "categories": sorted(reports),
        "category_count": len(reports),
        "target_reviews_per_category": sorted({entry.get("target_limit_requested") for entry in acquired.values()} - {None}),
        "sampling": sorted({json.dumps({k: v for k, v in (entry.get("sampling") or {}).items() if k in ("strategy", "chunks", "seed")}, sort_keys=True) for entry in acquired.values()}),
        "downloaded_reviews": int(sum(r["total_raw_read"] for r in reports.values())),
        "retained_reviews": int(len(combined_df)),
        "rejected_reviews": int(sum(r["total_raw_read"] - r["retained_after_cleaning"] for r in reports.values())),
        "distinct_reviewers_in_download": int(sum(entry.get("distinct_reviewers", 0) for entry in acquired.values())),
        "review_dates": {"first": str(combined_df["review_date"].min()), "last": str(combined_df["review_date"].max())},
        "reviews_per_category": {str(k): int(v) for k, v in combined_df["category"].value_counts().sort_index().items()},
        "automatic_titles_left_out": int(sum(r["automatic_titles_left_out"] for r in reports.values())),
        "reviews_with_amazon_markers": int(sum(r["reviews_with_amazon_markers"] for r in reports.values())),
        "cleaning_rules": CLEANING_RULES,
    }


def run_full_pipeline(categories: Optional[List[str]] = None) -> pd.DataFrame:
    """Run pipeline across all specified categories, save combined Parquet and update manifest."""
    ensure_directories()
    target_categories = categories or OFFICIAL_CATEGORIES
    logger.info(f"Running pipeline for {len(target_categories)} categories: {target_categories}")

    all_dfs = []
    reports = {}

    for cat in target_categories:
        df, report = process_category(cat)
        reports[cat] = report.to_dict()
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No processed data was produced. Pipeline failed.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined Parquet
    combined_parquet = DATA_PROCESSED_DIR / "reviews_all.parquet"
    combined_table = pa.Table.from_pandas(
        combined_df, schema=REVIEW_PYARROW_SCHEMA, preserve_index=False
    )
    pq.write_table(combined_table, combined_parquet, compression="snappy")

    logger.info(
        f"Combined dataset saved to {combined_parquet}: "
        f"{len(combined_df):,} total rows ({combined_parquet.stat().st_size / (1024*1024):.2f} MB)"
    )

    # Validate with DuckDB
    con = duckdb.connect()
    duck_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{combined_parquet}')"
    ).fetchone()[0]
    cat_distribution = con.execute(
        f"SELECT category, COUNT(*) as cnt, AVG(rating) as avg_rating "
        f"FROM read_parquet('{combined_parquet}') GROUP BY category ORDER BY cnt DESC"
    ).df()
    con.close()

    logger.info(f"DuckDB validation verified {duck_count:,} rows in combined Parquet.")
    logger.info(f"Category distribution:\n{cat_distribution.to_string(index=False)}")

    # Update manifest
    manifest = load_manifest()
    manifest["pipeline_processing"] = {
        "processed_file": combined_parquet.relative_to(DATA_PROCESSED_DIR.parent.parent).as_posix(),
        "total_retained_reviews": len(combined_df),
        "category_reports": reports,
        "scope": describe_scope(manifest, reports, combined_df),
    }
    save_manifest(manifest)

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReviewPulse data processing pipeline.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=OFFICIAL_CATEGORIES,
        help="Categories to process",
    )
    args = parser.parse_args()
    run_full_pipeline(args.categories)
