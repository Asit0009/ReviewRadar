"""Tests for pipeline transformation and PyArrow schema conformance."""

import pyarrow as pa
import pytest

from reviewpulse.pipeline.cleaner import clean_and_normalize_record
from reviewpulse.pipeline.run_pipeline import REVIEW_PYARROW_SCHEMA


def test_clean_and_normalize_record_schema(sample_raw_record):
    """Verify cleaned record contains all fields expected in PyArrow schema."""
    cleaned = clean_and_normalize_record(sample_raw_record, category="All_Beauty")

    for field in REVIEW_PYARROW_SCHEMA:
        assert field.name in cleaned, f"Field {field.name} missing from cleaned record"

    assert cleaned["category"] == "All_Beauty"
    assert cleaned["sentiment"] == "positive"
    assert cleaned["sentiment_encoded"] == 2
    assert cleaned["rating"] == 5.0
    assert cleaned["text_length"] > 0
    assert cleaned["word_count"] > 0


def test_pyarrow_table_conversion(sample_processed_df):
    """Verify PyArrow Table conversion matches explicit schema without casting errors."""
    table = pa.Table.from_pandas(sample_processed_df, schema=REVIEW_PYARROW_SCHEMA, preserve_index=False)
    assert table.num_rows == len(sample_processed_df)
    assert table.num_columns == len(REVIEW_PYARROW_SCHEMA)
