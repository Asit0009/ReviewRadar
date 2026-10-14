"""Unit tests for schema validation, data quality rules, and text cleaning."""

import pytest

from reviewpulse.pipeline.cleaner import (
    clean_text,
    derive_sentiment_label,
    generate_review_id,
    normalize_timestamp,
)
from reviewpulse.pipeline.validator import ValidationReport, validate_review_record


def test_clean_text_html_and_whitespace():
    """Verify HTML entity unescaping and whitespace normalization."""
    raw = "Great &amp; sturdy! &quot;Loved it&quot;   \n\n Highly recommended.  "
    cleaned = clean_text(raw)
    assert cleaned == 'Great & sturdy! "Loved it" Highly recommended.'


def test_clean_text_empty_and_none():
    """Verify empty text handles gracefully."""
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_derive_sentiment_label():
    """Verify star ratings map strictly to negative, neutral, and positive classes."""
    # Negative (1-2 stars)
    label, enc = derive_sentiment_label(1.0)
    assert label == "negative" and enc == 0
    label, enc = derive_sentiment_label(2.0)
    assert label == "negative" and enc == 0

    # Neutral (3 stars)
    label, enc = derive_sentiment_label(3.0)
    assert label == "neutral" and enc == 1

    # Positive (4-5 stars)
    label, enc = derive_sentiment_label(4.0)
    assert label == "positive" and enc == 2
    label, enc = derive_sentiment_label(5.0)
    assert label == "positive" and enc == 2


def test_normalize_timestamp():
    """Verify millisecond timestamp converts to valid ISO-8601 UTC string."""
    # 1609459200000 = 2021-01-01T00:00:00+00:00
    iso_utc, date_str = normalize_timestamp(1609459200000)
    assert date_str == "2021-01-01"
    assert "2021-01-01" in iso_utc


def test_validate_valid_record(sample_raw_record):
    """Verify standard valid record passes validation."""
    report = ValidationReport("TestCategory")
    is_valid, reason = validate_review_record(sample_raw_record, report)
    assert is_valid is True
    assert reason is None
    assert report.valid_records == 1
    assert report.total_raw_read == 1


def test_validate_missing_field(sample_raw_record):
    """Verify missing required field fails validation."""
    report = ValidationReport("TestCategory")
    bad_record = sample_raw_record.copy()
    del bad_record["rating"]
    is_valid, reason = validate_review_record(bad_record, report)
    assert is_valid is False
    assert "Missing required field" in reason
    assert report.missing_required_fields == 1


def test_validate_invalid_rating(sample_raw_record):
    """Verify out-of-bounds ratings (0 or 6 stars) fail validation."""
    report = ValidationReport("TestCategory")
    bad_record = sample_raw_record.copy()
    bad_record["rating"] = 6.0
    is_valid, reason = validate_review_record(bad_record, report)
    assert is_valid is False
    assert "Invalid rating" in reason
    assert report.invalid_rating == 1


def test_validate_empty_text(sample_raw_record):
    """Verify empty or whitespace-only text fails validation."""
    report = ValidationReport("TestCategory")
    bad_record = sample_raw_record.copy()
    bad_record["text"] = "   "
    is_valid, reason = validate_review_record(bad_record, report)
    assert is_valid is False
    assert report.empty_or_null_text == 1


def test_deterministic_review_id():
    """Verify review ID generation is completely deterministic."""
    id1 = generate_review_id("user1", "asin1", 1600000000)
    id2 = generate_review_id("user1", "asin1", 1600000000)
    id3 = generate_review_id("user2", "asin1", 1600000000)
    assert id1 == id2
    assert id1 != id3
