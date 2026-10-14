"""Data validation and schema verification for raw Amazon Reviews 2023 records."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Expected raw fields from Amazon Reviews 2023 dataset
REQUIRED_REVIEW_FIELDS = ["rating", "text", "asin", "parent_asin", "user_id", "timestamp"]
OPTIONAL_REVIEW_FIELDS = ["title", "images", "helpful_vote", "verified_purchase"]

REQUIRED_META_FIELDS = ["parent_asin", "main_category", "title"]


class ValidationReport:
    """Tracks validation statistics and reasons for record rejection."""

    def __init__(self, category: str):
        self.category = category
        self.total_raw_read = 0
        self.valid_records = 0
        self.malformed_json = 0
        self.missing_required_fields = 0
        self.invalid_rating = 0
        self.empty_or_null_text = 0
        self.invalid_timestamp = 0
        self.duplicate_records = 0
        self.retained_after_cleaning = 0
        self.automatic_titles_left_out = 0  # "Five Stars" style titles kept out of the model text
        self.reviews_with_amazon_markers = 0  # reviews whose text had [[VIDEOID:...]] / [[ASIN:...]] removed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "total_raw_read": self.total_raw_read,
            "valid_records": self.valid_records,
            "malformed_json": self.malformed_json,
            "missing_required_fields": self.missing_required_fields,
            "invalid_rating": self.invalid_rating,
            "empty_or_null_text": self.empty_or_null_text,
            "invalid_timestamp": self.invalid_timestamp,
            "duplicate_records": self.duplicate_records,
            "retained_after_cleaning": self.retained_after_cleaning,
            "automatic_titles_left_out": self.automatic_titles_left_out,
            "reviews_with_amazon_markers": self.reviews_with_amazon_markers,
            "rejection_rate_percent": round(
                (
                    (self.total_raw_read - self.retained_after_cleaning)
                    / max(1, self.total_raw_read)
                )
                * 100,
                2,
            ),
        }


def validate_review_record(
    record: Dict[str, Any], report: ValidationReport
) -> Tuple[bool, Optional[str]]:
    """Validate an individual raw review record against data quality rules.

    Rules:
    1. Must contain all required fields.
    2. Rating must be a numeric value between 1.0 and 5.0 inclusive.
    3. Text must be non-empty string with at least 2 non-whitespace characters.
    4. Timestamp must be a positive integer or parseable unix epoch timestamp.
    """
    report.total_raw_read += 1

    # Rule 1: Check required fields
    for field in REQUIRED_REVIEW_FIELDS:
        if field not in record or record[field] is None:
            report.missing_required_fields += 1
            return False, f"Missing required field: {field}"

    # Rule 2: Rating validation
    try:
        rating = float(record["rating"])
        if rating < 1.0 or rating > 5.0:
            report.invalid_rating += 1
            return False, f"Invalid rating: {rating} (must be between 1.0 and 5.0)"
    except (ValueError, TypeError):
        report.invalid_rating += 1
        return False, f"Non-numeric rating: {record.get('rating')}"

    # Rule 3: Text validation
    text = record.get("text")
    if not isinstance(text, str) or len(text.strip()) < 2:
        report.empty_or_null_text += 1
        return False, "Empty or sub-minimal review text"

    # Rule 4: Timestamp validation
    try:
        ts = int(record["timestamp"])
        if ts <= 0:
            report.invalid_timestamp += 1
            return False, f"Invalid non-positive timestamp: {ts}"
    except (ValueError, TypeError):
        report.invalid_timestamp += 1
        return False, f"Malformed timestamp: {record.get('timestamp')}"

    report.valid_records += 1
    return True, None
