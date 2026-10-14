"""Data cleaning and field normalization for Amazon Reviews 2023."""

import hashlib
import html
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


_BR_TAG = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
# The source data contains Amazon's own markers for attached videos and linked products, such as [[VIDEOID:ab12...]]
# and [[ASIN:B00...]]. They are not words the reviewer wrote, so they are removed.
_AMAZON_MARKER = re.compile(r"\[\[(?:VIDEOID|ASIN):.{0,400}?\]\]", re.IGNORECASE | re.DOTALL)  # product names may contain [brackets]

# When a reviewer writes no title, Amazon fills in the star rating as words: "Five Stars", "One Star", "4 stars".
# That is the label itself (the number matches the rating 99.7% of the time in this data), so it must not be part of
# the text the models read. Titles that merely mention stars ("Five stars for this blender") are real text and stay.
_DEFAULT_RATING_TITLE = re.compile(r"^(?:one|two|three|four|five|[1-5](?:\.0)?)[\s-]*stars?[\s.!]*$", re.IGNORECASE)


def is_default_rating_title(title: Optional[str]) -> bool:
    """True if the title is only Amazon's automatic star-rating label."""
    return bool(title) and _DEFAULT_RATING_TITLE.match(title.strip()) is not None


def contains_amazon_marker(text: Optional[str]) -> bool:
    """True if the text has an Amazon [[VIDEOID:...]] or [[ASIN:...]] marker."""
    return bool(text) and _AMAZON_MARKER.search(text) is not None


def clean_text(text: Optional[str]) -> str:
    """Clean review text while preserving semantic content for NLP models.

    Transformations:
    1. Unescape HTML entities (&quot;, &amp;, etc.).
    2. Remove HTML tags such as <br /> (line breaks become spaces). Only things shaped like a tag
       (`<` followed by a letter or `/`) are removed, so text such as "5 < 6 and 7 > 3" is kept.
       Amazon's [[VIDEOID:...]] and [[ASIN:...]] markers are removed too.
    3. Normalize whitespace (tabs, newlines, multiple spaces -> single space).
    4. Strip leading and trailing whitespace.
    """
    if not text:
        return ""
    # HTML unescaping
    cleaned = html.unescape(text)
    # Reviews in this dataset contain raw "<br />" tags. Left in, they become the fake word "br" in
    # TF-IDF features and topic models, and show up as markup in the dashboard.
    cleaned = _BR_TAG.sub(" ", cleaned)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = _AMAZON_MARKER.sub(" ", cleaned)
    # Normalize multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def derive_sentiment_label(rating: float) -> Tuple[str, int]:
    """Derive 3-class sentiment label from Amazon star rating.

    NOTE: These are rating-derived labels used as proxy ground truth,
    not independent human annotations. Star ratings represent customer
    satisfaction scores.

    Mapping:
    - 1.0 - 2.0 stars -> 'negative' (encoded: 0)
    - 3.0 stars       -> 'neutral'  (encoded: 1)
    - 4.0 - 5.0 stars -> 'positive' (encoded: 2)
    """
    if rating <= 2.0:
        return "negative", 0
    elif rating == 3.0:
        return "neutral", 1
    else:
        return "positive", 2


def generate_review_id(user_id: str, asin: str, timestamp: int) -> str:
    """Generate a reproducible unique ID for each review interaction."""
    raw_key = f"{user_id}_{asin}_{timestamp}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]


def normalize_timestamp(timestamp_ms: int) -> Tuple[str, str]:
    """Convert millisecond timestamp to UTC ISO-8601 and date strings."""
    # Note: Amazon Reviews 2023 timestamps are in milliseconds
    # E.g., 1600000000000 -> seconds = 1600000000
    if timestamp_ms > 10_000_000_000:
        timestamp_sec = timestamp_ms / 1000.0
    else:
        timestamp_sec = float(timestamp_ms)

    dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    iso_str = dt.isoformat()
    date_str = dt.strftime("%Y-%m-%d")
    return iso_str, date_str


def clean_and_normalize_record(
    record: Dict[str, Any],
    category: str,
    meta_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Transform raw validated record into clean normalized schema."""
    rating = float(record["rating"])
    sentiment_label, sentiment_encoded = derive_sentiment_label(rating)

    cleaned_title = clean_text(record.get("title", ""))
    cleaned_body = clean_text(record.get("text", ""))

    # Combined text for NLP/TF-IDF models: "Title. Body". An automatic "Five Stars" title would leak the rating
    # to the models, so it is left out of the model text (the `title` column keeps it for display).
    model_title = "" if is_default_rating_title(cleaned_title) else cleaned_title
    if model_title and cleaned_body:
        combined_text = f"{model_title}. {cleaned_body}"
    else:
        combined_text = cleaned_body or model_title

    user_id = str(record.get("user_id", "")).strip()
    asin = str(record.get("asin", "")).strip()
    parent_asin = str(record.get("parent_asin", "")).strip()
    timestamp_raw = int(record["timestamp"])
    iso_utc, date_str = normalize_timestamp(timestamp_raw)

    helpful_vote = int(record.get("helpful_vote", 0) or 0)
    verified_purchase = bool(record.get("verified_purchase", False))

    review_id = generate_review_id(user_id, asin, timestamp_raw)

    # Product metadata enrichment (if available)
    product_title = ""
    product_price = None
    if meta_lookup and parent_asin in meta_lookup:
        meta_info = meta_lookup[parent_asin]
        product_title = clean_text(meta_info.get("title", ""))
        price_val = meta_info.get("price")
        if price_val is not None:
            product_price = str(price_val).strip()

    return {
        "review_id": review_id,
        "category": category,
        "rating": rating,
        "sentiment": sentiment_label,
        "sentiment_encoded": sentiment_encoded,
        "title": cleaned_title,
        "text": cleaned_body,
        "combined_text": combined_text,
        "text_length": len(cleaned_body),
        "word_count": len(cleaned_body.split()),
        "user_id": user_id,
        "asin": asin,
        "parent_asin": parent_asin,
        "timestamp": timestamp_raw,
        "timestamp_utc": iso_utc,
        "review_date": date_str,
        "helpful_vote": helpful_vote,
        "verified_purchase": verified_purchase,
        "product_title": product_title,
        "product_price": product_price,
    }
