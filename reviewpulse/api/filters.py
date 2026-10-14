"""Request-parameter validation for the review endpoints.

FastAPI already rejects wrong types (a non-date `date_from`, a rating of 9, `sentiment=spam`) with a 422.
This module adds the checks FastAPI cannot express on its own: the category must be one the dataset really
has, and ranges must not be reversed. Invalid input gets a clear 422, never a 500, and never silently
widens a query (an unknown category is an error, not "show everything").
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Literal, Optional

from fastapi import HTTPException, Query, Request

from reviewpulse.api.store import MAX_SEARCH_LENGTH, ReviewFilters

SentimentName = Literal["positive", "neutral", "negative"]
SortName = Literal["date", "rating", "helpful"]
OrderName = Literal["asc", "desc"]


def _shorten(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else value[:limit] + "..."


def resolve_category(request: Request, category: Optional[str]) -> Optional[str]:
    """Map the requested category (case-insensitive) to its canonical name, or raise 422."""
    if category is None:
        return None
    wanted = category.strip()
    if not wanted:
        return None
    allowed: Iterable[str] = request.app.state.allowed_categories()
    lookup = {name.lower(): name for name in allowed}
    canonical = lookup.get(wanted.lower())
    if canonical is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{_shorten(wanted)}'. Valid categories: {', '.join(sorted(lookup.values()))}.",
        )
    return canonical


def check_date_range(date_from: Optional[date], date_to: Optional[date]) -> None:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to.")


def scope_filters(
    request: Request,
    category: Optional[str] = Query(None, max_length=80, description="Category name, e.g. Electronics. Omit for all categories."),
    date_from: Optional[date] = Query(None, description="Only reviews on or after this date (YYYY-MM-DD)."),
    date_to: Optional[date] = Query(None, description="Only reviews on or before this date (YYYY-MM-DD)."),
) -> ReviewFilters:
    """Category and date range, used by the overview, trend and rating-distribution endpoints."""
    check_date_range(date_from, date_to)
    return ReviewFilters(category=resolve_category(request, category), date_from=date_from, date_to=date_to)


def topic_filters(
    request: Request,
    category: Optional[str] = Query(None, max_length=80),
    sentiment: Optional[SentimentName] = Query(None, description="positive, neutral or negative"),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
) -> ReviewFilters:
    check_date_range(date_from, date_to)
    return ReviewFilters(category=resolve_category(request, category), sentiment=sentiment, date_from=date_from, date_to=date_to)


def review_filters(
    request: Request,
    category: Optional[str] = Query(None, max_length=80),
    sentiment: Optional[SentimentName] = Query(None, description="positive, neutral or negative"),
    min_rating: Optional[float] = Query(None, ge=1.0, le=5.0),
    max_rating: Optional[float] = Query(None, ge=1.0, le=5.0),
    date_from: Optional[date] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[date] = Query(None, description="YYYY-MM-DD"),
    search: Optional[str] = Query(None, max_length=MAX_SEARCH_LENGTH, description="Text to look for in the review title and body"),
    verified_only: bool = Query(False),
    topic_id: Optional[int] = Query(None, ge=0, le=999, description="Only reviews tagged with this topic"),
) -> ReviewFilters:
    check_date_range(date_from, date_to)
    if min_rating is not None and max_rating is not None and min_rating > max_rating:
        raise HTTPException(status_code=422, detail="min_rating must not be greater than max_rating.")
    cleaned_search = search.strip() if search else None
    if topic_id is not None and not request.app.state.store.has_topics:
        raise HTTPException(status_code=503, detail="Topic assignments are not available yet. Run the topic model first.")
    return ReviewFilters(
        category=resolve_category(request, category),
        sentiment=sentiment,
        min_rating=min_rating,
        max_rating=max_rating,
        date_from=date_from,
        date_to=date_to,
        search=cleaned_search or None,
        verified_only=verified_only,
        topic_id=topic_id,
    )
