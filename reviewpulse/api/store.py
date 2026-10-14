"""Read-only DuckDB access to the processed review data.

Security rule for this module: no user-supplied value is ever pasted into SQL text.
 * Filter values (category, sentiment, ratings, dates, search text, topic id, limit, offset) are bound
   parameters (`?`), so quotes and SQL keywords in them are just data.
 * The only text that varies between queries is chosen from fixed allowlists in this file
   (sort columns and sort directions), never taken from the request.
 * The Parquet file path comes from configuration, not from requests.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb

logger = logging.getLogger(__name__)

SENTIMENTS: Tuple[str, ...] = ("positive", "neutral", "negative")

# SQL identifiers cannot be bound as parameters, so they are looked up in these allowlists.
SORT_COLUMNS: Dict[str, str] = {"date": "r.review_date", "rating": "r.rating", "helpful": "r.helpful_vote"}
SORT_ORDERS: Dict[str, str] = {"asc": "ASC", "desc": "DESC"}

MAX_SEARCH_LENGTH = 200


def escape_like(text: str) -> str:
    """Make `%`, `_` and `\\` in user text match literally inside a LIKE pattern (see ESCAPE clause below)."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sql_string_literal(value: str) -> str:
    """Quote a trusted, configuration-supplied string (the Parquet path) for use in a CREATE VIEW statement."""
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class ReviewFilters:
    """Validated filter values. Turned into a WHERE clause with placeholders by `where_clause`."""

    category: Optional[str] = None
    sentiment: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    search: Optional[str] = None
    verified_only: bool = False
    topic_id: Optional[int] = None

    def where_clause(self) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if self.category:
            clauses.append("r.category = ?")
            params.append(self.category)
        if self.sentiment:
            if self.sentiment not in SENTIMENTS:  # defence in depth: the API layer validates this first
                raise ValueError(f"Unsupported sentiment: {self.sentiment!r}")
            clauses.append("r.sentiment = ?")
            params.append(self.sentiment)
        if self.min_rating is not None:
            clauses.append("r.rating >= ?")
            params.append(float(self.min_rating))
        if self.max_rating is not None:
            clauses.append("r.rating <= ?")
            params.append(float(self.max_rating))
        # review_date is an ISO 'YYYY-MM-DD' string, so string comparison orders dates correctly.
        if self.date_from is not None:
            clauses.append("r.review_date >= ?")
            params.append(self.date_from.isoformat())
        if self.date_to is not None:
            clauses.append("r.review_date <= ?")
            params.append(self.date_to.isoformat())
        if self.search:
            clauses.append("r.combined_text ILIKE ? ESCAPE '\\'")
            params.append(f"%{escape_like(self.search)}%")
        if self.verified_only:
            clauses.append("r.verified_purchase = TRUE")
        if self.topic_id is not None:
            clauses.append("t.topic_id = ?")
            params.append(int(self.topic_id))
        return (" AND ".join(clauses) if clauses else "TRUE"), params


_FROM_WITH_TOPICS = "FROM reviews r LEFT JOIN review_topics t ON t.review_id = r.review_id"
_FROM_PLAIN = "FROM reviews r"


def _from(f: "ReviewFilters") -> str:
    """Join the topic table only when the filter needs it, so plain aggregates never depend on it."""
    return _FROM_WITH_TOPICS if f.topic_id is not None else _FROM_PLAIN


class ReviewStore:
    """Thin wrapper around an in-memory DuckDB database with views over the Parquet files."""

    def __init__(self, reviews_parquet: Path, review_topics_parquet: Optional[Path] = None):
        self.reviews_parquet = Path(reviews_parquet)
        self.review_topics_parquet = Path(review_topics_parquet) if review_topics_parquet else None
        self._lock = threading.Lock()
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._signature: Tuple[Optional[float], Optional[float]] = (None, None)
        self._connect()

    # ---------- connection management ----------
    @staticmethod
    def _mtime(path: Optional[Path]) -> Optional[float]:
        return path.stat().st_mtime if path is not None and path.exists() else None

    def _current_signature(self) -> Tuple[Optional[float], Optional[float]]:
        return (self._mtime(self.reviews_parquet), self._mtime(self.review_topics_parquet))

    def _connect(self) -> None:
        con = duckdb.connect(":memory:")
        if self.reviews_parquet.exists():
            con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet({_sql_string_literal(str(self.reviews_parquet))})")
        if self.review_topics_parquet is not None and self.review_topics_parquet.exists():
            con.execute(
                f"CREATE VIEW review_topics AS SELECT * FROM read_parquet({_sql_string_literal(str(self.review_topics_parquet))})"
            )
        else:
            con.execute("CREATE VIEW review_topics AS SELECT NULL::VARCHAR AS review_id, NULL::INTEGER AS topic_id, NULL::DOUBLE AS topic_score WHERE FALSE")
        self._con = con
        self._signature = self._current_signature()

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        # If the pipeline rewrote the Parquet files, pick the new data up without restarting the server.
        with self._lock:
            if self._current_signature() != self._signature:
                logger.info("Parquet files changed on disk; reloading views.")
                self._connect()
            assert self._con is not None
            return self._con.cursor()

    @property
    def has_reviews(self) -> bool:
        return self.reviews_parquet.exists()

    @property
    def has_topics(self) -> bool:
        return self.review_topics_parquet is not None and self.review_topics_parquet.exists()

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        cur = self._cursor()
        try:
            cur.execute(sql, list(params))
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            cur.close()

    # ---------- queries ----------
    def known_categories(self) -> List[str]:
        if not self.has_reviews:
            return []
        return [r["category"] for r in self._rows("SELECT DISTINCT category FROM reviews ORDER BY 1")]

    def total_reviews(self) -> int:
        if not self.has_reviews:
            return 0
        return int(self._rows("SELECT COUNT(*) AS n FROM reviews")[0]["n"])

    def eligible_for_retrieval(self) -> int:
        """Reviews with usable text. This is the number the semantic index is expected to cover."""
        if not self.has_reviews:
            return 0
        return int(self._rows("SELECT COUNT(*) AS n FROM reviews WHERE combined_text IS NOT NULL AND LENGTH(TRIM(combined_text)) >= 3")[0]["n"])

    def eligible_review_ids(self) -> List[str]:
        """IDs of the reviews the search index is expected to contain (used to detect an index built from older data)."""
        if not self.has_reviews:
            return []
        rows = self._rows("SELECT review_id FROM reviews WHERE combined_text IS NOT NULL AND LENGTH(TRIM(combined_text)) >= 3")
        return [r["review_id"] for r in rows]

    def categories_summary(self) -> List[Dict[str, Any]]:
        return self._rows(
            """
            SELECT category,
                   COUNT(*) AS review_count,
                   COUNT(DISTINCT parent_asin) AS unique_products,
                   ROUND(AVG(rating), 2) AS mean_rating,
                   ROUND(AVG(CAST(verified_purchase AS INT)) * 100, 1) AS verified_pct,
                   ROUND(AVG(text_length), 1) AS mean_text_length
            FROM reviews
            GROUP BY category
            ORDER BY review_count DESC, category
            """
        )

    def overview(self, f: ReviewFilters) -> Dict[str, Any]:
        where, params = f.where_clause()
        row = self._rows(
            f"""
            SELECT COUNT(*) AS total_reviews,
                   COUNT(DISTINCT r.parent_asin) AS total_products,
                   COUNT(DISTINCT r.user_id) AS total_users,
                   ROUND(AVG(r.rating), 2) AS average_rating,
                   MIN(r.review_date) AS date_min,
                   MAX(r.review_date) AS date_max,
                   COUNT(*) FILTER (WHERE r.sentiment = 'positive') AS positive,
                   COUNT(*) FILTER (WHERE r.sentiment = 'neutral') AS neutral,
                   COUNT(*) FILTER (WHERE r.sentiment = 'negative') AS negative
            {_from(f)}
            WHERE {where}
            """,
            params,
        )[0]
        return row

    def trend(self, f: ReviewFilters) -> List[Dict[str, Any]]:
        where, params = f.where_clause()
        return self._rows(
            f"""
            SELECT SUBSTR(r.review_date, 1, 7) AS year_month,
                   COUNT(*) AS review_count,
                   ROUND(AVG(r.rating), 2) AS avg_rating,
                   COUNT(*) FILTER (WHERE r.sentiment = 'positive') AS positive_count,
                   COUNT(*) FILTER (WHERE r.sentiment = 'neutral') AS neutral_count,
                   COUNT(*) FILTER (WHERE r.sentiment = 'negative') AS negative_count
            {_from(f)}
            WHERE {where}
            GROUP BY 1
            ORDER BY 1
            """,
            params,
        )

    def rating_counts(self, f: ReviewFilters) -> Dict[int, int]:
        where, params = f.where_clause()
        rows = self._rows(
            f"""
            SELECT CAST(ROUND(r.rating) AS INTEGER) AS stars, COUNT(*) AS n
            {_from(f)}
            WHERE {where}
            GROUP BY 1
            """,
            params,
        )
        return {int(r["stars"]): int(r["n"]) for r in rows}

    def topic_counts(self, f: ReviewFilters) -> List[Dict[str, Any]]:
        where, params = f.where_clause()
        return self._rows(
            f"""
            SELECT t.topic_id AS topic_id,
                   COUNT(*) AS review_count,
                   COUNT(*) FILTER (WHERE r.sentiment = 'positive') AS positive_count,
                   COUNT(*) FILTER (WHERE r.sentiment = 'neutral') AS neutral_count,
                   COUNT(*) FILTER (WHERE r.sentiment = 'negative') AS negative_count,
                   ROUND(AVG(r.rating), 2) AS avg_rating
            FROM reviews r JOIN review_topics t ON t.review_id = r.review_id
            WHERE {where}
            GROUP BY t.topic_id
            ORDER BY review_count DESC, t.topic_id
            """,
            params,
        )

    def count_reviews(self, f: ReviewFilters) -> int:
        where, params = f.where_clause()
        return int(self._rows(f"SELECT COUNT(*) AS n {_from(f)} WHERE {where}", params)[0]["n"])

    def reviews_page(self, f: ReviewFilters, sort: str, order: str, page: int, limit: int) -> List[Dict[str, Any]]:
        # Allowlist lookups: an unknown value raises KeyError instead of reaching the SQL text.
        sort_sql = SORT_COLUMNS[sort]
        order_sql = SORT_ORDERS[order]
        where, params = f.where_clause()
        offset = (page - 1) * limit
        return self._rows(
            f"""
            SELECT r.review_id, r.category, r.rating, r.sentiment, r.title, r.text, r.product_title,
                   r.review_date, r.verified_purchase, r.helpful_vote, t.topic_id, t.topic_score
            {_FROM_WITH_TOPICS}
            WHERE {where}
            ORDER BY {sort_sql} {order_sql}, r.review_id ASC
            LIMIT ? OFFSET ?
            """,
            [*params, int(limit), int(offset)],
        )

    def drift_frame(self):
        """Full review table as a DataFrame, for the drift check."""
        cur = self._cursor()
        try:
            return cur.execute("SELECT text_length, rating, helpful_vote FROM reviews").df()
        finally:
            cur.close()
