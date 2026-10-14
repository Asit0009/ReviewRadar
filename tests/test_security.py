"""SQL-injection and input-validation regression tests.

The original bug: `category` and `sentiment` were pasted into SQL with f-strings, so
    GET /reviews?category=zzz' OR '1'='1     returned EVERY review, and
    GET /reviews?sentiment=x'                crashed with HTTP 500.
These tests pin the fix: user input is only ever a bound parameter, invalid input gets a 422, and an invalid
filter can never widen a result.
"""

import re
from datetime import date
from pathlib import Path

import pytest

from reviewpulse.api.store import ReviewFilters, escape_like

CATEGORY_PAYLOADS = [
    "zzz' OR '1'='1",  # the exact input from the original finding
    "Electronics' OR '1'='1' --",
    "x'; DROP TABLE reviews; --",
    "' UNION SELECT review_id FROM reviews --",
    "Electronics') OR ('1'='1",
    'Electronics" OR "1"="1',
    "'",
    "''",
    "\\",
    "%",
    "Elec%",
    "_lectronics",
    "1; SELECT 1",
    "Electronics\x00",
    "Électronics",
    "A" * 500,
]

SENTIMENT_PAYLOADS = ["x'", "positive' OR '1'='1", "positive'; DROP TABLE reviews; --", "POSITIVE", " positive", "%", "", "neutral,negative"]

CATEGORY_ENDPOINTS = ["/api/v1/overview", "/api/v1/sentiment/trend", "/api/v1/ratings/distribution", "/api/v1/topics", "/api/v1/reviews", "/api/v1/models/metrics"]


# ---------- the original bug ----------
def test_original_category_injection_no_longer_returns_every_review(client, fixture_df):
    resp = client.get("/api/v1/reviews", params={"limit": 1, "category": "zzz' OR '1'='1"})
    assert resp.status_code == 422
    assert "records" not in resp.text and "total_records" not in resp.text
    assert len(fixture_df) > 0  # sanity: there WERE rows an injection could have returned


def test_original_sentiment_quote_no_longer_crashes(client):
    resp = client.get("/api/v1/reviews", params={"limit": 1, "sentiment": "x'"})
    assert resp.status_code == 422  # was HTTP 500


# ---------- every endpoint, every payload ----------
@pytest.mark.parametrize("endpoint", CATEGORY_ENDPOINTS)
@pytest.mark.parametrize("payload", CATEGORY_PAYLOADS)
def test_malicious_category_is_rejected_everywhere(client, endpoint, payload):
    resp = client.get(endpoint, params={"category": payload})
    assert resp.status_code in (422,), f"{endpoint} accepted category={payload!r}: {resp.status_code}"
    assert "Valid categories" in resp.json()["detail"] or resp.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/v1/reviews", "/api/v1/topics"])
@pytest.mark.parametrize("payload", SENTIMENT_PAYLOADS)
def test_malicious_sentiment_is_rejected(client, endpoint, payload):
    resp = client.get(endpoint, params={"sentiment": payload})
    assert resp.status_code == 422


@pytest.mark.parametrize("payload", CATEGORY_PAYLOADS)
def test_malicious_category_in_assistant_request(client, payload):
    resp = client.post("/api/v1/assistant/chat", json={"question": "battery life", "category": payload})
    assert resp.status_code == 422


def test_invalid_filter_never_widens_the_result(client, fixture_df):
    """A valid filter narrows; an invalid one must error. It must never behave like 'no filter'."""
    everything = client.get("/api/v1/reviews", params={"limit": 1}).json()["total_records"]
    assert everything == len(fixture_df)
    narrowed = client.get("/api/v1/reviews", params={"limit": 1, "category": "Electronics"}).json()["total_records"]
    assert narrowed == (fixture_df["category"] == "Electronics").sum() < everything
    for payload in CATEGORY_PAYLOADS:
        assert client.get("/api/v1/reviews", params={"limit": 1, "category": payload}).status_code != 200


def test_drop_table_attempt_leaves_data_intact(client, fixture_df):
    client.get("/api/v1/reviews", params={"category": "x'; DROP TABLE reviews; --"})
    client.get("/api/v1/reviews", params={"search": "'; DROP TABLE reviews; --"})
    after = client.get("/api/v1/overview").json()["total_reviews"]
    assert after == len(fixture_df)


# ---------- search text is data, not SQL ----------
@pytest.mark.parametrize("payload", ["zzz' OR '1'='1", "'; DROP TABLE reviews; --", "' UNION SELECT 1 --", '"', "it's", "\\", "'''"])
def test_search_injection_matches_only_literal_text(client, payload):
    resp = client.get("/api/v1/reviews", params={"search": payload, "limit": 1})
    assert resp.status_code == 200
    assert resp.json()["total_records"] == 0  # no review contains that literal text


def test_search_percent_and_underscore_are_literal(client, fixture_df):
    # If % and _ acted as LIKE wildcards, "%" and "_" would match every review. As literals they match only
    # reviews that really contain the character, which pandas can count independently.
    for term in ["100%", "great_value", "%", "_"]:
        expected = int(fixture_df["combined_text"].str.contains(term, case=False, regex=False).sum())
        got = client.get("/api/v1/reviews", params={"search": term}).json()["total_records"]
        assert got == expected, f"search={term!r}: API {got} vs python {expected}"
    assert client.get("/api/v1/reviews", params={"search": "100%"}).json()["total_records"] == 1
    assert client.get("/api/v1/reviews", params={"search": "%"}).json()["total_records"] == 1  # not "everything"
    assert client.get("/api/v1/reviews", params={"search": "_"}).json()["total_records"] < len(fixture_df)


def test_search_is_case_insensitive_and_matches_python_count(client, fixture_df):
    expected = int(fixture_df["combined_text"].str.contains("battery", case=False, regex=False).sum())
    assert expected > 0
    assert client.get("/api/v1/reviews", params={"search": "BATTERY"}).json()["total_records"] == expected


# ---------- dates, ratings, sort, pagination ----------
@pytest.mark.parametrize("bad", ["2022-01-01' OR '1'='1", "notadate", "2023-13-45", "2020-02-30", "01/02/2022", "'", "2022-1-1x"])
@pytest.mark.parametrize("name", ["date_from", "date_to"])
def test_bad_dates_rejected(client, name, bad):
    assert client.get("/api/v1/reviews", params={name: bad}).status_code == 422
    assert client.get("/api/v1/overview", params={name: bad}).status_code == 422


def test_reversed_date_range_rejected(client):
    for path in ["/api/v1/reviews", "/api/v1/overview", "/api/v1/sentiment/trend", "/api/v1/ratings/distribution"]:
        resp = client.get(path, params={"date_from": "2023-01-01", "date_to": "2022-01-01"})
        assert resp.status_code == 422, path
        assert "date_from" in resp.json()["detail"]


@pytest.mark.parametrize("params", [{"min_rating": 0}, {"min_rating": 6}, {"max_rating": -1}, {"min_rating": "1' OR '1'='1"}, {"max_rating": "abc"}])
def test_bad_rating_values_rejected(client, params):
    assert client.get("/api/v1/reviews", params=params).status_code == 422


def test_min_rating_above_max_rating_rejected(client):
    resp = client.get("/api/v1/reviews", params={"min_rating": 5, "max_rating": 2})
    assert resp.status_code == 422


@pytest.mark.parametrize("params", [{"sort": "rating;DROP TABLE reviews"}, {"sort": "review_id"}, {"sort": "r.rating"}, {"sort": "1"}, {"sort": ""}, {"order": "asc;--"}, {"order": "DESC"}, {"order": "random()"}])
def test_sort_and_order_are_allowlisted(client, params):
    assert client.get("/api/v1/reviews", params=params).status_code == 422


@pytest.mark.parametrize("params", [{"page": 0}, {"page": -1}, {"page": "abc"}, {"page": 10**9}, {"limit": 0}, {"limit": 101}, {"limit": -5}, {"limit": "1; DROP"}])
def test_bad_pagination_rejected(client, params):
    assert client.get("/api/v1/reviews", params=params).status_code == 422


def test_page_beyond_the_end_is_empty_not_an_error(client):
    resp = client.get("/api/v1/reviews", params={"page": 999999, "limit": 100})
    assert resp.status_code == 200
    assert resp.json()["reviews"] == []


@pytest.mark.parametrize("value", ["-1", "1 OR 1=1", "abc", "99999", "1.5"])
def test_bad_topic_id_rejected(client, value):
    assert client.get("/api/v1/reviews", params={"topic_id": value}).status_code == 422


def test_search_too_long_rejected(client):
    assert client.get("/api/v1/reviews", params={"search": "a" * 201}).status_code == 422


def test_predict_rejects_unknown_model_instead_of_silently_falling_back(client):
    resp = client.post("/api/v1/predict", json={"text": "great product", "model_type": "logistic_regression'; --"})
    assert resp.status_code == 422


@pytest.mark.parametrize("body", [{"text": "a"}, {"text": ""}, {"text": "x" * 5001}, {}, {"text": 5}])
def test_predict_input_validation(client, body):
    assert client.post("/api/v1/predict", json=body).status_code == 422


@pytest.mark.parametrize("body", [{"question": "a"}, {"question": "x" * 501}, {"question": "ok question", "top_k": 0}, {"question": "ok question", "top_k": 11}])
def test_chat_input_validation(client, body):
    assert client.post("/api/v1/assistant/chat", json=body).status_code == 422


# ---------- unit-level guarantees ----------
def test_where_clause_binds_values_instead_of_inlining_them():
    payload = "x' OR '1'='1"
    f = ReviewFilters(category=payload, sentiment="positive", search=payload, date_from=date(2022, 1, 1), min_rating=2, max_rating=4, topic_id=3, verified_only=True)
    sql, params = f.where_clause()
    assert payload not in sql and "OR '1'='1" not in sql
    assert sql.count("?") == len(params)
    assert params.count(payload) == 1  # category
    assert f"%{escape_like(payload)}%" in params  # search


def test_unknown_sentiment_is_refused_even_if_validation_is_bypassed():
    with pytest.raises(ValueError):
        ReviewFilters(sentiment="x' OR '1'='1").where_clause()


def test_escape_like():
    assert escape_like("100%_a\\b") == "100\\%\\_a\\\\b"


def test_source_has_no_string_built_sql_with_user_values():
    """Guard against the bug coming back: no f-string SQL that quotes a variable inline."""
    root = Path(__file__).resolve().parent.parent / "reviewpulse"
    risky = re.compile(r"""(=|LIKE|ILIKE|IN)\s*\(?\s*['"]\{|['"]\{(category|sentiment|search|clean_search|text|query)[^}]*\}['"]|'%\{""", re.IGNORECASE)
    offenders = []
    for path in list((root / "api").glob("*.py")) + list((root / "assistant").glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if risky.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "possible string-built SQL:\n" + "\n".join(offenders)


# ---------- the same checks against the real data (skipped if it has not been built) ----------
@pytest.mark.realdata
def test_real_data_original_injection_payloads(real_client, real_settings):
    total = real_client.get("/api/v1/overview").json()["total_reviews"]
    assert total > 1000
    bad = real_client.get("/api/v1/reviews", params={"limit": 1, "category": "zzz' OR '1'='1"})
    assert bad.status_code == 422
    assert real_client.get("/api/v1/reviews", params={"limit": 1, "sentiment": "x'"}).status_code == 422
    assert real_client.get("/api/v1/overview").json()["total_reviews"] == total  # data untouched
