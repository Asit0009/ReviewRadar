"""Functional tests for the API. Expected numbers are recomputed independently with pandas."""

import math

import pandas as pd
import pytest

from data_factory import CATEGORIES


def get(client, path, **params):
    resp = client.get(path, params=params)
    assert resp.status_code == 200, f"{path} {params} -> {resp.status_code} {resp.text[:200]}"
    return resp.json()


def in_range(df: pd.DataFrame, date_from=None, date_to=None) -> pd.DataFrame:
    if date_from:
        df = df[df["review_date"] >= date_from]
    if date_to:
        df = df[df["review_date"] <= date_to]
    return df


# ---------- health, categories ----------
def test_health_reports_what_is_loaded(client, fixture_df):
    data = get(client, "/api/v1/health")
    assert data["status"] == "healthy"
    assert data["total_reviews"] == len(fixture_df)
    assert data["artifacts_ready"]["parquet_data"] and data["artifacts_ready"]["sentiment_models"]
    assert data["artifacts_ready"]["vector_search_faiss"] and data["artifacts_ready"]["nmf_topics"]
    assert data["artifacts_ready"]["distilbert_model"] is False  # not part of the fixture
    assert data["retrieval"]["indexed_reviews"] == data["retrieval"]["eligible_reviews"] == len(fixture_df)
    assert data["retrieval"]["coverage_pct"] == 100.0 and data["retrieval"]["limited_by"] is None
    assert data["assistant"] == {"llm_configured": False, "llm_model": "gemini-test-model"}


def test_categories_match_the_data(client, fixture_df):
    data = get(client, "/api/v1/categories")
    assert data["total_categories"] == len(CATEGORIES)
    by_name = {c["category"]: c for c in data["categories"]}
    assert set(by_name) == set(CATEGORIES)
    for name, group in fixture_df.groupby("category"):
        assert by_name[name]["review_count"] == len(group)
        assert by_name[name]["mean_rating"] == pytest.approx(round(group["rating"].mean(), 2))
        assert by_name[name]["unique_products"] == group["parent_asin"].nunique()


# ---------- overview ----------
def test_overview_all_categories(client, fixture_df):
    data = get(client, "/api/v1/overview")
    assert data["total_reviews"] == len(fixture_df)
    assert data["average_rating"] == pytest.approx(round(fixture_df["rating"].mean(), 2))
    assert data["sentiment_distribution"] == fixture_df["sentiment"].value_counts().to_dict() | {}
    assert sum(data["sentiment_percentages"].values()) == pytest.approx(100, abs=0.05)
    assert data["category_filter"] is None and data["date_from"] is None


@pytest.mark.parametrize("category", CATEGORIES)
def test_overview_each_category(client, fixture_df, category):
    data = get(client, "/api/v1/overview", category=category)
    sub = fixture_df[fixture_df["category"] == category]
    assert data["total_reviews"] == len(sub)
    assert data["average_rating"] == pytest.approx(round(sub["rating"].mean(), 2))
    assert data["sentiment_distribution"]["negative"] == (sub["sentiment"] == "negative").sum()
    assert data["category_filter"] == category


def test_overview_category_is_case_insensitive(client):
    assert get(client, "/api/v1/overview", category="electronics")["category_filter"] == "Electronics"


def test_overview_date_filter_changes_the_numbers(client, fixture_df):
    full = get(client, "/api/v1/overview")
    data = get(client, "/api/v1/overview", date_from="2022-01-01", date_to="2022-12-31")
    sub = in_range(fixture_df, "2022-01-01", "2022-12-31")
    assert 0 < data["total_reviews"] == len(sub) < full["total_reviews"]
    assert data["date_min"] >= "2022-01-01" and data["date_max"] <= "2022-12-31"
    assert data["average_rating"] == pytest.approx(round(sub["rating"].mean(), 2))


def test_overview_with_no_matching_reviews_has_no_average(client):
    data = get(client, "/api/v1/overview", date_from="1990-01-01", date_to="1990-12-31")
    assert data["total_reviews"] == 0
    assert data["average_rating"] is None
    assert data["sentiment_percentages"] == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}


# ---------- monthly sentiment trend ----------
def test_trend_matches_pandas_monthly_counts(client, fixture_df):
    data = get(client, "/api/v1/sentiment/trend")
    df = fixture_df.assign(month=fixture_df["review_date"].str[:7])
    expected = df.groupby("month").agg(n=("rating", "size"), avg=("rating", "mean"))
    assert data["total_data_points"] == len(expected) == len(data["trend"])
    assert [p["year_month"] for p in data["trend"]] == sorted(expected.index)
    for point in data["trend"]:
        sub = df[df["month"] == point["year_month"]]
        assert point["review_count"] == len(sub)
        assert point["avg_rating"] == pytest.approx(round(sub["rating"].mean(), 2))
        assert point["positive_count"] == (sub["sentiment"] == "positive").sum()
        assert point["neutral_count"] == (sub["sentiment"] == "neutral").sum()
        assert point["negative_count"] == (sub["sentiment"] == "negative").sum()
        assert point["positive_count"] + point["neutral_count"] + point["negative_count"] == point["review_count"]
        assert point["positive_ratio"] == pytest.approx(point["positive_count"] / point["review_count"], abs=1e-4)


def test_trend_respects_category_and_dates(client, fixture_df):
    data = get(client, "/api/v1/sentiment/trend", category="Pet_Supplies", date_from="2021-06-01", date_to="2022-06-30")
    sub = in_range(fixture_df[fixture_df["category"] == "Pet_Supplies"], "2021-06-01", "2022-06-30")
    assert sum(p["review_count"] for p in data["trend"]) == len(sub) > 0
    assert data["trend"][0]["year_month"] >= "2021-06" and data["trend"][-1]["year_month"] <= "2022-06"
    assert data["category_filter"] == "Pet_Supplies" and data["date_from"] == "2021-06-01"


# ---------- rating distribution ----------
@pytest.mark.parametrize("category", [None, *CATEGORIES])
def test_rating_distribution_matches_pandas(client, fixture_df, category):
    params = {"category": category} if category else {}
    data = get(client, "/api/v1/ratings/distribution", **params)
    sub = fixture_df if category is None else fixture_df[fixture_df["category"] == category]
    assert [b["stars"] for b in data["buckets"]] == [5, 4, 3, 2, 1]
    counts = sub["rating"].round().astype(int).value_counts().to_dict()
    for bucket in data["buckets"]:
        assert bucket["count"] == counts.get(bucket["stars"], 0)
    assert data["total_reviews"] == len(sub) == sum(b["count"] for b in data["buckets"])
    assert sum(b["pct"] for b in data["buckets"]) == pytest.approx(100, abs=0.1)


def test_rating_distribution_date_filter(client, fixture_df):
    data = get(client, "/api/v1/ratings/distribution", date_from="2022-01-01")
    assert data["total_reviews"] == len(in_range(fixture_df, "2022-01-01")) < len(fixture_df)


# ---------- topics ----------
def test_topics_counts_follow_filters(client, fixture_df):
    data = get(client, "/api/v1/topics")
    assert data["review_tags_available"] is True and data["total_topics"] == len(CATEGORIES)
    assert data["reviews_considered"] == sum(t["prevalence_count"] for t in data["topics"])
    counts = [t["prevalence_count"] for t in data["topics"]]
    assert counts == sorted(counts, reverse=True)

    neg = get(client, "/api/v1/topics", sentiment="negative", category="Automotive")
    topic_id = CATEGORIES.index("Automotive")
    by_id = {t["topic_id"]: t for t in neg["topics"]}
    assert by_id[topic_id]["prevalence_count"] == neg["reviews_considered"] > 0
    assert all(t["prevalence_count"] == 0 for tid, t in by_id.items() if tid != topic_id)
    assert by_id[topic_id]["sentiment_counts"]["negative"] == by_id[topic_id]["prevalence_count"]
    assert neg["sentiment_filter"] == "negative" and neg["category_filter"] == "Automotive"


# ---------- reviews ----------
def test_reviews_pagination(client, fixture_df):
    page1 = get(client, "/api/v1/reviews", page=1, limit=25)
    page2 = get(client, "/api/v1/reviews", page=2, limit=25)
    assert page1["total_records"] == len(fixture_df)
    assert page1["total_pages"] == math.ceil(len(fixture_df) / 25)
    assert len(page1["reviews"]) == len(page2["reviews"]) == 25
    ids1, ids2 = [r["review_id"] for r in page1["reviews"]], [r["review_id"] for r in page2["reviews"]]
    assert not set(ids1) & set(ids2)
    last = get(client, "/api/v1/reviews", page=page1["total_pages"], limit=25)
    assert len(last["reviews"]) == len(fixture_df) - 25 * (page1["total_pages"] - 1)
    # the same request twice gives the same order (stable tie-break on review_id)
    assert ids1 == [r["review_id"] for r in get(client, "/api/v1/reviews", page=1, limit=25)["reviews"]]


def test_reviews_default_order_is_newest_first(client):
    dates = [r["review_date"] for r in get(client, "/api/v1/reviews", limit=100)["reviews"]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize("sort,order", [("rating", "asc"), ("rating", "desc"), ("date", "asc"), ("helpful", "desc")])
def test_reviews_sorting(client, sort, order):
    data = get(client, "/api/v1/reviews", sort=sort, order=order, limit=100)
    key = {"rating": "rating", "date": "review_date", "helpful": "helpful_vote"}[sort]
    values = [r[key] for r in data["reviews"]]
    assert values == sorted(values, reverse=(order == "desc"))
    assert data["sort"] == sort and data["order"] == order


@pytest.mark.parametrize("category", CATEGORIES)
def test_reviews_category_filter_each_category(client, fixture_df, category):
    data = get(client, "/api/v1/reviews", category=category, limit=100)
    assert data["total_records"] == (fixture_df["category"] == category).sum()
    assert {r["category"] for r in data["reviews"]} == {category}


def test_reviews_date_filter(client, fixture_df):
    data = get(client, "/api/v1/reviews", date_from="2022-03-01", date_to="2022-09-30", limit=100)
    sub = in_range(fixture_df, "2022-03-01", "2022-09-30")
    assert data["total_records"] == len(sub) > 0
    assert all("2022-03-01" <= r["review_date"] <= "2022-09-30" for r in data["reviews"])
    # boundaries are inclusive
    one_day = sub["review_date"].iloc[0]
    assert get(client, "/api/v1/reviews", date_from=one_day, date_to=one_day)["total_records"] == (fixture_df["review_date"] == one_day).sum()


def test_reviews_combined_filters(client, fixture_df):
    params = dict(category="Electronics", sentiment="negative", min_rating=1, max_rating=2, date_from="2021-01-01", date_to="2022-12-31", verified_only="true")
    data = get(client, "/api/v1/reviews", limit=100, **params)
    sub = in_range(fixture_df[(fixture_df["category"] == "Electronics") & (fixture_df["sentiment"] == "negative") & fixture_df["verified_purchase"]], "2021-01-01", "2022-12-31")
    assert data["total_records"] == len(sub)
    assert all(r["verified_purchase"] and r["sentiment"] == "negative" for r in data["reviews"])


def test_reviews_rating_range(client, fixture_df):
    data = get(client, "/api/v1/reviews", min_rating=3, max_rating=4, limit=100)
    assert data["total_records"] == fixture_df["rating"].between(3, 4).sum()
    assert {r["rating"] for r in data["reviews"]} <= {3.0, 4.0}


def test_reviews_topic_tags_and_filter(client, fixture_df):
    tagged = get(client, "/api/v1/reviews", category="Pet_Supplies", limit=100)["reviews"]
    with_tag = [r for r in tagged if r["topic_id"] is not None]
    assert with_tag and all(r["topic_id"] == CATEGORIES.index("Pet_Supplies") and r["topic_label"] == "Pet Stuff" and 0 < r["topic_score"] <= 1 for r in with_tag)
    assert any(r["topic_id"] is None and r["topic_label"] is None for r in tagged)  # every 10th review is untagged

    topic = CATEGORIES.index("Automotive")
    data = get(client, "/api/v1/reviews", topic_id=topic, limit=100)
    assert data["total_records"] > 0 and {r["category"] for r in data["reviews"]} == {"Automotive"}
    assert {r["topic_id"] for r in data["reviews"]} == {topic}


def test_reviews_text_is_clean_of_html(client):
    texts = " ".join(r["text"] for r in get(client, "/api/v1/reviews", limit=100)["reviews"])
    assert "<br" not in texts.lower()


# ---------- predict ----------
def test_predict_logistic_regression_returns_probabilities(client):
    data = client.post("/api/v1/predict", json={"text": "The battery drains quickly and it stopped working", "model_type": "logistic_regression"}).json()
    assert data["predicted_sentiment"] in {"positive", "neutral", "negative"}
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)
    assert data["model_used"] == "TF-IDF + LogisticRegression" and data["is_rating_derived_label"] is True
    assert data["sentiment_encoded"] == ["negative", "neutral", "positive"].index(data["predicted_sentiment"])


def test_predict_linear_svc_has_no_probabilities(client):
    data = client.post("/api/v1/predict", json={"text": "Easy to install and fits perfectly", "model_type": "linear_svc"}).json()
    assert data["probabilities"] is None and data["model_used"] == "TF-IDF + LinearSVC"


def test_predict_distilbert_is_unavailable_until_trained(client):
    resp = client.post("/api/v1/predict", json={"text": "Works great", "model_type": "distilbert"})
    assert resp.status_code == 503
    assert "DistilBERT" in resp.json()["detail"]


# ---------- model metrics ----------
def test_model_metrics_are_read_from_saved_file(client):
    data = get(client, "/api/v1/models/metrics")
    assert data["status"] == "available" and set(data["available_models"]) == {"logistic_regression", "linear_svc"}
    assert data["metrics"]["models"]["linear_svc"]["test"]["accuracy"] == 0.86
    assert "per_category" not in data["metrics"] and data["category_metrics"] is None


def test_model_metrics_per_category(client):
    data = get(client, "/api/v1/models/metrics", category="Pet_Supplies")
    assert data["category_filter"] == "Pet_Supplies"
    assert data["category_metrics"]["category"] == "Pet_Supplies"
    assert set(data["category_metrics"]["models"]) == {"logistic_regression", "linear_svc"}
    assert "reliable" in data["category_metrics"]["models"]["linear_svc"]


def test_model_metrics_by_category_table(client):
    data = get(client, "/api/v1/models/metrics/by-category")
    assert data["categories"] == sorted(CATEGORIES) and data["min_class_support_for_reliable"] == 20
    assert set(data["models"]) == {"logistic_regression", "linear_svc"}
    assert data["models"]["linear_svc"]["Pet_Supplies"]["test_samples"] == 40
    assert get(client, "/api/v1/health")["status"] == "healthy"


# ---------- assistant (no Gemini key in the fixture) ----------
def test_assistant_without_a_key_is_clearly_retrieval_only(client, fixture_df):
    data = client.post("/api/v1/assistant/chat", json={"question": "my dog refused to eat this food", "category": "Pet_Supplies", "top_k": 3}).json()
    assert data["llm_used"] is False and data["mode"] == "retrieval_only" and data["status"] == "setup_required"
    assert data["llm_model"] is None
    assert "no Gemini API key" in data["response"] and "Gemini wrote" not in data["response"]
    assert 0 < data["evidence_count"] <= 3 == len(data["evidence_reviews"]) + (3 - data["evidence_count"])
    assert {e["category"] for e in data["evidence_reviews"]} == {"Pet_Supplies"}
    sub = fixture_df[fixture_df["category"] == "Pet_Supplies"]
    assert data["numerical_facts"]["total_reviews"] == len(sub)
    assert data["numerical_facts"]["average_rating"] == pytest.approx(round(sub["rating"].mean(), 2))
    assert data["numerical_facts"]["scope"] == "Pet_Supplies"
    assert data["retrieval"]["indexed_reviews"] == data["retrieval"]["eligible_reviews"] == len(fixture_df)
    assert data["retrieval"]["coverage_pct"] == 100.0


def test_assistant_all_categories_facts(client, fixture_df):
    data = client.post("/api/v1/assistant/chat", json={"question": "battery life"}).json()
    assert data["numerical_facts"]["scope"] == "All Categories" and data["numerical_facts"]["total_reviews"] == len(fixture_df)


# ---------- monitoring ----------
def test_monitoring_status(client):
    data = get(client, "/api/v1/monitoring/status")
    assert data["status"] in {"healthy", "drift_detected"}
    assert data["reference_sample_size"] + data["current_sample_size"] == 240
    assert set(data["feature_metrics"]) == {"text_length", "rating", "helpful_vote"}


# ---------- cross-cutting ----------
def test_cors_allows_only_configured_origins(client):
    ok = client.options("/api/v1/overview", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in ok.headers
    bad = client.options("/api/v1/overview", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_security_headers_present(client):
    resp = client.get("/api/v1/health")
    assert resp.headers["x-content-type-options"] == "nosniff" and resp.headers["x-frame-options"] == "DENY"


def test_openapi_documents_the_new_parameters(client):
    spec = client.get("/openapi.json").json()
    reviews = {p["name"] for p in spec["paths"]["/api/v1/reviews"]["get"]["parameters"]}
    assert {"category", "sentiment", "min_rating", "max_rating", "date_from", "date_to", "search", "verified_only", "topic_id", "sort", "order", "page", "limit"} <= reviews
    assert "/api/v1/ratings/distribution" in spec["paths"]
    trend_item = spec["components"]["schemas"]["SentimentTrendItem"]["properties"]
    assert {"positive_count", "neutral_count", "negative_count", "positive_ratio"} <= set(trend_item)
