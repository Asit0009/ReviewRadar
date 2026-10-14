"""Checks against the REAL data and saved artifacts (skipped when they have not been built).

These tests tie the manifest, raw files, processed Parquet file, topic tags, search index and API together, so a
stale or half-finished pipeline run shows up as a failing test instead of a silently wrong dashboard.
"""

import json

import pandas as pd
import pytest

from reviewpulse.config import DATA_RAW_DIR, MANIFEST_PATH, OFFICIAL_CATEGORIES

pytestmark = pytest.mark.realdata

EXPECTED_CATEGORIES = [
    "Electronics", "Home_and_Kitchen", "Beauty_and_Personal_Care", "Sports_and_Outdoors", "Toys_and_Games",
    "Pet_Supplies", "Automotive", "Cell_Phones_and_Accessories", "Health_and_Household", "Office_Products",
]


@pytest.fixture(scope="module")
def manifest(real_settings):
    if not MANIFEST_PATH.exists():
        pytest.skip("no manifest")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reviews(real_settings):
    return pd.read_parquet(real_settings.reviews_parquet)


def test_the_project_covers_exactly_the_ten_required_categories(reviews, real_settings):
    assert sorted(OFFICIAL_CATEGORIES) == sorted(EXPECTED_CATEGORIES)
    assert sorted(reviews["category"].unique()) == sorted(EXPECTED_CATEGORIES)
    assert sorted(real_settings.categories) == sorted(EXPECTED_CATEGORIES)


def test_every_category_has_a_substantial_balanced_sample(reviews):
    counts = reviews["category"].value_counts()
    assert counts.min() >= 2000, counts.to_dict()
    assert counts.max() / counts.min() < 1.5  # no category dominates


def test_manifest_matches_the_raw_files_on_disk(manifest):
    for category in EXPECTED_CATEGORIES:
        entry = manifest["categories"][category]
        raw = DATA_RAW_DIR / f"{category}_reviews.jsonl"
        lines = [line for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert entry["downloaded_raw_reviews"] == len(lines), category
        assert entry["review_source_url"].endswith(f"/{category}.jsonl")
        assert entry["sampling"]["strategy"] in ("spread", "head")


def test_raw_data_comes_from_many_different_reviewers(manifest):
    """The source files are stored in runs by reviewer, so reading only the head gives a few people's reviews."""
    for category in EXPECTED_CATEGORIES:
        entry = manifest["categories"][category]
        assert entry["sampling"]["strategy"] == "spread", f"{category} was not spread-sampled"
        assert entry["distinct_reviewers"] / entry["downloaded_raw_reviews"] > 0.25, category


def test_pipeline_accounting_adds_up_per_category(manifest, reviews):
    reports = manifest["pipeline_processing"]["category_reports"]
    for category in EXPECTED_CATEGORIES:
        r = reports[category]
        # every non-blank raw line is either rejected for one reason, dropped as a duplicate, or kept
        rejected = r["malformed_json"] + (r["total_raw_read"] - r["valid_records"])
        assert r["valid_records"] - r["duplicate_records"] == r["retained_after_cleaning"], category
        assert r["retained_after_cleaning"] == (reviews["category"] == category).sum(), category
        assert rejected >= 0
    assert manifest["pipeline_processing"]["total_retained_reviews"] == len(reviews)


def test_reviews_are_spread_over_time(reviews):
    years = reviews["review_date"].str[:4].astype(int)
    assert years.min() <= 2015 and years.max() >= 2022 and years.nunique() >= 8


def test_model_test_support_sums_to_the_split(real_settings):
    metrics = json.loads(real_settings.metrics_json.read_text(encoding="utf-8"))
    for model, per_category in metrics["per_category"].items():
        assert sum(v["test_samples"] for v in per_category.values()) == metrics["split_info"]["test_samples"], model
    assert metrics["split_info"]["reviewers_shared_between_splits"] == 0


def test_api_counts_match_the_parquet_for_every_category(real_client, reviews):
    api = {c["category"]: c for c in real_client.get("/api/v1/categories").json()["categories"]}
    assert set(api) == set(EXPECTED_CATEGORIES)
    for category, group in reviews.groupby("category"):
        assert api[category]["review_count"] == len(group)
        overview = real_client.get("/api/v1/overview", params={"category": category}).json()
        assert overview["total_reviews"] == len(group)
        assert overview["average_rating"] == pytest.approx(round(group["rating"].mean(), 2))
    assert real_client.get("/api/v1/overview").json()["total_reviews"] == len(reviews)


def test_api_date_filter_matches_pandas_on_real_data(real_client, reviews):
    got = real_client.get("/api/v1/reviews", params={"date_from": "2021-01-01", "date_to": "2021-12-31", "category": "Electronics", "limit": 1}).json()["total_records"]
    sub = reviews[(reviews["category"] == "Electronics") & reviews["review_date"].between("2021-01-01", "2021-12-31")]
    assert got == len(sub) > 0


def test_monthly_sentiment_trend_adds_up_on_real_data(real_client, reviews):
    trend = real_client.get("/api/v1/sentiment/trend").json()["trend"]
    assert sum(p["review_count"] for p in trend) == len(reviews)
    assert sum(p["positive_count"] for p in trend) == (reviews["sentiment"] == "positive").sum()
    assert sum(p["negative_count"] for p in trend) == (reviews["sentiment"] == "negative").sum()


def test_health_reports_full_search_coverage(real_client, reviews):
    """Every searchable review is indexed. The number of searchable reviews is worked out here from the data itself."""
    from reviewpulse.nlp.semantic_search import MIN_TEXT_CHARS

    eligible = int((reviews["combined_text"].fillna("").str.strip().str.len() >= MIN_TEXT_CHARS).sum())
    h = real_client.get("/api/v1/health").json()
    assert h["status"] == "healthy" and h["total_reviews"] == len(reviews)
    assert h["retrieval"]["ready"] and h["retrieval"]["coverage_pct"] == 100.0
    assert h["retrieval"]["eligible_reviews"] == eligible and h["retrieval"]["indexed_reviews"] == eligible
    # Only a handful of near-empty reviews (a body such as "OK") are too short to search.
    assert 0 <= len(reviews) - eligible < len(reviews) * 0.01


@pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
def test_real_assistant_retrieval_is_scoped_to_the_category(real_client, category):
    data = real_client.post("/api/v1/assistant/chat", json={"question": "stopped working after a few weeks", "category": category, "top_k": 5}).json()
    assert data["retrieval"]["coverage_pct"] == 100.0
    assert data["evidence_count"] == len(data["evidence_reviews"]) > 0
    assert {e["category"] for e in data["evidence_reviews"]} == {category}
    # The fixture blanks the Gemini key, so this must be an honest retrieval-only reply and never a pretend LLM answer.
    assert data["llm_used"] is False and data["mode"] == "retrieval_only" and data["status"] == "setup_required"


RELEVANT_QUESTIONS = [
    "phone case does not fit my phone",
    "dog refused to eat the food",
    "battery drains quickly",
    "the zipper broke",
    "shipping was late and the box was damaged",
]
UNRELATED_QUESTIONS = [
    "What is the capital of Australia?",
    "Who won the 1998 football world cup?",
    "Explain quantum chromodynamics",
    "What is the weather in Paris tomorrow?",
    "Write a poem about the sea",
]


def test_default_similarity_threshold_keeps_real_matches(real_client, real_settings):
    assert real_settings.retrieval_min_score == pytest.approx(0.40)
    for question in RELEVANT_QUESTIONS:
        data = real_client.post("/api/v1/assistant/chat", json={"question": question, "top_k": 4}).json()
        assert data["evidence_count"] >= 3, f"{question!r} found only {data['evidence_count']} reviews"
        assert data["evidence_reviews"][0]["score"] >= 0.5, question
        assert all(e["score"] >= real_settings.retrieval_min_score for e in data["evidence_reviews"])


def test_default_similarity_threshold_returns_no_evidence_for_unrelated_questions(real_client):
    """Weak look-alike reviews ("Was a Christmas gift") must not be shown as evidence for a question the reviews cannot answer."""
    for question in UNRELATED_QUESTIONS:
        data = real_client.post("/api/v1/assistant/chat", json={"question": question, "top_k": 4}).json()
        assert data["evidence_count"] == 0 and data["evidence_reviews"] == [], f"{question!r} returned {data['evidence_count']} reviews"
        assert data["numerical_facts"]["total_reviews"] > 0  # the database numbers are still returned


def test_each_requested_real_model_is_the_one_that_answers(real_client):
    """Through the real app and the real saved models: asking for a model returns that model's own answer."""
    expected = {
        "logistic_regression": "TF-IDF + LogisticRegression",
        "linear_svc": "TF-IDF + LinearSVC",
        "distilbert": "DistilBERT (distilbert-base-uncased, fine-tuned)",
    }
    available = real_client.get("/api/v1/models/metrics").json()["available_models"]
    assert set(available) == set(expected), "all three models should be trained and evaluated"
    for model_type, label in expected.items():
        for text, wanted in [("It broke after one day, total waste of money.", "negative"), ("Works perfectly, arrived early, would buy again.", "positive")]:
            data = real_client.post("/api/v1/predict", json={"text": text, "model_type": model_type}).json()
            assert data["model_type"] == model_type and data["model_used"] == label
            assert data["predicted_sentiment"] == wanted, f"{model_type} on {text!r}"
    # The old silent-fallback bug: a misspelled field must be an error, not the default model.
    resp = real_client.post("/api/v1/predict", json={"text": "great product", "model": "distilbert"})
    assert resp.status_code == 422


# ---------- topics: every assignment must map to the right real review ----------
@pytest.fixture(scope="module")
def topic_data(real_settings, reviews):
    import joblib

    d = real_settings.review_topics_parquet.parent
    return {
        "dir": d,
        "tags": pd.read_parquet(real_settings.review_topics_parquet),
        "labels": json.loads(real_settings.topic_labels_json.read_text(encoding="utf-8")),
        "vectorizer": joblib.load(d / "nmf_vectorizer.joblib"),
        "nmf": joblib.load(d / "nmf_model.joblib"),
    }


def test_every_saved_topic_equals_the_topic_recomputed_from_that_reviews_own_text(topic_data, reviews):
    """Joins by review_id, so a shifted or shuffled row order would show up as mismatches."""
    import numpy as np

    tagged = topic_data["tags"].merge(reviews[["review_id", "combined_text"]], on="review_id", how="left")
    assert tagged["combined_text"].notna().all(), "a tag points at a review that does not exist"
    weights = topic_data["nmf"].transform(topic_data["vectorizer"].transform(tagged["combined_text"].fillna("")))
    recomputed = weights.argmax(axis=1)
    mismatched = int((recomputed != tagged["topic_id"].to_numpy()).sum())
    assert mismatched == 0, f"{mismatched} of {len(tagged)} saved topics differ from the topic recomputed from the review text"
    share = weights.max(axis=1) / weights.sum(axis=1)
    assert np.allclose(share, tagged["topic_score"].to_numpy(), atol=1e-3)


def test_reviews_without_a_tag_really_have_no_topic_words(topic_data, reviews):
    untagged = reviews[~reviews["review_id"].isin(topic_data["tags"]["review_id"])]
    matrix = topic_data["vectorizer"].transform(untagged["combined_text"].fillna(""))
    assert matrix.sum(axis=1).A1.max() == 0, "a review with known topic words was left untagged"
    assert len(untagged) + len(topic_data["tags"]) == len(reviews)


def test_topic_sizes_in_the_saved_file_equal_the_counts_in_the_tags(topic_data):
    counts = topic_data["tags"]["topic_id"].value_counts()
    total = len(topic_data["tags"])
    for t in topic_data["labels"]["topics"]:
        assert t["prevalence_count"] == counts.get(t["topic_id"], 0)
        assert t["prevalence_pct"] == pytest.approx(t["prevalence_count"] / total * 100, abs=0.01)
    assert sum(t["prevalence_count"] for t in topic_data["labels"]["topics"]) == total


@pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
def test_category_topic_counts_use_only_that_categorys_reviews(real_client, topic_data, reviews, category):
    tagged = topic_data["tags"].merge(reviews[["review_id", "category"]], on="review_id")
    expected = tagged[tagged["category"] == category]["topic_id"].value_counts()
    body = real_client.get("/api/v1/topics", params={"category": category}).json()
    assert body["category_filter"] == category and body["review_tags_available"] is True
    assert body["reviews_considered"] == len(tagged[tagged["category"] == category])
    for item in body["topics"]:
        assert item["prevalence_count"] == expected.get(item["topic_id"], 0), (category, item["label"])
    assert sum(i["prevalence_count"] for i in body["topics"]) == body["reviews_considered"]


@pytest.mark.parametrize("category", ["Pet_Supplies", "Cell_Phones_and_Accessories", "Beauty_and_Personal_Care"])
def test_reviews_filtered_by_topic_and_category_have_that_tag_and_that_category(real_client, topic_data, reviews, category):
    tags = topic_data["tags"].set_index("review_id")["topic_id"]
    by_id = reviews.set_index("review_id")
    busiest = int(tags[by_id.loc[tags.index, "category"] == category].value_counts().idxmax())
    body = real_client.get("/api/v1/reviews", params={"category": category, "topic_id": busiest, "limit": 50}).json()
    expected_total = int(((by_id.loc[tags.index, "category"] == category) & (tags == busiest)).sum())
    assert body["total_records"] == expected_total > 0
    for r in body["reviews"]:
        assert r["category"] == category and r["topic_id"] == busiest and tags[r["review_id"]] == busiest
        assert r["text"][:40].strip() and r["topic_label"]


def test_topic_names_are_backed_by_the_category_mix_of_their_reviews(topic_data, reviews):
    """A name is a claim about what the reviews are about. Check it against which categories the reviews come from."""
    tagged = topic_data["tags"].merge(reviews[["review_id", "category"]], on="review_id")
    mix = {t["label"]: tagged[tagged["topic_id"] == t["topic_id"]]["category"].value_counts(normalize=True) for t in topic_data["labels"]["topics"]}
    assert mix["Dogs, cats and pet supplies"].get("Pet_Supplies", 0) >= 0.70
    assert mix["Hair and skin care"].get("Beauty_and_Personal_Care", 0) >= 0.60
    assert mix["Phone cases"].get("Cell_Phones_and_Accessories", 0) >= 0.50
    assert mix["Screen protectors"].get("Cell_Phones_and_Accessories", 0) >= 0.50
    assert mix["Phones, chargers and batteries"].get("Cell_Phones_and_Accessories", 0) >= 0.50
    assert mix["Games, toys and fun for kids"].get("Toys_and_Games", 0) >= 0.40


def test_the_biggest_topic_is_labelled_as_mixed_if_it_is_spread_over_many_categories(topic_data, reviews):
    """No hiding a catch-all behind a specific-sounding name."""
    tagged = topic_data["tags"].merge(reviews[["review_id", "category"]], on="review_id")
    biggest = max(topic_data["labels"]["topics"], key=lambda t: t["prevalence_count"])
    top_category_share = tagged[tagged["topic_id"] == biggest["topic_id"]]["category"].value_counts(normalize=True).iloc[0]
    if top_category_share < 0.30:
        assert "mixed" in biggest["label"].lower(), f"{biggest['label']!r} covers {biggest['prevalence_pct']:.0f}% of reviews from many categories"
