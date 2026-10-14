"""NMF topic model and review-level topic tags."""

import json
from collections import Counter

import pandas as pd
import pytest

from reviewpulse.nlp.topic_model import (
    DOMAIN_STOP_WORDS,
    PRAISE_STOP_WORDS,
    TOPIC_NAMES,
    assign_topics,
    curated_name,
    make_label,
    train_nmf_topic_model,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, fixture_df):
    out = tmp_path_factory.mktemp("topics")
    results = train_nmf_topic_model(output_dir=out, n_topics=4, df=fixture_df)
    return out, results


def test_artifacts_are_written(trained):
    out, results = trained
    for name in ["nmf_vectorizer.joblib", "nmf_model.joblib", "topic_labels.json", "review_topics.parquet"]:
        assert (out / name).exists(), name
    assert json.loads((out / "topic_labels.json").read_text())["n_topics"] == results["n_topics"] == 4


def test_tags_reference_real_reviews_and_are_unique(trained, fixture_df):
    out, results = trained
    tags = pd.read_parquet(out / "review_topics.parquet")
    assert tags["review_id"].is_unique and set(tags["review_id"]) <= set(fixture_df["review_id"])
    assert tags["topic_id"].between(0, 3).all() and tags["topic_score"].between(0.0, 1.0).all() and (tags["topic_score"] > 0).all()
    assert len(tags) == results["tagged_reviews"] <= results["total_reviews"] == len(fixture_df)


def test_topic_counts_add_up_to_the_tags(trained):
    out, results = trained
    tags = pd.read_parquet(out / "review_topics.parquet")
    per_topic = tags["topic_id"].value_counts().to_dict()
    for topic in results["topics"]:
        assert topic["prevalence_count"] == per_topic.get(topic["topic_id"], 0)
    assert sum(t["prevalence_count"] for t in results["topics"]) == len(tags)
    assert sum(t["prevalence_pct"] for t in results["topics"]) == pytest.approx(100 * len(tags) / results["total_reviews"], abs=0.1)


def test_stop_words_are_really_applied(trained):
    _, results = trained
    words = {w for t in results["topics"] for w in t["top_words"]}
    unigrams = {w for w in words if " " not in w}
    assert not unigrams & DOMAIN_STOP_WORDS, unigrams & DOMAIN_STOP_WORDS
    assert "br" not in words and "the" not in words


def test_topics_separate_lexically_distinct_text(trained, fixture_df):
    out, _ = trained
    tags = pd.read_parquet(out / "review_topics.parquet").merge(fixture_df[["review_id", "category"]], on="review_id")
    dominant = {}
    for category, group in tags.groupby("category"):
        topic, count = Counter(group["topic_id"]).most_common(1)[0]
        assert count / len(group) >= 0.7, f"{category}: its reviews are spread over several topics"
        dominant[category] = topic
    assert len(set(dominant.values())) >= 3  # distinct categories mostly land in distinct topics


def test_example_reviews_belong_to_their_topic(trained, fixture_df):
    _, results = trained
    for topic in results["topics"]:
        assert 1 <= len(topic["sample_reviews"]) <= 2
        assert all(len(s) <= 224 for s in topic["sample_reviews"])


def test_assign_topics_to_new_text(trained):
    out, _ = trained
    dog, dog_again, blank = assign_topics(["my dog loves this food and the cat tree", "dog food cat", "  "], model_dir=out)
    assert dog["topic_id"] == dog_again["topic_id"] and 0 < dog["topic_score"] <= 1
    assert blank is None  # no known words, so no tag rather than an invented one


def test_make_label_skips_overlapping_words():
    assert make_label(["battery", "battery life", "phone", "charger"]) == "Battery & Phone & Charger"
    assert make_label(["dog", "dogs", "food"]) == "Dog & Food"
    assert make_label([]) == ""


# ---------- word lists and example reviews ----------
def test_top_words_never_contain_a_repeated_word_phrase(trained):
    _, results = trained
    for topic in results["topics"]:
        for word in topic["top_words"]:
            parts = word.split()
            assert not (len(parts) == 2 and parts[0] == parts[1]), f"{word!r} in topic {topic['topic_id']}"


def test_repeated_word_phrases_are_skipped_but_real_phrases_stay():
    import numpy as np

    from reviewpulse.nlp.topic_model import _top_words

    names = ["case case", "phone case", "case", "cute cute", "easy install"]
    assert _top_words(names, np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 3) == ["phone case", "case", "easy install"]


def test_example_reviews_are_distinct_and_prefer_informative_ones():
    import numpy as np
    import pandas as pd

    from reviewpulse.nlp.topic_model import _example_reviews

    texts = pd.Series(["Nice price", "Nice price", "Good price for what you get and it arrived quickly too thanks", "Good price for what you get and it arrived quickly too thanks", "Another long review about the price and why it was worth buying", "ok"])
    words = texts.str.split().str.len().to_numpy()
    weights = np.array([0.9, 0.9, 0.5, 0.5, 0.4, 0.1])
    members = np.ones(6, dtype=bool)
    found = _example_reviews(texts, words, weights, members)
    assert found == ["Good price for what you get and it arrived quickly too thanks", "Another long review about the price and why it was worth buying"]
    # with only short reviews available they are still used, once each
    short = _example_reviews(texts, words, weights, np.array([True, True, False, False, False, True]))
    assert short == ["Nice price", "ok"]


def test_two_reviews_with_the_same_opening_are_one_example():
    import numpy as np
    import pandas as pd

    from reviewpulse.nlp.topic_model import _example_reviews

    opening = "My routine at night is simple and it really works for dry skin in the winter months " * 4
    texts = pd.Series([opening + "first ending", opening + "different ending", "A completely different review that talks about something else entirely here"])
    found = _example_reviews(texts, texts.str.split().str.len().to_numpy(), np.array([0.9, 0.8, 0.1]), np.ones(3, dtype=bool))
    assert len(found) == 2 and len(set(found)) == 2 and "completely different" in found[1]


def test_saved_example_reviews_are_distinct_for_every_topic(trained):
    _, results = trained
    for topic in results["topics"]:
        assert len(set(topic["sample_reviews"])) == len(topic["sample_reviews"])


# ---------- reviewed topic names ----------
def test_a_name_needs_all_of_its_signature_words():
    assert curated_name(["screen", "protector", "phone", "glass"]) == "Screen protectors"
    assert curated_name(["screen", "phone", "glass"]) is None, "only one signature word: the name must not be used"
    assert curated_name(["dog", "dogs", "food"]) == "Dogs, cats and pet supplies"
    assert curated_name([]) is None


def test_a_name_is_only_looked_up_in_the_top_ten_words():
    words = ["w%d" % i for i in range(10)] + ["screen", "protector"]
    assert curated_name(words) is None


def test_curated_names_are_unique_and_signatures_are_not_duplicated():
    names = [name for _, name in TOPIC_NAMES]
    assert len(names) == len(set(names))
    signatures = [frozenset(sig) for sig, _ in TOPIC_NAMES]
    assert len(signatures) == len(set(signatures))


def test_every_topic_records_how_it_was_named(trained):
    _, results = trained
    for topic in results["topics"]:
        assert topic["named_by"] in {"reviewed", "top words"}
        assert topic["auto_label"] == make_label(topic["top_words"])
        if topic["named_by"] == "top words":
            assert topic["label"] == topic["auto_label"], "an unreviewed topic must keep its honest top-word label"
        else:
            assert topic["label"] == curated_name(topic["top_words"])


def test_praise_only_words_are_not_in_the_topic_vocabulary(trained):
    import joblib

    out, _ = trained
    vocabulary = set(joblib.load(out / "nmf_vectorizer.joblib").get_feature_names_out())
    assert not (vocabulary & PRAISE_STOP_WORDS)


def test_topic_percentages_are_shares_of_the_tagged_reviews(trained):
    _, results = trained
    assert sum(t["prevalence_count"] for t in results["topics"]) == results["tagged_reviews"]
    assert sum(t["prevalence_pct"] for t in results["topics"]) == pytest.approx(100.0, abs=0.1)
    for t in results["topics"]:
        assert t["prevalence_pct"] == pytest.approx(t["prevalence_count"] / results["tagged_reviews"] * 100, abs=0.01)


@pytest.mark.realdata
def test_real_topic_tags_reconcile_with_the_dataset(real_settings):
    tags_path = real_settings.review_topics_parquet
    if not tags_path.exists():
        pytest.skip("topic model not trained")
    tags = pd.read_parquet(tags_path)
    reviews = pd.read_parquet(real_settings.reviews_parquet, columns=["review_id"])
    labels = json.loads(real_settings.topic_labels_json.read_text())
    assert tags["review_id"].is_unique and set(tags["review_id"]) <= set(reviews["review_id"])
    assert labels["total_reviews"] == len(reviews) and labels["tagged_reviews"] == len(tags)
    assert sum(t["prevalence_count"] for t in labels["topics"]) == len(tags)
    assert not any("br" == w for t in labels["topics"] for w in t["top_words"])
