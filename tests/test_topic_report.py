"""The topic report is generated from saved files: its numbers must be the saved numbers."""

import json

import pytest

from reviewpulse.nlp.topic_model import train_nmf_topic_model
from reviewpulse.nlp.topic_report import build_report


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory, fixture_df):
    out = tmp_path_factory.mktemp("report_topics")
    train_nmf_topic_model(output_dir=out, n_topics=4, df=fixture_df)
    return out


@pytest.fixture(scope="module")
def experiments_file(tmp_path_factory):
    def stat(mean, low, high):
        return {"mean": mean, "min": low, "max": high}

    config = {"topics": 4, "biggest_topic_pct": stat(30.0, 28.0, 32.0), "npmi": stat(0.12, 0.11, 0.13), "distinct_words": stat(0.7, 0.7, 0.7), "weak_pct": stat(40.0, 39.0, 41.0)}
    path = tmp_path_factory.mktemp("exp") / "topic_experiments.json"
    path.write_text(json.dumps({"reviews": 240, "seeds": [0, 1, 2], "configs": {"first version": config, "new setup (adopted)": config}}), encoding="utf-8")
    return path


def test_the_report_states_the_saved_sizes_and_names(trained_dir, fixture_settings, experiments_file):
    info = json.loads((trained_dir / "topic_labels.json").read_text(encoding="utf-8"))
    text = build_report(trained_dir, fixture_settings.reviews_parquet, experiments_file)
    assert f"**{info['tagged_reviews']:,} tagged**" in text
    for topic in info["topics"]:
        assert topic["label"].replace("|", "\\|") in text
        assert f"{topic['prevalence_count']:,}" in text
        assert f"{topic['prevalence_pct']:.1f}%" in text
    for heading in ["## 1. What was done", "## 2. Why these settings", "## 3. The topics", "## 4. The largest topic", "## 5. What the reviews", "## 6. Limitations"]:
        assert heading in text


def test_the_comparison_table_uses_the_experiment_numbers(trained_dir, fixture_settings, experiments_file):
    text = build_report(trained_dir, fixture_settings.reviews_parquet, experiments_file)
    assert "| first version | 4 | 30.0% (28.0-32.0) | +0.120 (+0.110 to +0.130) | 0.70 | 40.0% |" in text
    assert "three seeds in the table gave 28.0% to 32.0%" in text


def test_example_reviews_in_the_report_are_real_reviews(trained_dir, fixture_settings, fixture_df, experiments_file):
    text = build_report(trained_dir, fixture_settings.reviews_parquet, experiments_file)
    quoted = [line.split('"', 1)[1].rsplit('"', 1)[0] for line in text.splitlines() if line.startswith("- strongest") or line.startswith("- random")]
    assert quoted
    haystack = " ".join(" ".join(t.split()) for t in fixture_df["combined_text"]).replace("\\|", "|")
    for excerpt in quoted:
        assert excerpt.rstrip(".").replace("...", "").strip()[:60] in haystack, f"not a real review excerpt: {excerpt!r}"


def test_a_missing_experiment_file_is_explained_not_an_error(trained_dir, fixture_settings, tmp_path):
    text = build_report(trained_dir, fixture_settings.reviews_parquet, tmp_path / "does_not_exist.json")
    assert "topic_experiments" in text and "## 3. The topics" in text
