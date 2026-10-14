"""Sentiment models: metrics, data split hygiene, saved-model loading and inference."""

import json
import sys
import threading
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from reviewpulse.modeling.dataset import load_and_split_data
from reviewpulse.modeling.evaluation import (
    MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS,
    compute_metrics,
    load_metrics,
    per_category_metrics,
    record_model_result,
)
from reviewpulse.heavy_imports import HEAVY_IMPORT_LOCK
from reviewpulse.modeling import inference
from reviewpulse.modeling.inference import ModelUnavailable, SentimentPredictors
from reviewpulse.modeling.train_classical import compute_metrics as compute_metrics_reexport


# ---------- metrics ----------
def test_tfidf_vectorizer_fit_and_transform(sample_processed_df):
    vec = TfidfVectorizer(max_features=100, stop_words="english")
    X = vec.fit_transform(sample_processed_df["combined_text"])
    assert X.shape[0] == len(sample_processed_df) and X.shape[1] > 0


def test_logistic_regression_fit_and_predict(sample_processed_df):
    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(sample_processed_df["combined_text"])
    clf = LogisticRegression(class_weight="balanced", max_iter=200).fit(X, sample_processed_df["sentiment_encoded"].values)
    assert set(clf.predict(X)) <= {0, 1, 2}


def test_compute_metrics_matches_hand_calculation():
    y_true = np.array([0, 1, 2, 2, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert m["negative_f1"] == 1.0 and m["neutral_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["positive_precision"] == 1.0 and m["positive_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 1, 2]]
    assert m["macro_f1"] == pytest.approx(np.mean([m["negative_f1"], m["neutral_f1"], m["positive_f1"]]), abs=1e-3)
    assert m["macro_precision"] == pytest.approx(np.mean([m["negative_precision"], m["neutral_precision"], m["positive_precision"]]), abs=1e-3)
    assert compute_metrics_reexport is compute_metrics


def test_compute_metrics_survives_a_missing_class():
    m = compute_metrics(np.array([0, 0, 2, 2]), np.array([0, 2, 2, 2]))  # no neutral reviews at all
    assert len(m["confusion_matrix"]) == 3 and m["neutral_f1"] == 0.0 and m["accuracy"] == 0.75


def test_per_category_metrics_flag_unreliable_small_classes():
    n = MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS
    big = pd.DataFrame({"category": ["Big"] * (3 * n), "sentiment_encoded": [0, 1, 2] * n})
    small = pd.DataFrame({"category": ["Small"] * 9, "sentiment_encoded": [0, 0, 0, 0, 0, 0, 0, 1, 2]})
    test_df = pd.concat([big, small], ignore_index=True)
    result = per_category_metrics(test_df, test_df["sentiment_encoded"].to_numpy())
    assert result["Big"]["reliable"] is True and "note" not in result["Big"] and result["Big"]["accuracy"] == 1.0
    assert result["Small"]["reliable"] is False and "fewer than" in result["Small"]["note"]
    assert result["Small"]["class_support"] == {"negative": 7, "neutral": 1, "positive": 1} and result["Small"]["test_samples"] == 9


def test_record_model_result_keeps_other_models(tmp_path):
    path = tmp_path / "metrics.json"
    record_model_result(path, "logistic_regression", {"test": {"accuracy": 0.8}}, per_category={"A": {"accuracy": 0.8}})
    record_model_result(path, "distilbert", {"test": {"accuracy": 0.9}})
    data = load_metrics(path)
    assert set(data["models"]) == {"logistic_regression", "distilbert"} and data["per_category"]["logistic_regression"]["A"]["accuracy"] == 0.8
    assert "updated_at" in data


# ---------- data split ----------
def reviewers_frame(n_users=60, per_user=10, seed=0):
    """Reviews written by repeat reviewers whose rating habits differ (some always happy, some often unhappy)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        positive_rate = rng.choice([0.95, 0.8, 0.5])
        for i in range(per_user):
            rating = 5.0 if rng.random() < positive_rate else float(rng.choice([1, 2, 3]))
            sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
            rows.append(
                {
                    "review_id": f"r{u}-{i}",
                    "user_id": f"user{u}",
                    "category": ["A", "B"][u % 2],
                    "rating": rating,
                    "sentiment": sentiment,
                    "sentiment_encoded": {"negative": 0, "neutral": 1, "positive": 2}[sentiment],
                    "combined_text": f"review text number {u} {i} about thing {rng.integers(10**6)}",
                }
            )
    return pd.DataFrame(rows)


def test_no_reviewer_appears_in_two_splits(tmp_path):
    path = tmp_path / "r.parquet"
    reviewers_frame().to_parquet(path)
    train, val, test, info = load_and_split_data(path)
    users = [set(d["user_id"]) for d in (train, val, test)]
    assert not (users[0] & users[1]) and not (users[0] & users[2]) and not (users[1] & users[2])
    assert info["reviewers_shared_between_splits"] == 0
    assert info["train_reviewers"] + info["val_reviewers"] + info["test_reviewers"] == 60
    assert "user_id" in info["split_strategy"]


def test_grouped_split_keeps_sizes_and_class_balance_reasonable(tmp_path):
    path = tmp_path / "r.parquet"
    df = reviewers_frame(n_users=200, per_user=8, seed=1)
    df.to_parquet(path)
    train, val, test, info = load_and_split_data(path)
    assert len(train) + len(val) + len(test) == info["total_deduplicated_samples"] == len(df)
    for part, target in ((train, 0.714), (val, 0.143), (test, 0.143)):
        assert abs(len(part) / len(df) - target) < 0.04
    overall = df["sentiment"].value_counts(normalize=True)
    for part in (val, test):
        assert (part["sentiment"].value_counts(normalize=True).reindex(overall.index).fillna(0) - overall).abs().max() < 0.07


def test_a_plain_random_split_would_have_leaked_reviewers():
    """Documents why the grouped split exists: a random split puts the same reviewer on both sides."""
    from sklearn.model_selection import train_test_split

    df = reviewers_frame(n_users=60, per_user=10)
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    assert test["user_id"].isin(set(train["user_id"])).mean() > 0.9


def test_split_is_leakage_free_stratified_and_deterministic(tmp_path, fixture_settings):
    df = pd.read_parquet(fixture_settings.reviews_parquet)
    df = pd.concat([df, df.iloc[:20].assign(review_id=lambda d: d["review_id"] + "-dup")], ignore_index=True)  # 20 exact-text duplicates
    path = tmp_path / "with_dups.parquet"
    df.to_parquet(path)
    train, val, test, info = load_and_split_data(path)
    assert info["total_deduplicated_samples"] == len(df) - 20
    assert not set(train["combined_text"]) & set(test["combined_text"])
    assert not set(train["combined_text"]) & set(val["combined_text"]) and not set(val["combined_text"]) & set(test["combined_text"])
    assert len(train) + len(val) + len(test) == info["total_deduplicated_samples"]
    assert abs(len(test) / info["total_deduplicated_samples"] - 0.143) < 0.03
    _, _, test_again, info_again = load_and_split_data(path)
    assert info_again["test_ids_md5"] == info["test_ids_md5"] and list(test_again["review_id"]) == list(test["review_id"])


# ---------- inference ----------
@pytest.fixture(scope="module")
def predictors(fixture_settings):
    return SentimentPredictors(fixture_settings.models_dir)


def test_saved_classical_models_load_and_predict(predictors):
    assert predictors.available_models() == ["logistic_regression", "linear_svc"] and predictors.classical_ready
    lr = predictors.predict("The battery drains quickly and it stopped working", "logistic_regression")
    assert lr["predicted_sentiment"] == "negative" and lr["sentiment_encoded"] == 0
    assert set(lr["probabilities"]) == {"negative", "neutral", "positive"}
    assert sum(lr["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)
    assert lr["probabilities"]["negative"] == max(lr["probabilities"].values())
    svc = predictors.predict("Easy to install and fits my truck perfectly", "linear_svc")
    assert svc["predicted_sentiment"] == "positive" and svc["probabilities"] is None


def test_unknown_model_type_is_an_error_not_a_fallback(predictors):
    with pytest.raises(ValueError):
        predictors.predict("text", "random_forest")


def test_missing_distilbert_artifacts_raise_model_unavailable(predictors):
    assert "distilbert" not in predictors.available_models() and not predictors.distilbert_artifacts_present()
    with pytest.raises(ModelUnavailable, match="not been trained"):
        predictors.predict("text", "distilbert")


def _other_thread_can_take_the_import_lock() -> bool:
    """True if a different thread could take the heavy-import lock right now (i.e. nobody is importing)."""
    result: list = []

    def attempt():
        got = HEAVY_IMPORT_LOCK.acquire(blocking=False)
        result.append(got)
        if got:
            HEAVY_IMPORT_LOCK.release()

    t = threading.Thread(target=attempt)
    t.start()
    t.join()
    return result[0]


@pytest.fixture
def fake_distilbert(tmp_path, monkeypatch):
    """A predictor with DistilBERT-looking files and stand-in `torch` / `transformers` modules (no real model)."""
    models = tmp_path / "models"
    (models / inference.DISTILBERT_DIR_NAME).mkdir(parents=True)
    (models / inference.DISTILBERT_DIR_NAME / "config.json").write_text("{}")
    (models / inference.DISTILBERT_DIR_NAME / "model.safetensors").write_bytes(b"x")
    state = types.SimpleNamespace(calls=0, fail=True, lock_free_during_load=None)

    class Model:
        def eval(self):
            return self

    def from_pretrained(path):
        state.calls += 1
        state.lock_free_during_load = _other_thread_can_take_the_import_lock()
        if state.fail:
            raise RuntimeError("cannot import name 'AutoModelForSequenceClassification'")
        return Model()

    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
    stub.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return SentimentPredictors(models), state


def test_distilbert_loads_while_holding_the_shared_import_lock(fake_distilbert):
    predictors, state = fake_distilbert
    state.fail = False
    predictors._load_distilbert()
    assert state.lock_free_during_load is False, "another thread could import torch/transformers during the DistilBERT load"
    assert _other_thread_can_take_the_import_lock(), "the lock must be released after loading"


def test_distilbert_load_waits_for_another_thread_that_is_importing(fake_distilbert):
    """The race we hit in the server: a request must wait while the warm-up thread is importing the big libraries."""
    predictors, state = fake_distilbert
    state.fail = False
    done = threading.Event()
    with HEAVY_IMPORT_LOCK:
        threading.Thread(target=lambda: (predictors._load_distilbert(), done.set()), daemon=True).start()
        assert not done.wait(0.5), "DistilBERT loaded while another thread held the import lock"
        assert state.calls == 0
    assert done.wait(10) and state.calls >= 1


def test_a_failed_distilbert_load_is_retried_later_not_cached_forever(fake_distilbert, monkeypatch):
    predictors, state = fake_distilbert
    with pytest.raises(ModelUnavailable, match="could not be loaded"):
        predictors._load_distilbert()
    first_attempts = state.calls
    with pytest.raises(ModelUnavailable):  # inside the cool-down the stored error is reused: no repeated slow imports
        predictors._load_distilbert()
    assert state.calls == first_attempts
    monkeypatch.setattr(inference, "DISTILBERT_RETRY_SECONDS", 0)
    state.fail = False  # the problem went away
    assert predictors._load_distilbert()["model"] is not None, "after the cool-down a new attempt must be made"
    assert state.calls > first_attempts


def test_warm_up_does_nothing_without_artifacts_and_loads_with_them(predictors, fake_distilbert):
    predictors.warm_up()  # no DistilBERT files: must not raise
    warm, state = fake_distilbert
    state.fail = False
    warm.warm_up()
    assert state.calls >= 1 and warm._distilbert is not None


def test_missing_models_directory_is_handled(tmp_path):
    p = SentimentPredictors(tmp_path / "empty")
    assert p.available_models() == [] and not p.classical_ready
    with pytest.raises(ModelUnavailable):
        p.predict("text", "logistic_regression")


def test_corrupt_model_file_disables_only_that_model(tmp_path, fixture_settings):
    models = tmp_path / "models"
    models.mkdir()
    for name in ["tfidf_vectorizer.joblib", "logistic_regression.joblib"]:
        (models / name).write_bytes((fixture_settings.models_dir / name).read_bytes())
    (models / "linear_svc.joblib").write_bytes(b"not a joblib file")
    p = SentimentPredictors(models)
    assert p.available_models() == ["logistic_regression"]
    with pytest.raises(ModelUnavailable):
        p.predict("text", "linear_svc")


# ---------- the real saved models ----------
@pytest.fixture(scope="module")
def real_metrics(real_settings):
    return json.loads(real_settings.metrics_json.read_text(encoding="utf-8"))


@pytest.mark.realdata
@pytest.mark.parametrize("key,filename", [("logistic_regression", "logistic_regression.joblib"), ("linear_svc", "linear_svc.joblib")])
def test_reported_metrics_are_reproduced_by_the_saved_models(real_settings, real_metrics, key, filename):
    """Reload the saved vectorizer and model, re-score the held-out test split, and compare with metrics.json."""
    _, _, test_df, split = load_and_split_data(real_settings.reviews_parquet)
    assert split["test_ids_md5"] == real_metrics["split_info"]["test_ids_md5"], "metrics.json was scored on a different test set"
    vectorizer = joblib.load(real_settings.models_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(real_settings.models_dir / filename)
    pred = model.predict(vectorizer.transform(test_df["combined_text"]))
    fresh = compute_metrics(test_df["sentiment_encoded"].to_numpy(), pred)
    saved = real_metrics["models"][key]["test"]
    for field in ["accuracy", "macro_f1", "weighted_f1", "negative_f1", "neutral_f1", "positive_f1", "confusion_matrix"]:
        assert fresh[field] == saved[field], f"{key}.{field}: recomputed {fresh[field]} vs saved {saved[field]}"
    assert sum(map(sum, saved["confusion_matrix"])) == len(test_df) == real_metrics["split_info"]["test_samples"]


@pytest.mark.realdata
def test_real_models_classify_obvious_examples(real_settings):
    p = SentimentPredictors(real_settings.models_dir)
    assert {"logistic_regression", "linear_svc"} <= set(p.available_models())
    for model in ["logistic_regression", "linear_svc"]:
        assert p.predict("Terrible. It broke after two days and customer service refused a refund. Waste of money.", model)["predicted_sentiment"] == "negative"
        assert p.predict("Excellent quality, works perfectly and arrived fast. Highly recommend, five stars!", model)["predicted_sentiment"] == "positive"


@pytest.mark.realdata
def test_per_category_metrics_cover_every_category_and_sum_to_the_test_set(real_settings, real_metrics):
    from reviewpulse.config import OFFICIAL_CATEGORIES

    for model, cats in real_metrics["per_category"].items():
        assert set(cats) == set(OFFICIAL_CATEGORIES), model
        assert sum(c["test_samples"] for c in cats.values()) == real_metrics["split_info"]["test_samples"]


@pytest.mark.realdata
def test_distilbert_result_is_only_reported_if_artifacts_and_full_evaluation_exist(real_settings, real_metrics):
    d = real_metrics["models"].get("distilbert")
    if d is None:
        pytest.skip("DistilBERT has not been trained; metrics.json correctly contains no DistilBERT entry")
    assert (real_settings.models_dir / "distilbert_sentiment" / "config.json").exists()
    assert d["evaluated_on"]["test_samples"] == real_metrics["split_info"]["test_samples"]
    assert sum(map(sum, d["test"]["confusion_matrix"])) == d["evaluated_on"]["test_samples"]
    assert d["training"]["epochs"] >= 1 and d["training"]["train_samples"] > 0
