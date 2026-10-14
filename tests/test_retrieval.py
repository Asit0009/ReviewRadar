"""Semantic retrieval: coverage, vector-to-review mapping, filtering and relevance."""

import json

import faiss
import numpy as np
import pandas as pd
import pytest

from data_factory import CATEGORIES, TEXTS, fake_encoder
from reviewpulse.nlp.semantic_search import (
    INDEX_FILE,
    INFO_FILE,
    META_FILE,
    ReviewSemanticRetriever,
    build_faiss_index,
    ids_fingerprint,
    memory_cap,
    verify_index,
)


def test_faiss_index_flat_ip_cosine():
    """FAISS IndexFlatIP gives exact cosine similarity on normalized vectors."""
    index = faiss.IndexFlatIP(4)
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    v3 = np.array([[0.7071, 0.7071, 0.0, 0.0]], dtype=np.float32)
    index.add(np.vstack([v1, v2, v3]))
    distances, indices = index.search(v1, 2)
    assert indices[0][0] == 0 and abs(distances[0][0] - 1.0) < 1e-4
    assert indices[0][1] == 2 and abs(distances[0][1] - 0.7071) < 1e-3


@pytest.fixture(scope="module")
def retriever(fixture_settings):
    return ReviewSemanticRetriever(fixture_settings.vector_store_dir, encoder=fake_encoder, min_score=0.25)


# ---------- coverage ----------
def test_default_build_indexes_every_eligible_review(fixture_settings, fixture_df):
    info = json.loads((fixture_settings.vector_store_dir / INFO_FILE).read_text())
    assert info["eligible_reviews"] == info["indexed_reviews"] == len(fixture_df)
    assert info["coverage_pct"] == 100.0 and info["limited_by"] is None and info["max_samples_requested"] is None
    assert {c: v["indexed"] for c, v in info["per_category"].items()} == fixture_df["category"].value_counts().to_dict()
    assert info["embedding_model"] == "fake-bow-128"


def test_index_and_metadata_have_the_same_size(fixture_settings, fixture_df):
    index = faiss.read_index(str(fixture_settings.vector_store_dir / INDEX_FILE))
    meta = pd.read_parquet(fixture_settings.vector_store_dir / META_FILE)
    assert index.ntotal == len(meta) == len(fixture_df)
    assert meta["review_id"].is_unique and meta["vector_id"].tolist() == list(range(len(meta)))


# ---------- mapping ----------
def test_verify_index_confirms_every_sampled_vector_belongs_to_its_review(fixture_settings):
    report = verify_index(fixture_settings.vector_store_dir, encoder=fake_encoder, sample_size=240)
    assert report["ok"], report["problems"]
    assert report["self_retrieval_checked"] == report["self_retrieval_correct"] == 240


def test_metadata_matches_the_source_data(fixture_settings, fixture_df):
    meta = pd.read_parquet(fixture_settings.vector_store_dir / META_FILE).set_index("review_id")
    src = fixture_df.set_index("review_id").loc[meta.index]
    for column in ["category", "rating", "sentiment", "text", "title", "helpful_vote", "verified_purchase"]:
        assert (meta[column].to_numpy() == src[column].to_numpy()).all(), column


def test_every_hit_is_the_review_its_id_claims(retriever, fixture_df):
    by_id = fixture_df.set_index("review_id")
    for query in ["battery drains quickly", "my dog loves this food", "brake pads squeal", "printer paper jams"]:
        hits = retriever.search(query, top_k=5, min_score=0.0)
        assert len(hits) == 5
        for hit in hits:
            source = by_id.loc[hit["review_id"]]
            assert (hit["category"], hit["rating"], hit["sentiment"], hit["full_text"]) == (source["category"], source["rating"], source["sentiment"], source["text"])


def test_the_exact_text_of_a_review_retrieves_that_review_first(retriever, fixture_df):
    for _, row in fixture_df.sample(15, random_state=3).iterrows():
        top = retriever.search(row["combined_text"], top_k=1, min_score=0.0)[0]
        assert top["review_id"] == row["review_id"] and top["score"] > 0.99


# ---------- filtering ----------
@pytest.mark.parametrize("category", CATEGORIES)
def test_category_filter_returns_only_that_category_and_still_fills_top_k(retriever, category):
    # "battery screen phone" matches Electronics best, so for other categories a post-filter over a small pool would starve.
    hits = retriever.search("battery screen phone stopped working", top_k=6, category=category, min_score=0.0)
    assert len(hits) == 6
    assert {h["category"] for h in hits} == {category}


def test_sentiment_filter(retriever):
    hits = retriever.search("great quality", top_k=8, sentiment="negative", min_score=0.0)
    assert hits and {h["sentiment"] for h in hits} == {"negative"}


def test_unknown_category_returns_nothing(retriever):
    assert retriever.search("battery", top_k=5, category="Nope", min_score=0.0) == []


# ---------- relevant vs irrelevant ----------
def test_relevant_query_finds_the_right_reviews(retriever):
    hits = retriever.search("my dog refused to eat this food", top_k=5)
    assert hits and hits[0]["category"] == "Pet_Supplies" and hits[0]["sentiment"] == "negative"
    assert hits[0]["score"] > 0.5
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_irrelevant_query_returns_no_evidence_above_the_threshold(retriever):
    assert retriever.search("quantum chromodynamics lagrangian symmetry breaking", top_k=5) == []  # default min_score 0.25
    loose = retriever.search("quantum chromodynamics lagrangian symmetry breaking", top_k=5, min_score=0.0)
    assert all(h["score"] < 0.25 for h in loose)  # they exist, but are weak, and are the ones the threshold removes


def test_relevant_scores_beat_irrelevant_scores(retriever):
    good = retriever.search("battery drains quickly", top_k=1, min_score=0.0)[0]["score"]
    bad = retriever.search("medieval castle architecture", top_k=1, min_score=0.0)[0]["score"]
    # A 3-word query only partly overlaps a 15-word review, so the absolute score is modest; the gap is what matters.
    assert good > 0.3 and bad < 0.1 and good > bad + 0.25


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ", top_k=3) == []


# ---------- caps are recorded honestly ----------
def test_max_samples_cap_is_reported_and_category_balanced(tmp_path, fixture_settings, fixture_df):
    info = build_faiss_index(fixture_settings.reviews_parquet, tmp_path, max_samples=48, encoder=fake_encoder)
    assert info["indexed_reviews"] == 48 and info["eligible_reviews"] == len(fixture_df)
    assert info["coverage_pct"] == 20.0 and info["limited_by"] == "max_samples" and info["max_samples_requested"] == 48
    assert {v["indexed"] for v in info["per_category"].values()} == {12}
    assert verify_index(tmp_path, encoder=fake_encoder)["ok"]


def test_memory_limit_caps_the_index_and_says_so(tmp_path, fixture_settings):
    budget_for_100_vectors = int(100 * 2048 * 4 * 2 / 0.5)
    info = build_faiss_index(fixture_settings.reviews_parquet, tmp_path, encoder=fake_encoder, available_memory_bytes=budget_for_100_vectors)
    assert info["indexed_reviews"] == 100 and info["limited_by"] == "memory" and info["coverage_pct"] < 100


def test_memory_cap_function():
    assert memory_cap(384, None, 0.5) is None
    assert memory_cap(384, 1_000_000_000, 0.5) == 500_000_000 // (384 * 8)
    assert memory_cap(384, 1_000, 0.5) == 0


def test_duplicate_review_ids_are_refused(tmp_path, fixture_settings):
    df = pd.read_parquet(fixture_settings.reviews_parquet)
    df.loc[1, "review_id"] = df.loc[0, "review_id"]
    path = tmp_path / "dup.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        build_faiss_index(path, tmp_path / "out", encoder=fake_encoder)


def test_reviews_without_text_are_not_eligible(tmp_path, fixture_settings):
    df = pd.read_parquet(fixture_settings.reviews_parquet)
    df.loc[:4, "combined_text"] = ""
    path = tmp_path / "blank.parquet"
    df.to_parquet(path)
    info = build_faiss_index(path, tmp_path / "out", encoder=fake_encoder)
    assert info["eligible_reviews"] == len(df) - 5 == info["indexed_reviews"]


# ---------- status and failure detection ----------
def test_coverage_is_computed_against_the_live_dataset(retriever):
    cov = retriever.coverage(eligible_reviews_now=480)
    assert cov["indexed_reviews"] == 240 and cov["eligible_reviews"] == 480 and cov["coverage_pct"] == 50.0


def test_index_built_from_older_data_is_flagged_stale(retriever, fixture_df):
    fresh = ids_fingerprint(fixture_df["review_id"].tolist())
    assert retriever.coverage(eligible_ids_md5_now=fresh)["ready"] is True
    stale = retriever.coverage(eligible_ids_md5_now="0" * 32)
    assert stale["ready"] is False and stale["limited_by"] == "stale_index"


def test_missing_vector_store_is_not_ready(tmp_path):
    r = ReviewSemanticRetriever(tmp_path / "nothing", encoder=fake_encoder)
    assert not r.is_ready() and r.search("battery") == [] and r.load_error
    assert r.coverage()["ready"] is False and r.coverage()["indexed_reviews"] == 0


def test_verify_index_detects_a_metadata_mismatch(tmp_path, fixture_settings):
    build_faiss_index(fixture_settings.reviews_parquet, tmp_path, encoder=fake_encoder)
    meta = pd.read_parquet(tmp_path / META_FILE)
    meta.iloc[:-10].to_parquet(tmp_path / META_FILE, index=False)  # lose the last 10 rows
    report = verify_index(tmp_path, encoder=fake_encoder)
    assert report["ok"] is False and any("vectors" in p for p in report["problems"])


def test_verify_index_detects_shuffled_metadata(tmp_path, fixture_settings):
    build_faiss_index(fixture_settings.reviews_parquet, tmp_path, encoder=fake_encoder)
    meta = pd.read_parquet(tmp_path / META_FILE)
    text_cols = ["review_id", "combined_text"]
    meta[text_cols] = meta[text_cols].iloc[::-1].to_numpy()  # vectors no longer line up with their reviews
    meta.to_parquet(tmp_path / META_FILE, index=False)
    assert verify_index(tmp_path, encoder=fake_encoder, sample_size=50)["ok"] is False


def test_retriever_refuses_an_inconsistent_store(tmp_path, fixture_settings):
    build_faiss_index(fixture_settings.reviews_parquet, tmp_path, encoder=fake_encoder)
    pd.read_parquet(tmp_path / META_FILE).iloc[:10].to_parquet(tmp_path / META_FILE, index=False)
    r = ReviewSemanticRetriever(tmp_path, encoder=fake_encoder)
    assert not r.is_ready() and "metadata has 10 rows" in r.load_error


# ---------- real index (skipped until it has been built) ----------
@pytest.mark.realdata
def test_real_index_covers_the_real_dataset(real_settings):
    import duckdb

    info = json.loads((real_settings.vector_store_dir / INFO_FILE).read_text())
    parquet = str(real_settings.reviews_parquet).replace("'", "''")
    eligible = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{parquet}') WHERE combined_text IS NOT NULL AND LENGTH(TRIM(combined_text)) >= 3").fetchone()[0]
    assert info["eligible_reviews"] == eligible
    assert info["indexed_reviews"] == eligible, f"index covers {info['coverage_pct']}% ({info['limited_by']})"
    assert info["coverage_pct"] == 100.0


def test_embedding_model_loads_while_holding_the_shared_import_lock(monkeypatch):
    """Importing sentence-transformers must not overlap with the DistilBERT import (see reviewpulse/heavy_imports.py)."""
    import sys
    import threading
    import types

    from reviewpulse.heavy_imports import HEAVY_IMPORT_LOCK
    from reviewpulse.nlp.semantic_search import load_sentence_encoder

    seen = []

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            got = []

            def attempt():
                ok = HEAVY_IMPORT_LOCK.acquire(blocking=False)
                got.append(ok)
                if ok:
                    HEAVY_IMPORT_LOCK.release()

            t = threading.Thread(target=attempt)
            t.start()
            t.join()
            seen.append(got[0])

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    encode = load_sentence_encoder("any-model")
    assert seen == [False], "another thread could take the import lock while the embedding model was loading"
    assert encode(["a", "b"]).shape == (2, 4)
