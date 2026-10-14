"""Semantic review retrieval using Sentence-Transformers and FAISS.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim dense vectors)
Index: faiss.IndexFlatIP over L2-normalised vectors, i.e. exact cosine similarity.

What this module guarantees:
 * By default EVERY eligible review (non-empty text) is indexed. A cap is applied only when you pass
   --max-samples or the machine does not have enough free memory, and the reason is saved in index_info.json.
 * Vector number i in the FAISS index is row i of vector_metadata.parquet, and that row carries the real
   review_id, so a hit can always be traced back to the review it came from.
 * index_info.json records how many reviews were eligible, how many were indexed and the coverage percentage,
   so the API and dashboard never have to guess what the search covers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import faiss
import numpy as np
import pandas as pd

from reviewpulse.config import DATA_PROCESSED_DIR, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR, ensure_directories
from reviewpulse.heavy_imports import HEAVY_IMPORT_LOCK

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss_index.bin"
META_FILE = "vector_metadata.parquet"
INFO_FILE = "index_info.json"

META_COLUMNS = [
    "review_id",
    "category",
    "rating",
    "sentiment",
    "title",
    "text",
    "combined_text",
    "product_title",
    "timestamp_utc",
    "helpful_vote",
    "verified_purchase",
]

# An Encoder turns a list of texts into an (n, dim) float32 array of L2-normalised vectors.
Encoder = Callable[[Sequence[str]], np.ndarray]

MIN_TEXT_CHARS = 3


def ids_fingerprint(review_ids: Sequence[str]) -> str:
    """Order-independent fingerprint of a set of review ids (used to detect an index built from older data)."""
    return hashlib.md5("".join(sorted(review_ids)).encode("utf-8")).hexdigest()


def load_sentence_encoder(model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = 64) -> Encoder:
    # The lock keeps this first import from racing with the DistilBERT loader (see heavy_imports.py).
    with HEAVY_IMPORT_LOCK:
        from sentence_transformers import SentenceTransformer

        # Prefer the copy already in the local cache: it loads in a few seconds and works offline. Without this the
        # library contacts the Hugging Face Hub on every load, which made the first assistant question take a minute.
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(model_name)  # first run: download it

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    return encode


def eligible_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews with usable text. These are the reviews the index is supposed to cover."""
    text = df["combined_text"].fillna("").astype(str)
    return df[text.str.strip().str.len() >= MIN_TEXT_CHARS].reset_index(drop=True)


def memory_cap(dimension: int, available_bytes: Optional[int], memory_fraction: float) -> Optional[int]:
    """Largest number of vectors that fit in the memory budget, or None if memory could not be measured.

    Only the vectors scale with the number of reviews (the embedding model is a fixed cost that is already
    loaded by the time this runs), so the budget is just `fraction of free RAM / bytes per vector`.
    """
    if available_bytes is None:
        return None
    per_vector = dimension * 4 * 2  # float32 embeddings plus the copy held inside the FAISS index
    return int(available_bytes * memory_fraction) // per_vector


def _available_memory() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _balanced_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Take about n/len(categories) reviews from each category so a cap does not drop whole categories."""
    per_cat = max(1, n // max(1, df["category"].nunique()))
    parts = [g.sample(n=min(len(g), per_cat), random_state=seed) for _, g in df.groupby("category")]
    sampled = pd.concat(parts)
    if len(sampled) < n:  # top up from the unsampled remainder (small categories left some budget over)
        rest = df.drop(sampled.index).sample(n=min(n - len(sampled), len(df) - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, rest])
    return sampled.sort_index().reset_index(drop=True)


def build_faiss_index(
    parquet_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    chunk_size: int = 2048,
    memory_fraction: float = 0.5,
    encoder: Optional[Encoder] = None,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    available_memory_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Encode reviews and write the FAISS index, its metadata table and index_info.json.

    max_samples=None (the default) indexes every eligible review unless memory is too small.
    """
    ensure_directories()
    input_file = Path(parquet_path) if parquet_path else DATA_PROCESSED_DIR / "reviews_all.parquet"
    out_dir = Path(output_dir) if output_dir else VECTOR_STORE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_file.exists():
        raise FileNotFoundError(f"Processed file {input_file} not found. Run the pipeline first.")

    df_all = pd.read_parquet(input_file)
    df = eligible_reviews(df_all)
    if df["review_id"].duplicated().any():
        raise ValueError("review_id values must be unique before indexing.")
    eligible_count = len(df)
    eligible_per_category = df.groupby("category").size().to_dict()
    logger.info("Loaded %s reviews, %s eligible for the semantic index.", f"{len(df_all):,}", f"{eligible_count:,}")

    encode = encoder or load_sentence_encoder(embedding_model, batch_size)

    # Encode one review first to learn the vector size, which the memory estimate needs.
    probe = encode([df["combined_text"].iloc[0]])
    dimension = int(probe.shape[1])

    limited_by: Optional[str] = None
    target = eligible_count
    cap = memory_cap(dimension, available_memory_bytes if available_memory_bytes is not None else _available_memory(), memory_fraction)
    if cap is not None and cap < target:
        target, limited_by = max(1, cap), "memory"
        logger.warning("Only enough free memory for %s vectors; indexing a capped, category-balanced sample.", f"{target:,}")
    if max_samples is not None and max_samples < target:
        target, limited_by = max_samples, "max_samples"
    df_index = df if target >= eligible_count else _balanced_sample(df, target)
    logger.info("Indexing %s of %s eligible reviews (%s).", f"{len(df_index):,}", f"{eligible_count:,}", limited_by or "full coverage")

    index = faiss.IndexFlatIP(dimension)
    texts = df_index["combined_text"].tolist()
    for start in range(0, len(texts), chunk_size):
        vectors = np.ascontiguousarray(encode(texts[start : start + chunk_size]), dtype=np.float32)
        index.add(vectors)
        logger.info("  encoded %s / %s", f"{min(start + chunk_size, len(texts)):,}", f"{len(texts):,}")
    if index.ntotal != len(df_index):
        raise RuntimeError(f"FAISS holds {index.ntotal} vectors but {len(df_index)} reviews were encoded.")

    meta = df_index[META_COLUMNS].copy()
    meta.insert(0, "vector_id", np.arange(len(meta), dtype=np.int64))

    index_path, meta_path, info_path = out_dir / INDEX_FILE, out_dir / META_FILE, out_dir / INFO_FILE
    indexed_per_category = df_index.groupby("category").size().to_dict()
    info = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "index_type": "IndexFlatIP (exact cosine similarity on L2-normalised vectors)",
        "eligible_reviews": eligible_count,
        "indexed_reviews": len(df_index),
        "coverage_pct": round(len(df_index) * 100.0 / max(1, eligible_count), 2),
        "limited_by": limited_by,
        "max_samples_requested": max_samples,
        "source_file": input_file.name,
        "source_rows": len(df_all),
        "eligible_ids_md5": ids_fingerprint(df["review_id"].tolist()),
        "per_category": {c: {"eligible": int(eligible_per_category[c]), "indexed": int(indexed_per_category.get(c, 0))} for c in sorted(eligible_per_category)},
    }

    # Write to temporary names, then rename, so a crash never leaves an index that disagrees with its metadata.
    faiss.write_index(index, str(index_path) + ".tmp")
    meta.to_parquet(str(meta_path) + ".tmp", index=False)
    Path(str(index_path) + ".tmp").replace(index_path)
    Path(str(meta_path) + ".tmp").replace(meta_path)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Saved index (%.2f MB), metadata and %s. Coverage %.2f%%.", index_path.stat().st_size / 1e6, INFO_FILE, info["coverage_pct"])
    return info


def verify_index(index_dir: Optional[Path] = None, encoder: Optional[Encoder] = None, sample_size: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Check the saved index against its metadata.

    1. Vector count, metadata rows and index_info.json agree, and review ids are unique.
    2. Re-encode a random sample of review texts: each must come back as its own top hit, which proves
       that vector i really belongs to metadata row i.
    """
    d = Path(index_dir) if index_dir else VECTOR_STORE_DIR
    info = json.loads((d / INFO_FILE).read_text(encoding="utf-8"))
    index = faiss.read_index(str(d / INDEX_FILE))
    meta = pd.read_parquet(d / META_FILE)
    problems: List[str] = []
    if index.ntotal != len(meta):
        problems.append(f"index has {index.ntotal} vectors but metadata has {len(meta)} rows")
    if info["indexed_reviews"] != len(meta):
        problems.append(f"index_info says {info['indexed_reviews']} indexed reviews, metadata has {len(meta)}")
    if meta["review_id"].duplicated().any():
        problems.append("duplicate review_id values in metadata")
    if not np.array_equal(meta["vector_id"].to_numpy(), np.arange(len(meta))):
        problems.append("vector_id is not 0..n-1 in order")

    checked = hits = ties = 0
    if not problems:
        encode = encoder or load_sentence_encoder(info["embedding_model"])
        rows = meta.sample(n=min(sample_size, len(meta)), random_state=seed)
        vectors = np.ascontiguousarray(encode(rows["combined_text"].tolist()), dtype=np.float32)
        top_scores, ids = index.search(vectors, 1)
        for (_, row), query_vec, top, top_score in zip(rows.iterrows(), vectors, ids[:, 0], top_scores[:, 0]):
            checked += 1
            own = int(row["vector_id"])
            if int(top) == own:
                hits += 1
                continue
            # Different review came back first. That is only acceptable if this review's own vector scores exactly
            # as high, i.e. the two embeddings are identical. The embedding model ignores letter case, so
            # "Good Product" and "Good product" are the same vector. Anything else is a real mapping error.
            own_score = float(np.dot(query_vec, index.reconstruct(own)))
            if float(top_score) - own_score <= 1e-4:
                hits += 1
                ties += 1
            else:
                problems.append(
                    f"review {row['review_id']} (vector {own}) scored {own_score:.4f} against itself "
                    f"but vector {int(top)} scored {float(top_score):.4f}"
                )
    return {
        "ok": not problems,
        "problems": problems[:10],
        "indexed_reviews": len(meta),
        "eligible_reviews": info["eligible_reviews"],
        "coverage_pct": info["coverage_pct"],
        "self_retrieval_checked": checked,
        "self_retrieval_correct": hits,
        "self_retrieval_identical_text_ties": ties,
    }


class ReviewSemanticRetriever:
    """Loads a saved index and answers similarity queries. The embedding model loads on first use."""

    def __init__(self, index_dir: Optional[Path] = None, encoder: Optional[Encoder] = None, min_score: float = 0.0):
        self.dir = Path(index_dir) if index_dir else VECTOR_STORE_DIR
        self.min_score = min_score
        self._encoder = encoder
        self._encoder_lock = threading.Lock()
        self.index = None
        self.metadata: Optional[pd.DataFrame] = None
        self.info: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        if (self.dir / INDEX_FILE).exists() and (self.dir / META_FILE).exists():
            self._load()
        else:
            self.load_error = f"No vector store found in {self.dir}. Run: python -m reviewpulse.nlp.semantic_search"
            logger.warning(self.load_error)

    def _load(self) -> None:
        try:
            self.index = faiss.read_index(str(self.dir / INDEX_FILE))
            self.metadata = pd.read_parquet(self.dir / META_FILE)
            info_path = self.dir / INFO_FILE
            self.info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
            if self.index.ntotal != len(self.metadata):
                raise ValueError(f"index has {self.index.ntotal} vectors but metadata has {len(self.metadata)} rows")
            self._category_lower = self.metadata["category"].str.lower().to_numpy()
            self._sentiment_lower = self.metadata["sentiment"].str.lower().to_numpy()
            logger.info("Retriever loaded %s indexed reviews.", f"{self.index.ntotal:,}")
        except Exception as exc:
            self.index, self.metadata = None, None
            self.load_error = f"Vector store could not be loaded: {exc}"
            logger.warning(self.load_error)

    # ---------- status ----------
    def is_ready(self) -> bool:
        return self.index is not None and self.metadata is not None

    def coverage(self, eligible_reviews_now: Optional[int] = None, eligible_ids_md5_now: Optional[str] = None) -> Dict[str, Any]:
        """Facts about what the index covers, compared with the live dataset when its size/fingerprint is given."""
        indexed = int(self.index.ntotal) if self.index is not None else 0
        eligible = int(eligible_reviews_now if eligible_reviews_now is not None else self.info.get("eligible_reviews", indexed))
        limited_by = self.info.get("limited_by")
        ready = self.is_ready()
        if ready and eligible_ids_md5_now and self.info.get("eligible_ids_md5") and eligible_ids_md5_now != self.info["eligible_ids_md5"]:
            # The processed data changed after the index was built, so its review ids may not exist any more.
            ready, limited_by = False, "stale_index"
        return {
            "ready": ready,
            "indexed_reviews": indexed,
            "eligible_reviews": eligible,
            "coverage_pct": round(indexed * 100.0 / eligible, 2) if eligible else 0.0,
            "embedding_model": self.info.get("embedding_model"),
            "built_at": self.info.get("built_at"),
            "limited_by": limited_by,
        }

    def warm_up(self) -> None:
        """Load the embedding model now instead of during the first user question."""
        if self.is_ready():
            self._encode_query("warm up")

    # ---------- search ----------
    def _encode_query(self, query: str) -> np.ndarray:
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = load_sentence_encoder(self.info.get("embedding_model", EMBEDDING_MODEL_NAME))
            encoder = self._encoder
        return np.ascontiguousarray(encoder([query]), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Most similar reviews, best first. Filters are applied inside the search, so asking for a category
        never returns fewer hits just because other categories scored higher."""
        if not self.is_ready() or not query.strip():
            return []
        assert self.index is not None and self.metadata is not None

        mask = np.ones(len(self.metadata), dtype=bool)
        if category:
            mask &= self._category_lower == category.lower()
        if sentiment:
            mask &= self._sentiment_lower == sentiment.lower()
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []

        k = int(min(max(1, top_k), candidates.size))
        query_vec = self._encode_query(query)
        if candidates.size == len(mask):
            scores, ids = self.index.search(query_vec, k)
        else:
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(candidates.astype(np.int64))
            scores, ids = self.index.search(query_vec, k, params=params)

        threshold = self.min_score if min_score is None else min_score
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or float(score) < threshold:
                continue
            row = self.metadata.iloc[int(idx)]
            text = str(row["text"])
            results.append(
                {
                    "vector_id": int(idx),
                    "review_id": row["review_id"],
                    "score": round(float(score), 4),
                    "rating": float(row["rating"]),
                    "sentiment": row["sentiment"],
                    "category": row["category"],
                    "title": row["title"],
                    "text": text[:300] + ("..." if len(text) > 300 else ""),
                    "full_text": text,
                    "product_title": row["product_title"],
                    "timestamp_utc": row["timestamp_utc"],
                    "verified_purchase": bool(row["verified_purchase"]),
                    "helpful_vote": int(row["helpful_vote"]),
                }
            )
        return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build (and optionally verify) the FAISS semantic search index.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on indexed reviews. Default: index every eligible review.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Reviews encoded and added to the index at a time.")
    parser.add_argument("--memory-fraction", type=float, default=0.5, help="Share of currently free RAM the index may use.")
    parser.add_argument("--verify", action="store_true", help="Run verify_index() after building, or on its own with --verify-only.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        build_faiss_index(max_samples=args.max_samples, batch_size=args.batch_size, chunk_size=args.chunk_size, memory_fraction=args.memory_fraction)
    if args.verify or args.verify_only:
        print(json.dumps(verify_index(), indent=2))
