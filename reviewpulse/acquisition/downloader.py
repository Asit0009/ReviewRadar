"""Real data acquisition module for Amazon Reviews 2023 dataset.

Streams reviews and item metadata directly from official Hugging Face endpoints:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from reviewpulse.config import (
    DATA_RAW_DIR,
    DEFAULT_LIMIT_PER_CATEGORY,
    DEFAULT_META_LIMIT_PER_CATEGORY,
    HF_META_BASE_URL,
    HF_REVIEWS_BASE_URL,
    MANIFEST_PATH,
    OFFICIAL_CATEGORIES,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_machine_resources() -> Dict[str, float]:
    """Inspect and log host hardware resources before data acquisition."""
    total, used, free = shutil.disk_usage(".")
    disk_total_gb = round(total / (1024**3), 2)
    disk_free_gb = round(free / (1024**3), 2)

    import psutil

    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / (1024**3), 2)
    ram_available_gb = round(ram.available / (1024**3), 2)
    cpu_count = os.cpu_count() or 1

    # Check CUDA availability
    has_gpu = False
    gpu_name = "None"
    try:
        import torch

        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    resource_info = {
        "disk_total_gb": disk_total_gb,
        "disk_free_gb": disk_free_gb,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "cpu_count": cpu_count,
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
    }

    logger.info(
        f"Resource inspection: Free Disk={disk_free_gb} GB / {disk_total_gb} GB | "
        f"Available RAM={ram_available_gb} GB / {ram_total_gb} GB | "
        f"CPUs={cpu_count} | GPU={gpu_name}"
    )

    if disk_free_gb < 1.0:
        logger.warning("Low disk space warning: less than 1.0 GB available!")

    return resource_info


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def stream_download_jsonl(
    url: str, output_path: Path, max_records: int, chunk_size: int = 65536, max_retries: int = 3
) -> int:
    """Stream raw JSONL lines over HTTP with retry resilience and atomic temp write."""
    logger.info(f"Streaming from {url} -> {output_path} (limit={max_records:,})")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReviewPulseDataPipeline/1.0"
        },
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_err = None

    for attempt in range(1, max_retries + 1):
        records_written = 0
        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=45) as response, open(
                tmp_path, "w", encoding="utf-8"
            ) as out_file:
                buffer = ""
                while records_written < max_records:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        if buffer.strip():
                            out_file.write(buffer.strip() + "\n")
                            records_written += 1
                        break

                    text_chunk = chunk.decode("utf-8", errors="replace")
                    buffer += text_chunk

                    while "\n" in buffer and records_written < max_records:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            out_file.write(line + "\n")
                            records_written += 1

            if tmp_path.exists():
                if output_path.exists():
                    output_path.unlink()
                tmp_path.rename(output_path)

            elapsed = time.time() - start_time
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Completed streaming {output_path.name}: {records_written:,} records, "
                f"{file_size_mb:.2f} MB in {elapsed:.1f}s"
            )
            return records_written
        except Exception as e:
            last_err = e
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            logger.warning(
                f"Download attempt {attempt}/{max_retries} for {output_path.name} failed: {e}. "
                f"{'Retrying in 3s...' if attempt < max_retries else 'No more retries.'}"
            )
            if attempt < max_retries:
                time.sleep(3)

    raise last_err


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReviewPulseDataPipeline/1.0"


def _range_request(url: str, start: int, end: int, timeout: int = 45) -> tuple:
    """Fetch bytes [start, end] of a remote file. Returns (data, total_file_size)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")  # e.g. "bytes 100-200/22616233652"
        total = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else 0
        return response.read(), total


def _lines_from_chunk(data: bytes, from_file_start: bool, max_lines: int) -> List[str]:
    """Complete, valid JSON lines from a byte chunk. A chunk that starts mid-file begins mid-line, so its first line is dropped."""
    lines = data.decode("utf-8", errors="replace").split("\n")
    if not from_file_start:
        lines = lines[1:]
    lines = lines[:-1]  # the last line may be cut off by the end of the chunk
    out: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def sample_download_jsonl(
    url: str,
    output_path: Path,
    max_records: int,
    n_chunks: int = 500,
    seed: int = 42,
    workers: int = 8,
    max_retries: int = 3,
) -> Dict:
    """Download `max_records` reviews taken from `n_chunks` places spread evenly across a huge remote JSONL file.

    Why not just the first lines: the source files are stored in runs of reviews by the same reviewer, so the first
    few thousand lines come from only a few hundred people. Their writing and rating habits then dominate the data
    set, and the same person would end up on both sides of a train/test split. Taking many small chunks from all over
    the file gives many more distinct reviewers and a spread of dates.

    The request offsets are fixed by `seed`, so the same call always fetches the same reviews.
    """
    logger.info(f"Sampling {max_records:,} records from {n_chunks} spread-out chunks of {url}")
    _, total_size = _range_request(url, 0, 0)
    if total_size <= 0:
        raise RuntimeError(f"Server did not report a file size for {url}; range sampling is not possible.")

    per_chunk = math.ceil(max_records * 1.15 / n_chunks)  # take a few extra, then trim after removing duplicates
    chunk_bytes = max(32 * 1024, per_chunk * 4096)
    rng = random.Random(seed)
    span = max(1, (total_size - chunk_bytes) // n_chunks)
    offsets = [min(i * span + rng.randrange(span), total_size - chunk_bytes) for i in range(n_chunks)]
    offsets[0] = 0

    def fetch(i: int) -> List[str]:
        last_err: Exception = RuntimeError("no attempt made")
        for attempt in range(1, max_retries + 1):
            try:
                data, _ = _range_request(url, offsets[i], offsets[i] + chunk_bytes)
                return _lines_from_chunk(data, from_file_start=offsets[i] == 0, max_lines=per_chunk)
            except Exception as exc:  # network hiccup: retry this chunk only
                last_err = exc
                time.sleep(1.5 * attempt)
        logger.warning(f"Chunk {i} at offset {offsets[i]:,} failed after {max_retries} attempts: {last_err}")
        return []

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(fetch, range(n_chunks)))

    seen, kept = set(), []
    for lines in chunks:
        for line in lines:
            rec = json.loads(line)
            key = (rec.get("user_id"), rec.get("asin"), rec.get("timestamp"))
            if key in seen:
                continue
            seen.add(key)
            kept.append(line)
    # Trim evenly (every k-th record) rather than cutting the tail, so every part of the file stays represented.
    if len(kept) > max_records:
        step = len(kept) / max_records
        kept = [kept[int(i * step)] for i in range(max_records)]

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    if output_path.exists():
        output_path.unlink()
    tmp.rename(output_path)
    users = len({json.loads(line).get("user_id") for line in kept})
    logger.info(f"Saved {output_path.name}: {len(kept):,} records from {users:,} distinct reviewers in {time.time() - started:.0f}s")
    return {
        "records": len(kept),
        "distinct_users": users,
        "sampling": {"strategy": "spread", "chunks": n_chunks, "seed": seed, "lines_per_chunk": per_chunk, "source_file_size_bytes": total_size},
    }


def load_manifest() -> Dict:
    """Load existing manifest or create a new one."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest: {e}. Starting fresh.")
    return {
        "dataset_name": "McAuley-Lab/Amazon-Reviews-2023",
        "dataset_guide_url": "https://amazon-reviews-2023.github.io/main.html",
        "huggingface_url": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_snapshot": {},
        "categories": {},
    }


PATH_KEYS = {"review_file_path", "meta_file_path", "processed_file"}


def _with_portable_paths(value):
    """Paths in the manifest are written with forward slashes, so the file reads the same on every machine."""
    if isinstance(value, dict):
        return {k: (v.replace("\\", "/") if k in PATH_KEYS and isinstance(v, str) else _with_portable_paths(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_with_portable_paths(v) for v in value]
    return value


def save_manifest(manifest: Dict):
    """Save manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(_with_portable_paths(manifest), f, indent=2)
    logger.info(f"Data manifest saved to {MANIFEST_PATH}")


def _count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def acquire_category_data(
    category: str,
    review_limit: int = DEFAULT_LIMIT_PER_CATEGORY,
    meta_limit: int = DEFAULT_META_LIMIT_PER_CATEGORY,
    force: bool = False,
    sampling: str = "spread",
    chunks: int = 500,
    seed: int = 42,
) -> Dict:
    """Acquire real raw reviews (and optionally product metadata) for one category.

    sampling="spread" (default) takes small chunks from all over the source file; "head" takes the first lines.
    meta_limit=0 skips product metadata. Metadata only matches reviews that sit near the start of its file, so with
    spread sampling almost no review would get a product title and the download would be wasted.
    """
    if sampling not in ("spread", "head"):
        raise ValueError("sampling must be 'spread' or 'head'")
    ensure_directories()
    manifest = load_manifest()

    review_url = HF_REVIEWS_BASE_URL.format(category=category)
    meta_url = HF_META_BASE_URL.format(category=category)
    review_file = DATA_RAW_DIR / f"{category}_reviews.jsonl"
    meta_file = DATA_RAW_DIR / f"{category}_meta.jsonl"

    previous = manifest.get("categories", {}).get(category, {})
    same_sampling = previous.get("sampling", {}).get("strategy", "head") == sampling and previous.get("target_limit_requested") == review_limit
    sampling_info: Dict = previous.get("sampling", {}) if same_sampling else {}

    if review_file.exists() and not force and same_sampling and _count_lines(review_file) >= review_limit:
        review_count = _count_lines(review_file)
        logger.info(f"Raw reviews for {category} already exist ({review_count:,} lines, {sampling} sampling). Skipping download.")
    elif sampling == "spread":
        result = sample_download_jsonl(review_url, review_file, max_records=review_limit, n_chunks=chunks, seed=seed)
        review_count, sampling_info = result["records"], result["sampling"]
    else:
        review_count = stream_download_jsonl(review_url, review_file, max_records=review_limit)
        sampling_info = {"strategy": "head"}

    meta_count, meta_sha, meta_size = 0, None, 0
    if meta_limit > 0:
        if meta_file.exists() and not force and _count_lines(meta_file) >= meta_limit:
            meta_count = _count_lines(meta_file)
        else:
            meta_count = stream_download_jsonl(meta_url, meta_file, max_records=meta_limit)
        meta_sha, meta_size = calculate_sha256(meta_file), meta_file.stat().st_size

    reviews = [json.loads(line) for line in open(review_file, encoding="utf-8") if line.strip()]
    category_manifest = {
        "category_name": category,
        "review_source_url": review_url,
        "review_file_path": str(review_file.relative_to(DATA_RAW_DIR.parent.parent)),
        "review_file_size_bytes": review_file.stat().st_size,
        "review_sha256": calculate_sha256(review_file),
        "downloaded_raw_reviews": review_count,
        "distinct_reviewers": len({r.get("user_id") for r in reviews}),
        "sampling": sampling_info,
        "meta_source_url": meta_url if meta_limit > 0 else None,
        "meta_file_size_bytes": meta_size,
        "meta_sha256": meta_sha,
        "downloaded_raw_meta": meta_count,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_working_limit_capped": True,
        "target_limit_requested": review_limit,
    }
    manifest["hardware_snapshot"] = inspect_machine_resources()
    manifest["categories"][category] = category_manifest
    save_manifest(manifest)
    return category_manifest


def acquire_all_categories(
    categories: Optional[List[str]] = None,
    review_limit: int = DEFAULT_LIMIT_PER_CATEGORY,
    meta_limit: int = DEFAULT_META_LIMIT_PER_CATEGORY,
    force: bool = False,
    sampling: str = "spread",
    chunks: int = 500,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Acquire real raw reviews for all target categories."""
    target_categories = categories or OFFICIAL_CATEGORIES
    logger.info(f"Starting data acquisition for {len(target_categories)} categories (review_limit={review_limit:,}, sampling={sampling})")
    inspect_machine_resources()
    results = {}
    for cat in target_categories:
        try:
            results[cat] = acquire_category_data(cat, review_limit=review_limit, meta_limit=meta_limit, force=force, sampling=sampling, chunks=chunks, seed=seed)
        except Exception as e:
            logger.error(f"Failed to acquire data for {cat}: {e}")
            raise
    logger.info("Data acquisition completed successfully for all categories.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire official Amazon Reviews 2023 data.")
    parser.add_argument("--categories", nargs="+", default=OFFICIAL_CATEGORIES, help="Category names to download")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT_PER_CATEGORY, help="Reviews to keep per category")
    parser.add_argument("--meta-limit", type=int, default=DEFAULT_META_LIMIT_PER_CATEGORY, help="Product metadata records per category (0 = skip)")
    parser.add_argument("--sampling", choices=["spread", "head"], default="spread", help="spread: chunks from all over each file (default). head: the first lines.")
    parser.add_argument("--chunks", type=int, default=500, help="Number of spread-out chunks per category")
    parser.add_argument("--seed", type=int, default=42, help="Seed that fixes which chunks are fetched")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
    args = parser.parse_args()
    acquire_all_categories(
        categories=args.categories, review_limit=args.limit, meta_limit=args.meta_limit, force=args.force,
        sampling=args.sampling, chunks=args.chunks, seed=args.seed,
    )
