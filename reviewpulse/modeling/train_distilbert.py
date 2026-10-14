"""DistilBERT fine-tuning for 3-class sentiment (negative / neutral / positive).

This trains a real model and evaluates it on the SAME held-out test reviews as the TF-IDF models, so the
scores are directly comparable. Nothing is reported unless training and evaluation both finish.

Resource-aware defaults (this was developed on a 4-core CPU with under 2 GB of free RAM):
  * trains on a stratified subset of the training split (--train-samples, 0 = all of it),
  * freezes the word-embedding table (saves ~280 MB of gradients and optimizer state),
  * uses a small batch with gradient accumulation and pads each batch only to its longest review.
With a GPU, run with --train-samples 0 --epochs 3 to use the full training split.

Usage:
    python -m reviewpulse.modeling.train_distilbert                       # default CPU-sized run
    python -m reviewpulse.modeling.train_distilbert --train-samples 0 --epochs 3   # full data (GPU recommended)
    python -m reviewpulse.modeling.train_distilbert --eval-only           # re-score a saved model
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from reviewpulse.config import MODELS_DIR, ensure_directories
from reviewpulse.modeling.dataset import load_and_split_data
from reviewpulse.modeling.evaluation import (
    compute_metrics,
    load_metrics,
    now_utc,
    per_category_metrics,
    record_model_result,
)
from reviewpulse.modeling.inference import DISTILBERT_DIR_NAME

logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = MODELS_DIR / DISTILBERT_DIR_NAME


def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Keep about n rows while preserving the class proportions of the original split."""
    if n <= 0 or n >= len(df):
        return df
    return df.groupby("sentiment_encoded", group_keys=False).apply(
        lambda g: g.sample(n=max(1, round(n * len(g) / len(df))), random_state=seed), include_groups=False
    ).pipe(lambda sub: df.loc[sub.index])


def make_batches(token_ids: List[List[int]], order: Sequence[int], batch_size: int, pad_id: int):
    """Yield (indices, input_ids, attention_mask) padded only to the longest sequence in the batch."""
    for start in range(0, len(order), batch_size):
        idx = list(order[start : start + batch_size])
        longest = max(len(token_ids[i]) for i in idx)
        ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, i in enumerate(idx):
            seq = token_ids[i]
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        yield idx, ids, mask


@torch.no_grad()
def predict(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int = 32) -> np.ndarray:
    """Predicted class ids for texts. Sorted by length internally so batches have little padding."""
    model.eval()
    token_ids = tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: len(token_ids[i]))
    preds = np.zeros(len(texts), dtype=np.int64)
    done = 0
    started = time.time()
    for idx, ids, mask in make_batches(token_ids, order, batch_size, tokenizer.pad_token_id):
        logits = model(input_ids=ids, attention_mask=mask).logits
        preds[idx] = logits.argmax(dim=-1).numpy()
        done += len(idx)
        if (done // batch_size) % 25 == 0:
            logger.info("  scored %s / %s reviews (%.0fs)", f"{done:,}", f"{len(texts):,}", time.time() - started)
    return preds


def train(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    ensure_directories()
    torch.manual_seed(42)
    np.random.seed(42)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, split_info = load_and_split_data()
    existing = load_metrics(MODELS_DIR / "metrics.json")
    if existing.get("split_info", {}).get("test_ids_md5") != split_info["test_ids_md5"]:
        raise SystemExit("metrics.json was produced from a different data split. Run train_classical first so all models share one test set.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL if not args.eval_only else str(OUTPUT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL if not args.eval_only else str(OUTPUT_DIR), num_labels=3)
    model.to(device)
    training_info: Dict[str, Any] = {}

    if args.eval_only:
        previous = existing.get("models", {}).get("distilbert", {}).get("training", {})
        training_info = previous
    else:
        train_sub = stratified_subset(train_df, args.train_samples)
        logger.info("Training on %s of %s training reviews, %s epochs, device=%s", f"{len(train_sub):,}", f"{len(train_df):,}", args.epochs, device)

        # Class weights make mistakes on the rare neutral/negative classes count more, like class_weight='balanced'.
        counts = np.bincount(train_sub["sentiment_encoded"].to_numpy(), minlength=3).astype(np.float64)
        weights = torch.tensor(len(train_sub) / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        if args.freeze_embeddings:
            for p in model.distilbert.embeddings.word_embeddings.parameters():
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        texts = train_sub["combined_text"].tolist()
        labels = torch.tensor(train_sub["sentiment_encoded"].to_numpy(), dtype=torch.long)
        token_ids = tokenizer(texts, truncation=True, max_length=args.max_length)["input_ids"]
        steps_per_epoch = math.ceil(len(texts) / (args.batch_size * args.grad_accum))
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

        loss_history: List[float] = []
        started = time.time()
        step = 0
        rng = np.random.default_rng(42)
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(len(texts)).tolist()
            running, seen_batches = 0.0, 0
            optimizer.zero_grad()
            for b, (idx, ids, mask) in enumerate(make_batches(token_ids, order, args.batch_size, tokenizer.pad_token_id)):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = loss_fn(logits, labels[idx].to(device))
                (loss / args.grad_accum).backward()
                running += loss.item()
                seen_batches += 1
                if (b + 1) % args.grad_accum == 0 or (b + 1) * args.batch_size >= len(texts):
                    clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 10 == 0:
                        elapsed = time.time() - started
                        eta = elapsed / step * (total_steps - step)
                        logger.info("epoch %d/%d step %d/%d loss %.4f elapsed %.0fs eta %.0fs", epoch + 1, args.epochs, step, total_steps, running / seen_batches, elapsed, eta)
            loss_history.append(round(running / max(1, seen_batches), 4))
            logger.info("Epoch %d finished. Mean training loss %.4f", epoch + 1, loss_history[-1])
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)  # saved every epoch so an interrupted run still leaves a usable model
            tokenizer.save_pretrained(OUTPUT_DIR)

        training_info = {
            "base_model": BASE_MODEL,
            "device": str(device),
            "train_samples": int(len(texts)),
            "train_split_size": int(len(train_df)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "max_length": args.max_length,
            "learning_rate": args.lr,
            "class_weighted_loss": True,
            "frozen_word_embeddings": bool(args.freeze_embeddings),
            "loss_per_epoch": loss_history,
            "training_seconds": round(time.time() - started, 1),
        }

    # ---- evaluation on held-out data (same test reviews as the classical models) ----
    val_eval = val_df if args.val_samples <= 0 else stratified_subset(val_df, args.val_samples)
    logger.info("Scoring %s validation reviews...", f"{len(val_eval):,}")
    val_pred = predict(model, tokenizer, val_eval["combined_text"].tolist(), args.max_length)
    logger.info("Scoring all %s test reviews...", f"{len(test_df):,}")
    test_pred = predict(model, tokenizer, test_df["combined_text"].tolist(), args.max_length)

    val_metrics = compute_metrics(val_eval["sentiment_encoded"].to_numpy(), val_pred)
    test_metrics = compute_metrics(test_df["sentiment_encoded"].to_numpy(), test_pred)
    result = {
        "model_name": BASE_MODEL,
        "validation": val_metrics,
        "test": test_metrics,
        "evaluated_on": {"validation_samples": int(len(val_eval)), "test_samples": int(len(test_df))},
        "training": training_info,
        "trained_at": now_utc(),
        "artifacts_saved_to": f"artifacts/models/{DISTILBERT_DIR_NAME}",
    }
    record_model_result(MODELS_DIR / "metrics.json", "distilbert", result, per_category=per_category_metrics(test_df, test_pred))
    logger.info("DistilBERT test accuracy %.4f, macro-F1 %.4f", test_metrics["accuracy"], test_metrics["macro_f1"])
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for review sentiment.")
    parser.add_argument("--train-samples", type=int, default=6000, help="Training reviews to use; 0 means the whole training split.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Batches per optimizer step (effective batch = batch-size x this).")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val-samples", type=int, default=1500, help="Validation reviews to score; 0 means all of them.")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = library default).")
    parser.add_argument("--no-freeze-embeddings", dest="freeze_embeddings", action="store_false", help="Train the word-embedding table too (needs more memory).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; score the saved model and update metrics.json.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
