"""Dataset loading and leakage-free splitting for sentiment classification."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from reviewpulse.config import DATA_PROCESSED_DIR, ensure_directories

logger = logging.getLogger(__name__)


def load_and_split_data(
    parquet_path: Optional[Path] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    group_column: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load processed reviews and split them into train / validation / test without leakage.

    Guarantees:
    1. Exact duplicate review texts are removed before splitting.
    2. **No reviewer appears in more than one split.** People write consistently (same rating habits, same phrases),
       so if one person's reviews were spread across train and test, the model would be tested on writing it had
       already seen and the scores would be too optimistic. Splitting by reviewer prevents that.
    3. Within that rule, the three sentiment classes are kept in similar proportions in every split
       (scikit-learn's StratifiedGroupKFold).
    4. The test set is only used for the final evaluation.
    5. Targets (rating, sentiment) are never part of the input text.
    """
    ensure_directories()
    input_file = Path(parquet_path) if parquet_path else DATA_PROCESSED_DIR / "reviews_all.parquet"
    if not input_file.exists():
        raise FileNotFoundError(f"Processed file {input_file} not found. Run pipeline first.")

    df = pd.read_parquet(input_file)
    initial_count = len(df)

    # Clean missing text and drop duplicate review texts to prevent cross-split leakage
    df = df.dropna(subset=["combined_text", "sentiment"]).copy()
    df = df[df["combined_text"].str.strip().str.len() >= 3]
    df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    dedup_count = len(df)
    logger.info(f"Loaded {initial_count:,} reviews. After dropping text duplicates: {dedup_count:,} reviews.")

    # Cut the data into k folds of whole reviewers, then use one fold as test, one as validation and the rest to train.
    # With 15% / 15% this means k = 7 folds of about 14.3% each.
    n_splits = max(3, round(1.0 / min(test_size, val_size)))
    n_test_folds = max(1, round(test_size * n_splits))
    n_val_folds = max(1, round(val_size * n_splits))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [idx for _, idx in splitter.split(df, df["sentiment_encoded"], groups=df[group_column])]
    test_idx = np.concatenate(folds[:n_test_folds])
    val_idx = np.concatenate(folds[n_test_folds : n_test_folds + n_val_folds])
    train_idx = np.concatenate(folds[n_test_folds + n_val_folds :])
    train_df, val_df, test_df = (df.iloc[sorted(i)].reset_index(drop=True) for i in (train_idx, val_idx, test_idx))

    users = {name: set(part[group_column]) for name, part in (("train", train_df), ("val", val_df), ("test", test_df))}
    overlap = (users["train"] & users["val"]) | (users["train"] & users["test"]) | (users["val"] & users["test"])
    if overlap:  # cannot happen with a group split; this guards against future edits
        raise RuntimeError(f"{len(overlap)} reviewers appear in more than one split")

    # Fingerprint of the exact test reviews, so a saved result can be matched to the split it was scored on.
    test_ids_md5 = hashlib.md5("".join(sorted(test_df["review_id"])).encode("utf-8")).hexdigest()

    split_metadata = {
        "source_rows": initial_count,
        "total_deduplicated_samples": dedup_count,
        "test_ids_md5": test_ids_md5,
        "split_strategy": f"StratifiedGroupKFold by {group_column} ({n_splits} folds: {n_test_folds} test, {n_val_folds} validation, rest train)",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_pct": round(len(train_df) / dedup_count * 100, 1),
        "val_pct": round(len(val_df) / dedup_count * 100, 1),
        "test_pct": round(len(test_df) / dedup_count * 100, 1),
        "train_reviewers": len(users["train"]),
        "val_reviewers": len(users["val"]),
        "test_reviewers": len(users["test"]),
        "reviewers_shared_between_splits": 0,
        "random_state": random_state,
        "class_distribution_train": train_df["sentiment"].value_counts().to_dict(),
        "class_distribution_val": val_df["sentiment"].value_counts().to_dict(),
        "class_distribution_test": test_df["sentiment"].value_counts().to_dict(),
    }

    logger.info(
        f"Split complete: Train={len(train_df):,} ({split_metadata['train_pct']}%), "
        f"Val={len(val_df):,} ({split_metadata['val_pct']}%), "
        f"Test={len(test_df):,} ({split_metadata['test_pct']}%); no reviewer appears in two splits."
    )

    return train_df, val_df, test_df, split_metadata
