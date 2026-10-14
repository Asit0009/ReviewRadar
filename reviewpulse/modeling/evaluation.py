"""Metrics shared by every sentiment model, plus the helpers that write them to metrics.json.

Everything the dashboard shows about model quality is read from metrics.json, and metrics.json is only
ever written by the training scripts after a real evaluation on held-out data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

TARGET_NAMES = ["negative", "neutral", "positive"]
LABEL_IDS = [0, 1, 2]

# A per-category score is only called reliable when every class has at least this many test reviews.
# With fewer, one extra mistake moves that class's F1 by several points.
MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS = 20


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Accuracy, macro/weighted F1, per-class precision/recall/F1 and the 3x3 confusion matrix.

    `labels` is fixed so a split that happens to lack one class still returns all three columns.
    """
    report = classification_report(
        y_true, y_pred, labels=LABEL_IDS, target_names=TARGET_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    metrics: Dict[str, Any] = {
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }
    for name in TARGET_NAMES:
        metrics[f"{name}_f1"] = round(float(report[name]["f1-score"]), 4)
    for name in TARGET_NAMES:
        metrics[f"{name}_precision"] = round(float(report[name]["precision"]), 4)
    for name in TARGET_NAMES:
        metrics[f"{name}_recall"] = round(float(report[name]["recall"]), 4)
    metrics["macro_precision"] = round(float(report["macro avg"]["precision"]), 4)
    metrics["macro_recall"] = round(float(report["macro avg"]["recall"]), 4)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def per_category_metrics(test_df: pd.DataFrame, y_pred: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    """Test-set metrics for each category separately, with sample counts and a reliability flag."""
    out: Dict[str, Dict[str, Any]] = {}
    y_true_all = test_df["sentiment_encoded"].to_numpy()
    y_pred_all = np.asarray(y_pred)
    categories = test_df["category"].to_numpy()
    for category in sorted(set(categories)):
        mask = categories == category
        y_true = y_true_all[mask]
        y_hat = y_pred_all[mask]
        support = {name: int((y_true == i).sum()) for i, name in enumerate(TARGET_NAMES)}
        full = compute_metrics(y_true, y_hat)
        reliable = min(support.values()) >= MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS
        out[category] = {
            "test_samples": int(mask.sum()),
            "class_support": support,
            "reliable": reliable,
            "accuracy": full["accuracy"],
            "macro_f1": full["macro_f1"],
            "negative_f1": full["negative_f1"],
            "neutral_f1": full["neutral_f1"],
            "positive_f1": full["positive_f1"],
            **(
                {}
                if reliable
                else {
                    "note": f"At least one class has fewer than {MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS} test reviews, "
                    "so these scores can swing a lot and should not be compared across categories."
                }
            ),
        }
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": {}, "per_category": {}}


def save_metrics(metrics_path: Path, data: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metrics_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(metrics_path)


def record_model_result(
    metrics_path: Path,
    model_key: str,
    model_result: Dict[str, Any],
    per_category: Optional[Dict[str, Dict[str, Any]]] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge one model's evaluation into metrics.json without touching the other models' results."""
    data = load_metrics(metrics_path)
    data.setdefault("models", {})[model_key] = model_result
    if per_category is not None:
        data.setdefault("per_category", {})[model_key] = per_category
    if split_info is not None:
        data["split_info"] = split_info
    data["updated_at"] = now_utc()
    save_metrics(metrics_path, data)
    return data
