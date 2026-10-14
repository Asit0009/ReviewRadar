"""Training and evaluation of classical sentiment models (TF-IDF + LogReg, LinearSVC).

How the evaluation stays honest:
 * The text vectorizer is fitted on the training split only.
 * The regularisation strength C is chosen on the validation split.
 * The test split is scored once, after the choice, and those numbers are what the dashboard shows.

Experiments are tracked with MLflow and the models are saved to artifacts/models/.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from reviewpulse.config import MODEL_FIGURES_DIR, MODELS_DIR, ensure_directories
from reviewpulse.modeling.dataset import load_and_split_data
from reviewpulse.modeling.evaluation import (
    TARGET_NAMES,
    compute_metrics,
    load_metrics,
    now_utc,
    per_category_metrics,
    save_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Candidate regularisation strengths. Smaller C = stronger regularisation.
C_GRID: Tuple[float, ...] = (0.015625, 0.03125, 0.0625, 0.25, 1.0, 4.0, 16.0)

__all__ = ["compute_metrics", "train_and_evaluate_classical", "plot_confusion_matrix", "TARGET_NAMES"]


def plot_confusion_matrix(cm: np.ndarray, model_name: str, out_path: Path):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("True Sentiment (Rating-Derived)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def select_by_validation(
    make_model: Callable[[float], Any],
    X_train,
    y_train,
    X_val,
    y_val,
    grid: Sequence[float] = C_GRID,
) -> Tuple[Any, float, List[Dict[str, float]]]:
    """Fit one model per C and keep the one with the best validation macro-F1."""
    best_model, best_c, best_score = None, None, -1.0
    search: List[Dict[str, float]] = []
    for c in grid:
        model = make_model(c)
        model.fit(X_train, y_train)
        score = float(f1_score(y_val, model.predict(X_val), average="macro", zero_division=0))
        search.append({"C": c, "validation_macro_f1": round(score, 4)})
        logger.info("  C=%s -> validation macro-F1 %.4f", c, score)
        if score > best_score:
            best_model, best_c, best_score = model, c, score
    return best_model, float(best_c), search


def train_and_evaluate_classical() -> Dict:
    """Train TF-IDF + Logistic Regression and TF-IDF + LinearSVC, track in MLflow, write metrics.json."""
    ensure_directories()
    started = time.time()
    train_df, val_df, test_df, split_info = load_and_split_data()

    logger.info("Fitting TfidfVectorizer on training data only...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=3)
    X_train = vectorizer.fit_transform(train_df["combined_text"])
    X_val = vectorizer.transform(val_df["combined_text"])
    X_test = vectorizer.transform(test_df["combined_text"])
    y_train = train_df["sentiment_encoded"].values
    y_val = val_df["sentiment_encoded"].values
    y_test = test_df["sentiment_encoded"].values
    logger.info("TF-IDF feature matrix shape: Train=%s, Test=%s", X_train.shape, X_test.shape)

    mlflow.set_experiment("ReviewPulse_Sentiment_Classification")

    existing = load_metrics(MODELS_DIR / "metrics.json")
    all_metrics: Dict[str, Any] = {"split_info": split_info, "models": {}, "per_category": {}}
    # Keep an earlier DistilBERT result only if it was scored on exactly the same test reviews.
    old_split = existing.get("split_info", {})
    if "distilbert" in existing.get("models", {}) and old_split.get("test_ids_md5") == split_info.get("test_ids_md5"):
        all_metrics["models"]["distilbert"] = existing["models"]["distilbert"]
        if "distilbert" in existing.get("per_category", {}):
            all_metrics["per_category"]["distilbert"] = existing["per_category"]["distilbert"]

    specs = [
        (
            "logistic_regression",
            "Logistic Regression",
            "TF-IDF_LogisticRegression",
            lambda c: LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"),
            "confusion_matrix_lr.png",
            "logistic_regression.joblib",
        ),
        (
            "linear_svc",
            "LinearSVC",
            "TF-IDF_LinearSVC",
            lambda c: LinearSVC(C=c, max_iter=3000, class_weight="balanced", random_state=42),
            "confusion_matrix_svc.png",
            "linear_svc.joblib",
        ),
    ]

    fitted: Dict[str, Any] = {}
    for key, title, run_name, factory, figure_name, _ in specs:
        logger.info("Training %s (class_weight='balanced'); choosing C on the validation split...", title)
        with mlflow.start_run(run_name=run_name):
            model, best_c, search = select_by_validation(factory, X_train, y_train, X_val, y_val)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            val_metrics = compute_metrics(y_val, val_pred)
            test_metrics = compute_metrics(y_test, test_pred)

            mlflow.log_params({"model_type": title, "C": best_c, "class_weight": "balanced"})
            mlflow.log_metrics(
                {
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            fig_path = MODEL_FIGURES_DIR / figure_name
            plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), title, fig_path)
            mlflow.log_artifact(str(fig_path))

        all_metrics["models"][key] = {
            "validation": val_metrics,
            "test": test_metrics,
            "hyperparameters": {"C": best_c, "class_weight": "balanced", "search": search},
        }
        all_metrics["per_category"][key] = per_category_metrics(test_df, test_pred)
        fitted[key] = model

    all_metrics["trained_at"] = now_utc()
    all_metrics["updated_at"] = all_metrics["trained_at"]
    all_metrics["training_seconds"] = round(time.time() - started, 1)
    all_metrics["vectorizer"] = {"max_features": 25000, "ngram_range": [1, 2], "sublinear_tf": True, "min_df": 3, "vocabulary_size": len(vectorizer.vocabulary_)}

    # Save artifacts only after every evaluation succeeded, so a failed run never leaves half-new files.
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    for key, _, _, _, _, filename in specs:
        joblib.dump(fitted[key], MODELS_DIR / filename)
    save_metrics(MODELS_DIR / "metrics.json", all_metrics)

    lr, svc = all_metrics["models"]["logistic_regression"]["test"], all_metrics["models"]["linear_svc"]["test"]
    logger.info(
        "Evaluation Summary (Test Set):\n  Logistic Regression -> Accuracy: %.4f, Macro-F1: %.4f\n  LinearSVC           -> Accuracy: %.4f, Macro-F1: %.4f",
        lr["accuracy"], lr["macro_f1"], svc["accuracy"], svc["macro_f1"],
    )
    return all_metrics


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression and Linear SVC and write artifacts/models/metrics.json.").parse_args()  # no options; this makes --help print help instead of running the job
    train_and_evaluate_classical()
