"""Loading saved sentiment models and running predictions.

Classical models (TF-IDF + Logistic Regression / LinearSVC) load at start-up because they are small.
DistilBERT is loaded the first time someone asks for it, since it needs a few hundred MB of memory.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from reviewpulse.heavy_imports import HEAVY_IMPORT_LOCK
from reviewpulse.setup_steps import SETUP_COMMANDS

logger = logging.getLogger(__name__)

LABELS = ["negative", "neutral", "positive"]  # index = sentiment_encoded used in training
DISTILBERT_DIR_NAME = "distilbert_sentiment"
DISTILBERT_MAX_LENGTH = 128  # must match the length used when fine-tuning
DISTILBERT_RETRY_SECONDS = 60  # after a failed load, wait this long before trying again


class ModelUnavailable(RuntimeError):
    """The requested model has no saved artifacts (or could not be loaded)."""


class SentimentPredictors:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.lr_model = None
        self.svc_model = None
        self._distilbert: Optional[Dict[str, Any]] = None
        self._distilbert_error: Optional[str] = None
        self._distilbert_failed_at = 0.0
        self._distilbert_lock = threading.Lock()
        self._load_classical()

    # ---------- loading ----------
    def _load_joblib(self, name: str):
        path = self.models_dir / name
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:  # a corrupt file should disable one model, not the whole API
            logger.warning("Could not load %s: %s", path, exc)
            return None

    def _load_classical(self) -> None:
        self.vectorizer = self._load_joblib("tfidf_vectorizer.joblib")
        self.lr_model = self._load_joblib("logistic_regression.joblib")
        self.svc_model = self._load_joblib("linear_svc.joblib")

    @property
    def distilbert_dir(self) -> Path:
        return self.models_dir / DISTILBERT_DIR_NAME

    def distilbert_artifacts_present(self) -> bool:
        d = self.distilbert_dir
        return (d / "config.json").exists() and any((d / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))

    def available_models(self) -> List[str]:
        names: List[str] = []
        if self.vectorizer is not None and self.lr_model is not None:
            names.append("logistic_regression")
        if self.vectorizer is not None and self.svc_model is not None:
            names.append("linear_svc")
        if self.distilbert_artifacts_present():
            names.append("distilbert")
        return names

    @property
    def classical_ready(self) -> bool:
        return self.vectorizer is not None and self.lr_model is not None

    # ---------- prediction ----------
    def predict(self, text: str, model_type: str) -> Dict[str, Any]:
        if model_type == "distilbert":
            return self._predict_distilbert(text)
        if model_type not in ("logistic_regression", "linear_svc"):
            raise ValueError(f"Unsupported model_type: {model_type!r}")
        model = self.lr_model if model_type == "logistic_regression" else self.svc_model
        if self.vectorizer is None or model is None:
            raise ModelUnavailable(f"The {model_type} model has not been trained yet. Run: {SETUP_COMMANDS['sentiment_models']}")

        features = self.vectorizer.transform([text])
        encoded = int(model.predict(features)[0])
        probabilities: Optional[Dict[str, float]] = None
        if hasattr(model, "predict_proba"):
            row = model.predict_proba(features)[0]
            by_class = {int(c): float(p) for c, p in zip(model.classes_, row)}
            probabilities = {LABELS[i]: round(by_class.get(i, 0.0), 4) for i in range(3)}
        label = "LogisticRegression" if model_type == "logistic_regression" else "LinearSVC"
        return {
            "model_type": model_type,
            "predicted_sentiment": LABELS[encoded],
            "sentiment_encoded": encoded,
            "probabilities": probabilities,
            "model_used": f"TF-IDF + {label}",
        }

    def _load_distilbert(self) -> Dict[str, Any]:
        with self._distilbert_lock:
            if self._distilbert is not None:
                return self._distilbert
            if not self.distilbert_artifacts_present():
                raise ModelUnavailable(f"DistilBERT has not been trained yet. Run: {SETUP_COMMANDS['distilbert_model']}")
            # A failed load is remembered only briefly: one bad moment must not switch the model off until a restart.
            if self._distilbert_error is not None and time.monotonic() - self._distilbert_failed_at < DISTILBERT_RETRY_SECONDS:
                raise ModelUnavailable(self._distilbert_error)
            try:
                with HEAVY_IMPORT_LOCK:  # one thread at a time does the first torch/transformers import
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(str(self.distilbert_dir))
                    model = AutoModelForSequenceClassification.from_pretrained(str(self.distilbert_dir))
                    model.eval()
                self._distilbert = {"tokenizer": tokenizer, "model": model, "torch": torch}
                self._distilbert_error = None
                logger.info("Loaded DistilBERT from %s", self.distilbert_dir)
                return self._distilbert
            except Exception as exc:
                self._distilbert_error = f"DistilBERT could not be loaded: {exc}"
                self._distilbert_failed_at = time.monotonic()
                logger.warning(self._distilbert_error)
                raise ModelUnavailable(self._distilbert_error) from exc

    def warm_up(self) -> None:
        """Load DistilBERT now (if it was trained) so the first prediction does not wait for the import."""
        if self.distilbert_artifacts_present():
            self._load_distilbert()

    def _predict_distilbert(self, text: str) -> Dict[str, Any]:
        bundle = self._load_distilbert()
        torch = bundle["torch"]
        encoded_input = bundle["tokenizer"](text, truncation=True, max_length=DISTILBERT_MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = bundle["model"](**encoded_input).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        encoded = int(max(range(3), key=lambda i: probs[i]))
        return {
            "model_type": "distilbert",
            "predicted_sentiment": LABELS[encoded],
            "sentiment_encoded": encoded,
            "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
            "model_used": "DistilBERT (distilbert-base-uncased, fine-tuned)",
        }
