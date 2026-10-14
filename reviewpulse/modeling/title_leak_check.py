"""How much did Amazon's automatic "Five Stars" titles help the sentiment models?

    python -m reviewpulse.modeling.title_leak_check          # about 2 minutes

When a reviewer writes no title, Amazon fills in the star rating as words. That is the label itself, so the cleaner
now leaves such titles out of the model text (`combined_text`). This script measures what the leak was worth, with
everything else held fixed: the same train/test split, the same TF-IDF settings and the same regularisation strength
(the one chosen by the real training run). Only the text differs:

  * "leaky" text   = automatic title + body, as the models used to see it
  * "clean" text   = body only (or a real title + body), as the models see it now

Results are written to reports/title_leak_experiment.json.
"""

import argparse
import json
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from reviewpulse.config import MODELS_DIR, REPORTS_DIR
from reviewpulse.modeling.dataset import load_and_split_data
from reviewpulse.pipeline.cleaner import is_default_rating_title

logger = logging.getLogger(__name__)
OUTPUT_PATH = REPORTS_DIR / "title_leak_experiment.json"


def with_leaky_text(df: pd.DataFrame) -> pd.Series:
    """The text the models saw before the fix: an automatic rating title is put back in front of the body."""
    titles = df["title"].fillna("")
    automatic = titles.map(is_default_rating_title)
    return pd.Series(np.where(automatic, titles.str.strip() + ". " + df["combined_text"], df["combined_text"]), index=df.index)


def run() -> Dict[str, Any]:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    chosen_c = {name: metrics["models"][name]["hyperparameters"]["C"] for name in ("logistic_regression", "linear_svc")}
    train, _, test, info = load_and_split_data()
    automatic_test = test["title"].fillna("").map(is_default_rating_title).to_numpy()
    y_test = test["sentiment_encoded"].to_numpy()

    rows = []
    for label, make_text in (("leaky (automatic title kept)", with_leaky_text), ("clean (automatic title removed)", lambda d: d["combined_text"])):
        vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=3)
        x_train = vectorizer.fit_transform(make_text(train).fillna(""))
        x_test = vectorizer.transform(make_text(test).fillna(""))
        for name, model in (
            ("logistic_regression", LogisticRegression(C=chosen_c["logistic_regression"], max_iter=1000, class_weight="balanced", random_state=42)),
            ("linear_svc", LinearSVC(C=chosen_c["linear_svc"], max_iter=3000, class_weight="balanced", random_state=42)),
        ):
            predicted = model.fit(x_train, train["sentiment_encoded"]).predict(x_test)
            rows.append(
                {
                    "text": label,
                    "model": name,
                    "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
                    "macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
                    "accuracy_on_reviews_with_automatic_title": round(float(accuracy_score(y_test[automatic_test], predicted[automatic_test])), 4),
                    "accuracy_on_other_reviews": round(float(accuracy_score(y_test[~automatic_test], predicted[~automatic_test])), 4),
                }
            )
            logger.info("%-34s %-20s accuracy %.4f  macro-F1 %.4f  (automatic-title reviews %.4f, others %.4f)", label, name, rows[-1]["accuracy"], rows[-1]["macro_f1"], rows[-1]["accuracy_on_reviews_with_automatic_title"], rows[-1]["accuracy_on_other_reviews"])

    result = {
        "split": {"train": len(train), "test": len(test), "split_strategy": info["split_strategy"]},
        "test_reviews_with_automatic_title": int(automatic_test.sum()),
        "test_reviews_with_automatic_title_pct": round(float(automatic_test.mean() * 100), 1),
        "regularisation_C": chosen_c,
        "results": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved %s", OUTPUT_PATH)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description="Measure how much Amazon's automatic rating titles helped the sentiment models (saves reports/title_leak_experiment.json).").parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
