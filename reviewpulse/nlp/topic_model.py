"""Interpretable topic discovery using TF-IDF + NMF (Non-negative Matrix Factorization).

NMF splits the review-by-word TF-IDF matrix into two smaller non-negative matrices:
  * topic-by-word weights (H): which words define each topic, and
  * review-by-topic weights (W): how much of each topic a review contains.
Because every weight is non-negative, each topic reads like a list of words that tend to appear together,
which is easy to explain.

Outputs (in artifacts/nmf_topics/):
  topic_labels.json     topics with top words, example reviews and overall prevalence
  review_topics.parquet one row per review: its dominant topic and how strong that topic was
  nmf_vectorizer.joblib / nmf_model.joblib   so new text can be assigned to a topic later
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from reviewpulse.config import DATA_PROCESSED_DIR, NLP_DIR, ensure_directories

logger = logging.getLogger(__name__)

# Words that are common in every review and would otherwise dominate the topics.
DOMAIN_STOP_WORDS = frozenset(
    {
        "product", "products", "item", "items", "amazon", "bought", "buy", "buying", "purchase", "purchased",
        "ordered", "order", "one", "just", "like", "use", "used", "really", "good", "great", "well", "time",
        "would", "get", "got", "much", "also", "stars", "star", "love", "nice", "ok", "okay", "works", "work",
        "thing", "things", "am", "ve", "ll", "don", "didn", "doesn", "isn", "wasn", "won", "can", "will",
        # Everyday words that otherwise glue most reviews into one big, uninformative topic.
        "does", "did", "best", "better", "bit", "lot", "way", "made", "make", "going", "say", "said", "want",
        "wanted", "need", "needed", "little", "come", "came", "tried", "try", "found", "bad", "pretty", "day", "days",
    }
)

# Words that only praise or thank. They say how the reviewer feels, not what the review is about, and they show up
# in the top words of almost every topic. Removing them made topics about the product, not about the mood
# (see docs/TOPIC_REPORT.md for the comparison).
PRAISE_STOP_WORDS = frozenset(
    {
        "excellent", "awesome", "amazing", "happy", "thank", "thanks", "recommend", "highly", "definitely",
        "loved", "loves", "super", "absolutely", "wonderful", "fantastic",
    }
)

DEFAULT_N_TOPICS = 20  # 14 left one catch-all topic with a third of all reviews; 20 splits off real product themes
DEFAULT_MAX_FEATURES = 8000  # the most frequent words and phrases only: rare ones add noise, not themes
DEFAULT_MIN_DF = 20  # a word or phrase must appear in at least this many reviews
MIN_WORDS_TO_FIT = 8  # topics are learned from reviews with at least this many words, then applied to every review
MIN_TOPIC_SHARE_TO_TAG = 0.0  # tag every review that has any topic signal at all


# Readable names. Each one was chosen by reading the real reviews of the topic it describes (see
# reports/topic_report.md), not from the word list alone. A name is only used when ALL its signature words are among
# the topic's top 10 words, so after retraining with different data a name can never be attached to the wrong topic.
# The topic then simply keeps its automatic label made of top words. Order matters: the first matching rule wins.
TOPIC_NAMES: Sequence = (
    ({"screen", "protector"}, "Screen protectors"),
    ({"charge", "battery"}, "Phones, chargers and batteries"),
    ({"case", "iphone"}, "Phone cases"),
    ({"using", "long", "worked"}, "General use and results (mixed products)"),
    ({"hair", "shampoo"}, "Hair and skin care"),
    ({"dog", "dogs"}, "Dogs, cats and pet supplies"),
    ({"fun", "game"}, "Games, toys and fun for kids"),
    ({"gift", "christmas"}, "Gifts"),
    ({"cute", "adorable"}, "Cute products (kids, baby, pets)"),
    ({"old", "year"}, "Age and years (\"5 year old\", \"a year\")"),
    ({"money", "waste"}, "Worth the money or a waste"),
    ({"price", "value"}, "Price and value"),
    ({"easy", "install"}, "Easy to install and use"),
    ({"quality", "poor"}, "Quality (good or poor)"),
    ({"exactly", "described"}, "Exactly as described"),
    ({"perfect", "perfect size"}, "Perfect size and fit"),
    ({"fit", "car"}, "Fit for cars and vehicles"),
    ({"small", "size"}, "Size: too small or too big"),
    ({"looks", "sturdy"}, "Looks and feel"),
    ({"color", "colors"}, "Colors and brightness"),
)


def curated_name(top_words: Sequence[str]) -> Optional[str]:
    """The reviewed name for a topic, or None if no rule matches its top words."""
    words = set(top_words[:10])
    for signature, name in TOPIC_NAMES:
        if signature <= words:
            return name
    return None


def _top_words(feature_names: Sequence[str], topic_vector: np.ndarray, count: int) -> List[str]:
    """The highest-weighted words and phrases of a topic, without "phrases" that just repeat one word ("case case").

    Such pairs appear when a reviewer writes the same word twice ("cute cute cute!"). They add nothing to the word
    list, so they are skipped (they still take part in the model itself).
    """
    words: List[str] = []
    for index in np.argsort(topic_vector)[::-1]:
        word = feature_names[index]
        parts = word.split()
        if len(parts) == 2 and parts[0] == parts[1]:
            continue
        words.append(word)
        if len(words) == count:
            break
    return words


def _example_reviews(texts: pd.Series, word_counts: np.ndarray, topic_weights: np.ndarray, members: np.ndarray, how_many: int = 2, min_words: int = 12) -> List[str]:
    """Reviews that show the topic best: the strongest members, each with different text.

    Reviews of at least `min_words` words are preferred because "Nice price" says nothing about a topic. Short ones are
    used only if the topic has too few longer ones.
    """
    member_ids = np.where(members)[0]
    strongest_first = member_ids[np.argsort(topic_weights[member_ids])[::-1]]
    preferred = [i for i in strongest_first if word_counts[i] >= min_words]
    chosen: List[str] = []
    seen = set()
    for i in preferred + list(strongest_first):
        excerpt = _excerpt(texts.iloc[i])
        key = excerpt.lower()  # compare what will be shown: two reviews can share their first lines and differ later
        if key in seen:
            continue
        seen.add(key)
        chosen.append(excerpt)
        if len(chosen) == how_many:
            break
    return chosen


def make_label(top_words: Sequence[str], count: int = 3) -> str:
    """Name a topic after its top words, skipping words that repeat another chosen word ("battery", "battery life")."""
    chosen: List[str] = []
    for word in top_words:
        if any(word in c or c in word for c in chosen):
            continue
        chosen.append(word)
        if len(chosen) == count:
            break
    return " & ".join(w.title() for w in chosen)


def _excerpt(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]  # do not cut in the middle of a word
    return cut + "..."


def train_nmf_topic_model(
    parquet_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_topics: int = DEFAULT_N_TOPICS,
    max_features: int = DEFAULT_MAX_FEATURES,
    top_words_count: int = 10,
    df: Optional[pd.DataFrame] = None,
    min_words_to_fit: int = MIN_WORDS_TO_FIT,
) -> Dict[str, Any]:
    """Train the topic model on all reviews, tag every review, and save everything."""
    ensure_directories()
    out_dir = Path(output_dir) if output_dir else NLP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        input_file = Path(parquet_path) if parquet_path else DATA_PROCESSED_DIR / "reviews_all.parquet"
        if not input_file.exists():
            raise FileNotFoundError(f"Processed file {input_file} not found. Run the pipeline first.")
        df = pd.read_parquet(input_file)
    df = df.reset_index(drop=True)
    logger.info("Training TF-IDF + NMF on %s reviews with n_topics=%s...", f"{len(df):,}", n_topics)

    stop_words = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS | PRAISE_STOP_WORDS)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=stop_words,
        min_df=min(DEFAULT_MIN_DF, max(1, len(df) // 50)),  # the cap keeps tiny test data sets usable
        max_df=0.5,
    )
    texts = df["combined_text"].fillna("")
    # Very short reviews ("Great product") carry almost no topic information. If they are included, one big
    # catch-all topic soaks them up. So the topics are learned from longer reviews only and then applied to all.
    word_counts = df["word_count"] if "word_count" in df.columns else texts.str.split().str.len()
    fit_mask = (word_counts >= min_words_to_fit).to_numpy()
    if fit_mask.sum() < max(50, n_topics * 5):
        fit_mask[:] = True  # too little data to be picky (tiny test data sets)
    vectorizer.fit(texts[fit_mask])
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    nmf.fit(vectorizer.transform(texts[fit_mask]))
    H = nmf.components_  # topic x word
    W = nmf.transform(vectorizer.transform(texts))  # review x topic, for EVERY review

    row_sums = W.sum(axis=1)
    has_signal = row_sums > 0
    shares = np.divide(W, row_sums[:, None], out=np.zeros_like(W), where=has_signal[:, None])
    dominant = shares.argmax(axis=1)
    dominant_share = shares.max(axis=1)
    tagged = has_signal & (dominant_share >= MIN_TOPIC_SHARE_TO_TAG)

    topics_info: List[Dict[str, Any]] = []
    for topic_idx, topic_vec in enumerate(H):
        top_words = _top_words(feature_names, topic_vec, top_words_count)
        members = (dominant == topic_idx) & tagged
        examples = _example_reviews(texts, word_counts.to_numpy(), W[:, topic_idx], members)
        auto_label = make_label(top_words)
        named = curated_name(top_words)
        topics_info.append(
            {
                "topic_id": int(topic_idx),
                "label": named or auto_label,  # what the dashboard shows
                "auto_label": auto_label,  # always the top-word label, kept for transparency
                "named_by": "reviewed" if named else "top words",
                "top_words": top_words,
                "sample_reviews": examples,
                "prevalence_count": int(members.sum()),
                "prevalence_pct": round(float(members.sum() / max(1, tagged.sum()) * 100), 2),  # % of TAGGED reviews, like the API
            }
        )
        logger.info("Topic %s (%s%%): %s", topic_idx, topics_info[-1]["prevalence_pct"], ", ".join(top_words[:6]))

    review_topics = pd.DataFrame(
        {
            "review_id": df["review_id"].astype(str),
            "topic_id": dominant.astype(np.int32),
            "topic_score": dominant_share.astype(np.float32),
        }
    )[tagged]
    if review_topics["review_id"].duplicated().any():
        raise ValueError("review_id must be unique to tag reviews with topics.")

    tagged_df = df[tagged].assign(dominant_topic=dominant[tagged])
    category_dist = pd.crosstab(tagged_df["category"], tagged_df["dominant_topic"], normalize="index").round(4)
    sentiment_dist = pd.crosstab(tagged_df["sentiment"], tagged_df["dominant_topic"], normalize="index").round(4)

    results: Dict[str, Any] = {
        "n_topics": n_topics,
        "total_reviews": int(len(df)),
        "tagged_reviews": int(tagged.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {"max_features": max_features, "ngram_range": [1, 2], "min_df": vectorizer.min_df, "max_df": 0.5, "nmf_init": "nndsvda", "min_words_to_fit": min_words_to_fit, "reviews_used_to_fit": int(fit_mask.sum())},
        "topics": topics_info,
        "category_distribution": {str(k): v for k, v in category_dist.to_dict().items()},
        "sentiment_distribution": {str(k): v for k, v in sentiment_dist.to_dict().items()},
    }

    joblib.dump(vectorizer, out_dir / "nmf_vectorizer.joblib")
    joblib.dump(nmf, out_dir / "nmf_model.joblib")
    review_topics.to_parquet(out_dir / "review_topics.parquet", index=False)
    (out_dir / "topic_labels.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved %s topics and %s review tags to %s", n_topics, f"{len(review_topics):,}", out_dir)
    return results


def assign_topics(texts: Sequence[str], model_dir: Optional[Path] = None) -> List[Optional[Dict[str, float]]]:
    """Assign new text to its dominant topic using the saved model (None when no known words appear)."""
    d = Path(model_dir) if model_dir else NLP_DIR
    vectorizer = joblib.load(d / "nmf_vectorizer.joblib")
    nmf = joblib.load(d / "nmf_model.joblib")
    W = nmf.transform(vectorizer.transform(list(texts)))
    out: List[Optional[Dict[str, float]]] = []
    for row in W:
        total = row.sum()
        out.append(None if total <= 0 else {"topic_id": int(row.argmax()), "topic_score": float(row.max() / total)})
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train the NMF topic model and tag every review with a topic.")
    parser.add_argument("--n-topics", type=int, default=DEFAULT_N_TOPICS)
    args = parser.parse_args()
    train_nmf_topic_model(n_topics=args.n_topics)
