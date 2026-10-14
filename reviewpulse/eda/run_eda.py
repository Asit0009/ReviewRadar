"""Exploratory Data Analysis (EDA) module for ReviewPulse.

Computes comprehensive statistics using DuckDB and Pandas over processed Parquet data,
generates publication-quality charts, and produces a plain-English markdown report.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from reviewpulse.config import (
    DATA_PROCESSED_DIR,
    EDA_FIGURES_DIR,
    EDA_REPORTS_DIR,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_plot_style():
    """Apply clean, modern styling to matplotlib charts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


def run_eda_analysis(parquet_path: Optional[Path] = None) -> Dict[str, Any]:
    """Execute full EDA, generate charts, and compile reports/eda/eda_report.md."""
    ensure_directories()
    input_file = parquet_path or (DATA_PROCESSED_DIR / "reviews_all.parquet")
    if not input_file.exists():
        raise FileNotFoundError(f"Processed file {input_file} not found. Run pipeline first.")

    logger.info(f"Running EDA on {input_file} using DuckDB and Pandas...")
    con = duckdb.connect()

    # 1. Overview counts
    total_reviews = con.execute(f"SELECT COUNT(*) FROM read_parquet('{input_file}')").fetchone()[0]
    total_products = con.execute(f"SELECT COUNT(DISTINCT parent_asin) FROM read_parquet('{input_file}')").fetchone()[0]
    total_users = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{input_file}')").fetchone()[0]
    date_min, date_max = con.execute(f"SELECT MIN(review_date), MAX(review_date) FROM read_parquet('{input_file}')").fetchone()

    # 2. Category distribution
    cat_df = con.execute(f"""
        SELECT 
            category, 
            COUNT(*) AS review_count,
            COUNT(DISTINCT parent_asin) AS unique_products,
            COUNT(DISTINCT user_id) AS unique_users,
            ROUND(AVG(rating), 2) AS mean_rating,
            ROUND(AVG(CAST(verified_purchase AS INT)) * 100, 1) AS verified_pct,
            ROUND(AVG(text_length), 1) AS mean_text_length,
            ROUND(AVG(helpful_vote), 2) AS mean_helpful_votes
        FROM read_parquet('{input_file}')
        GROUP BY category
        ORDER BY review_count DESC
    """).df()

    # 3. Rating & Sentiment distribution
    rating_df = con.execute(f"""
        SELECT 
            rating, 
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / {total_reviews}, 2) AS percentage
        FROM read_parquet('{input_file}')
        GROUP BY rating
        ORDER BY rating
    """).df()

    sentiment_df = con.execute(f"""
        SELECT 
            sentiment, 
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / {total_reviews}, 2) AS percentage
        FROM read_parquet('{input_file}')
        GROUP BY sentiment
        ORDER BY count DESC
    """).df()

    # 4. Verified purchase patterns
    verified_stats = con.execute(f"""
        SELECT 
            verified_purchase,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / {total_reviews}, 2) AS pct_of_total,
            ROUND(AVG(rating), 2) AS mean_rating,
            ROUND(AVG(helpful_vote), 2) AS mean_helpful
        FROM read_parquet('{input_file}')
        GROUP BY verified_purchase
    """).df()

    # 5. Review length statistics
    length_stats = con.execute(f"""
        SELECT 
            MIN(text_length) AS min_length,
            ROUND(AVG(text_length), 1) AS mean_length,
            MEDIAN(text_length) AS median_length,
            MAX(text_length) AS max_length,
            COUNT(CASE WHEN text_length = 0 THEN 1 END) AS empty_text_count,
            COUNT(CASE WHEN text_length > 2000 THEN 1 END) AS long_text_count
        FROM read_parquet('{input_file}')
    """).df().iloc[0].to_dict()

    # 6. Helpful vote patterns
    helpful_stats = con.execute(f"""
        SELECT 
            COUNT(CASE WHEN helpful_vote > 0 THEN 1 END) AS reviews_with_helpful_votes,
            ROUND(COUNT(CASE WHEN helpful_vote > 0 THEN 1 END) * 100.0 / {total_reviews}, 2) AS pct_with_helpful_votes,
            MAX(helpful_vote) AS max_helpful_votes,
            ROUND(AVG(helpful_vote), 2) AS avg_helpful_votes
        FROM read_parquet('{input_file}')
    """).df().iloc[0].to_dict()

    # 7. Metadata coverage
    meta_stats = con.execute(f"""
        SELECT 
            COUNT(CASE WHEN product_title IS NOT NULL AND product_title != '' THEN 1 END) AS reviews_with_product_title,
            ROUND(COUNT(CASE WHEN product_title IS NOT NULL AND product_title != '' THEN 1 END) * 100.0 / {total_reviews}, 2) AS product_title_coverage_pct,
            COUNT(CASE WHEN product_price IS NOT NULL AND product_price != '' THEN 1 END) AS reviews_with_product_price,
            ROUND(COUNT(CASE WHEN product_price IS NOT NULL AND product_price != '' THEN 1 END) * 100.0 / {total_reviews}, 2) AS product_price_coverage_pct
        FROM read_parquet('{input_file}')
    """).df().iloc[0].to_dict()

    # 8. Monthly review volume and mean rating over time (months with at least 20 reviews, so early sparse months do not add noise)
    time_df = con.execute(f"""
        SELECT 
            SUBSTR(review_date, 1, 7) AS year_month,
            COUNT(*) AS review_count,
            ROUND(AVG(rating), 2) AS avg_rating
        FROM read_parquet('{input_file}')
        GROUP BY year_month
        HAVING COUNT(*) >= 20
        ORDER BY year_month
    """).df()

    # 9. Top n-grams in positive vs negative reviews
    pos_samples = con.execute(f"""
        SELECT combined_text FROM read_parquet('{input_file}') 
        WHERE sentiment = 'positive' LIMIT 5000
    """).df()["combined_text"].tolist()

    neg_samples = con.execute(f"""
        SELECT combined_text FROM read_parquet('{input_file}') 
        WHERE sentiment = 'negative' LIMIT 5000
    """).df()["combined_text"].tolist()

    def get_top_ngrams(corpus, n_range=(1, 2), top_k=10):
        try:
            vec = CountVectorizer(ngram_range=n_range, stop_words="english", max_features=1000)
            X = vec.fit_transform(corpus)
            freqs = np.asarray(X.sum(axis=0)).ravel()
            words = vec.get_feature_names_out()
            sorted_indices = np.argsort(freqs)[::-1][:top_k]
            return [(words[idx], int(freqs[idx])) for idx in sorted_indices]
        except Exception:
            return []

    top_pos_ngrams = get_top_ngrams(pos_samples, (1, 2), 10)
    top_neg_ngrams = get_top_ngrams(neg_samples, (1, 2), 10)

    con.close()

    # ==================== Generate Charts ====================
    set_plot_style()

    # Chart 1: Rating Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(rating_df["rating"].astype(str), rating_df["count"], color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_title("Distribution of Star Ratings (1.0 to 5.0)", fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(EDA_FIGURES_DIR / "rating_distribution.png", dpi=200)
    plt.close(fig)

    # Chart 2: Category Volume
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=cat_df, y="category", x="review_count", palette="Blues_r", ax=ax)
    ax.set_title("Retained Reviews by Category", fontweight="bold")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Category")
    for i, v in enumerate(cat_df["review_count"]):
        ax.text(v + (max(cat_df["review_count"]) * 0.01), i, f"{v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(EDA_FIGURES_DIR / "reviews_by_category.png", dpi=200)
    plt.close(fig)

    # Chart 3: Sentiment Imbalance
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}
    ordered_colors = [colors.get(s, "#94a3b8") for s in sentiment_df["sentiment"]]
    ax.bar(sentiment_df["sentiment"], sentiment_df["count"], color=ordered_colors)
    ax.set_title("Rating-Derived Sentiment Class Distribution", fontweight="bold")
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Review Count")
    for i, row in sentiment_df.iterrows():
        ax.text(i, row["count"] + 50, f"{row['count']:,} ({row['percentage']}%)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(EDA_FIGURES_DIR / "sentiment_imbalance.png", dpi=200)
    plt.close(fig)

    # Chart 4: Reviews and Average Rating Over Time
    if not time_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax2 = ax1.twinx()
        x_indices = range(len(time_df))
        ax1.plot(x_indices, time_df["review_count"], color="#6366f1", marker="o", label="Review Count", linewidth=1.5)
        ax2.plot(x_indices, time_df["avg_rating"], color="#f59e0b", marker="s", label="Avg Rating", linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Year-Month")
        ax1.set_ylabel("Review Volume", color="#6366f1")
        ax2.set_ylabel("Average Star Rating", color="#f59e0b")
        step = max(1, len(time_df) // 10)
        ax1.set_xticks([i for i in x_indices[::step]])
        ax1.set_xticklabels(time_df["year_month"].iloc[::step], rotation=45, ha="right")
        ax1.set_title("Review Volume and Average Rating Over Time", fontweight="bold")
        fig.tight_layout()
        fig.savefig(EDA_FIGURES_DIR / "ratings_over_time.png", dpi=200)
        plt.close(fig)

    # Chart 5: Verified vs Unverified Impact
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Unverified", "Verified"]
    x = range(len(labels))
    ax.bar(x, verified_stats["mean_rating"], color=["#94a3b8", "#10b981"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Average Rating (Stars)")
    ax.set_title("Average Star Rating by Purchase Verification", fontweight="bold")
    for i, v in enumerate(verified_stats["mean_rating"]):
        ax.text(i, v + 0.1, f"{v:.2f} ★", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(EDA_FIGURES_DIR / "verified_purchase_impact.png", dpi=200)
    plt.close(fig)

    # ==================== Write Plain-English Markdown Report ====================
    pos_pct = sentiment_df[sentiment_df["sentiment"] == "positive"]["percentage"].values[0] if not sentiment_df[sentiment_df["sentiment"] == "positive"].empty else 0
    neg_pct = sentiment_df[sentiment_df["sentiment"] == "negative"]["percentage"].values[0] if not sentiment_df[sentiment_df["sentiment"] == "negative"].empty else 0
    neu_pct = sentiment_df[sentiment_df["sentiment"] == "neutral"]["percentage"].values[0] if not sentiment_df[sentiment_df["sentiment"] == "neutral"].empty else 0

    report_content = f"""# ReviewPulse: Exploratory Data Analysis (EDA) Report

**Dataset Source:** McAuley-Lab / Amazon Reviews 2023  
**Analysis Engine:** DuckDB, Pandas, PyArrow  
**Dataset Path:** `{input_file.as_posix()}`  
**Temporal Span:** {date_min} to {date_max}  

---

## 1. Executive Summary & Key Dataset Metrics

This report presents empirical findings from real Amazon Reviews 2023 data across {len(cat_df)} retail categories. All values reported below are computed directly from the validated and cleaned Parquet dataset.

| Metric | Computed Value | Description |
| :--- | :--- | :--- |
| **Total Cleaned Reviews** | **{total_reviews:,}** | Retained records after schema validation & deduplication |
| **Unique Products (Parent ASINs)** | **{total_products:,}** | Distinct products represented across all categories |
| **Unique Customer Accounts** | **{total_users:,}** | Distinct reviewer accounts |
| **Date Range** | **{date_min} to {date_max}** | Earliest to latest review submission |
| **Positive Class (4–5★)** | **{pos_pct}%** | Strong positive sentiment majority |
| **Negative Class (1–2★)** | **{neg_pct}%** | Critical reviews (primary signal for product defects) |
| **Neutral Class (3★)** | **{neu_pct}%** | Mixed or lukewarm sentiment |

---

## 2. Category-Level Distribution & Comparison

Each category exhibits unique rating behavior and verified purchase dynamics:

| Category | Reviews | Unique Products | Mean Rating | Verified % | Mean Length | Helpful Votes/Review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in cat_df.iterrows():
        report_content += (
            f"| `{row['category']}` | {row['review_count']:,} | {row['unique_products']:,} | "
            f"{row['mean_rating']}★ | {row['verified_pct']}% | {row['mean_text_length']:.0f} chars | "
            f"{row['mean_helpful_votes']} |\n"
        )

    report_content += f"""

![Review Volume by Category](figures/reviews_by_category.png)

### Key Observations:
1. **Balanced intake by design:** each category was sampled with the same limit ({int(cat_df['review_count'].min()):,} to {int(cat_df['review_count'].max()):,} reviews kept per category), so no single large category dominates. Reviews were taken from many small chunks spread over each source file, which gives many different reviewers ({total_users:,} in total) and years. It is still a sample, not a random draw from all of Amazon.
2. **Rating differences between categories:** the highest average rating is `{cat_df.sort_values('mean_rating').iloc[-1]['category']}` ({cat_df['mean_rating'].max()}★) and the lowest is `{cat_df.sort_values('mean_rating').iloc[0]['category']}` ({cat_df['mean_rating'].min()}★).
3. **Verified purchases:** between {cat_df['verified_pct'].min()}% and {cat_df['verified_pct'].max()}% of reviews per category are verified purchases.

---

## 3. Rating Distribution & Sentiment Class Imbalance

Star ratings form the basis for our derived sentiment classification target:
- **Negative (1–2 Stars):** {neg_pct}% of total records ({sentiment_df[sentiment_df['sentiment']=='negative']['count'].values[0]:,} reviews)
- **Neutral (3 Stars):** {neu_pct}% of total records ({sentiment_df[sentiment_df['sentiment']=='neutral']['count'].values[0]:,} reviews)
- **Positive (4–5 Stars):** {pos_pct}% of total records ({sentiment_df[sentiment_df['sentiment']=='positive']['count'].values[0]:,} reviews)

| Star Rating | Review Count | Percentage |
| :---: | :---: | :---: |
"""
    for _, row in rating_df.iterrows():
        report_content += f"| {row['rating']} Stars | {row['count']:,} | {row['percentage']}% |\n"

    report_content += f"""

![Star Rating Distribution](figures/rating_distribution.png)  
![Sentiment Class Imbalance](figures/sentiment_imbalance.png)

### Practical Modeling Implication:
Real-world e-commerce reviews suffer from significant **positive class imbalance** (over {pos_pct}% positive). A naive model predicting "positive" for every review would achieve {pos_pct}% accuracy while providing zero utility for detecting dissatisfied customers. Therefore, **Macro-F1** and per-class precision/recall must be the primary evaluation metrics rather than raw accuracy.

---

## 4. Review Text Length & Data Quality

| Length Metric | Computed Value |
| :--- | :--- |
| **Minimum Text Length** | {length_stats['min_length']} characters |
| **Median Text Length** | {length_stats['median_length']} characters |
| **Mean Text Length** | {length_stats['mean_length']} characters |
| **Maximum Text Length** | {length_stats['max_length']} characters |
| **Empty Reviews After Cleaning** | {length_stats['empty_text_count']} (Filtered out during validation) |
| **Long Reviews (>2000 chars)** | {length_stats['long_text_count']:,} |

Customer reviews exhibit a classic right-skewed distribution. The median length of {length_stats['median_length']} characters shows that most shoppers leave concise, 2-to-4 sentence impressions, while a small minority write comprehensive multi-paragraph essays.

---

## 5. Temporal Trends & Purchase Verification Impact

![Ratings Over Time](figures/ratings_over_time.png)  
![Verified vs Unverified Impact](figures/verified_purchase_impact.png)

- **Verification Impact:** Verified purchasers rate products at an average of **{verified_stats[verified_stats['verified_purchase']==True]['mean_rating'].values[0]}★**, compared to **{verified_stats[verified_stats['verified_purchase']==False]['mean_rating'].values[0]}★** for unverified submissions.
- **Helpful Votes Distribution:** {helpful_stats['pct_with_helpful_votes']}% of reviews have received at least 1 helpful vote, with the maximum recorded being {helpful_stats['max_helpful_votes']:,} votes.

---

## 6. Vocabulary & N-Gram Comparison

Top phrases identified via frequency analysis:

### Positive Reviews (Top Terms & Bigrams):
"""
    for phrase, count in top_pos_ngrams[:8]:
        report_content += f"- `{phrase}` ({count:,} occurrences)\n"

    report_content += """
### Negative Reviews (Top Terms & Bigrams):
"""
    for phrase, count in top_neg_ngrams[:8]:
        report_content += f"- `{phrase}` ({count:,} occurrences)\n"

    report_content += f"""
---

## 7. Conclusions & Modeling Strategy

1. **Class Balancing Strategy:** Given the {pos_pct}% / {neu_pct}% / {neg_pct}% split, we apply `class_weight='balanced'` in classical models (Logistic Regression and LinearSVC) to penalize minority-class errors proportionately.
2. **Feature Leakage Prevention:** Review star rating and metadata average ratings are strictly excluded from the feature matrix $X$. The models rely solely on text tokens.
3. **Evaluation Protocol:** Models will be compared using Macro-F1, per-class F1, and Confusion Matrices on a completely held-out test split.
"""

    report_path = EDA_REPORTS_DIR / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    summary_json_path = EDA_REPORTS_DIR / "eda_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_reviews": total_reviews,
                "total_products": total_products,
                "total_users": total_users,
                "date_min": str(date_min),
                "date_max": str(date_max),
                "length_stats": length_stats,
                "helpful_stats": helpful_stats,
                "meta_stats": meta_stats,
            },
            f,
            indent=2,
        )

    logger.info(f"EDA report successfully generated at {report_path}")
    return {"total_reviews": total_reviews, "report_path": str(report_path)}


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Exploratory data analysis: writes reports/eda/eda_report.md, eda_summary.json and the figures.").parse_args()  # no options; this makes --help print help instead of running the job
    run_eda_analysis()
