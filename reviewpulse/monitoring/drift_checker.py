"""Statistical data drift detection using the Kolmogorov-Smirnov test and the Population Stability Index (PSI).

Question answered: do newer reviews look different from older reviews?

Reviews are sorted by date. The oldest 70% are the *reference* (what the models were built on) and the newest
30% are the *current* data. For each numeric feature we compare the two groups.

Decision rule (a feature is flagged as drifted when either holds):
  * PSI >= 0.2, a large shift in the distribution, or
  * the KS test is significant (p < 0.05) AND the KS statistic is at least 0.1.

Why not p-value alone? With tens of thousands of reviews the KS test becomes significant for differences far too
small to matter (a KS statistic of 0.02 is real but irrelevant). Requiring a minimum effect size avoids false alarms.

PSI interpretation: < 0.1 stable, 0.1 to 0.2 moderate change, >= 0.2 large change.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from reviewpulse.config import DATA_PROCESSED_DIR, ensure_directories

logger = logging.getLogger(__name__)

REFERENCE_FRACTION = 0.7
PSI_ALERT = 0.2
KS_ALPHA = 0.05
KS_MIN_EFFECT = 0.1  # minimum KS statistic for a significant difference to count as drift


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Population Stability Index between a reference and a current distribution."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Quantile bins from the reference distribution. Repeated edges (many identical values, e.g. 0 votes) are merged.
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.unique(np.percentile(expected, percentiles))
    if len(bins) < 2:
        return 0.0
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    eps = 1e-4  # keeps empty buckets from causing division by zero
    k = len(expected_counts)
    expected_pct = (expected_counts + eps) / (len(expected) + eps * k)
    actual_pct = (actual_counts + eps) / (len(actual) + eps * k)
    return round(float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))), 4)


def _compare(ref: np.ndarray, cur: np.ndarray, buckets: int, use_ks: bool = True) -> Dict[str, Any]:
    psi = calculate_psi(ref, cur, num_buckets=buckets)
    result: Dict[str, Any] = {"psi": psi}
    drifted = psi >= PSI_ALERT
    if use_ks:
        ks = ks_2samp(ref, cur)
        result.update(ks_statistic=round(float(ks.statistic), 4), ks_pvalue=round(float(ks.pvalue), 4))
        drifted = drifted or (ks.pvalue < KS_ALPHA and ks.statistic >= KS_MIN_EFFECT)
        result["metric"] = "KS_2sample + PSI"
    else:
        result["metric"] = "PSI"
    result["drift_detected"] = bool(drifted)
    return result


def run_drift_analysis(
    reference_df: Optional[pd.DataFrame] = None,
    current_df: Optional[pd.DataFrame] = None,
    parquet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare older and newer reviews. Pass two DataFrames to compare them directly (used by tests)."""
    ensure_directories()
    if reference_df is None or current_df is None:
        input_file = parquet_path or (DATA_PROCESSED_DIR / "reviews_all.parquet")
        if not Path(input_file).exists():
            return {"status": "unavailable", "message": "Processed Parquet file not found. Drift analysis requires data."}
        df = pd.read_parquet(input_file, columns=["text_length", "rating", "helpful_vote", "timestamp", "review_date"])
        if len(df) < 100:
            return {"status": "unavailable", "message": "Insufficient data points for drift check."}
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        split_idx = int(len(df) * REFERENCE_FRACTION)
        reference_df, current_df = df.iloc[:split_idx], df.iloc[split_idx:]

    features = {
        "text_length": _compare(reference_df["text_length"].to_numpy(), current_df["text_length"].to_numpy(), 10),
        "rating": _compare(reference_df["rating"].to_numpy(), current_df["rating"].to_numpy(), 5),
        "helpful_vote": _compare(reference_df["helpful_vote"].to_numpy(), current_df["helpful_vote"].to_numpy(), 5, use_ks=False),
    }
    drifted = sum(1 for f in features.values() if f["drift_detected"])

    report: Dict[str, Any] = {
        "status": "healthy" if drifted == 0 else "drift_detected",
        "reference_sample_size": int(len(reference_df)),
        "current_sample_size": int(len(current_df)),
        "drifted_features_count": drifted,
        "total_features_evaluated": len(features),
        "feature_metrics": features,
    }
    if "review_date" in reference_df.columns and len(reference_df) and len(current_df):
        report["reference_period"] = {"from": str(reference_df["review_date"].min()), "to": str(reference_df["review_date"].max())}
        report["current_period"] = {"from": str(current_df["review_date"].min()), "to": str(current_df["review_date"].max())}
    return report
