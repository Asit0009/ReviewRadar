"""Drift detection: PSI maths and the decision rule."""

import numpy as np
import pandas as pd
import pytest

from reviewpulse.monitoring.drift_checker import PSI_ALERT, calculate_psi, run_drift_analysis


def frame(n=2000, mean=300.0, rating_p=(0.05, 0.05, 0.1, 0.2, 0.6), seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "text_length": rng.normal(mean, 80, n).clip(5),
            "rating": rng.choice([1, 2, 3, 4, 5], n, p=rating_p),
            "helpful_vote": rng.poisson(1.0, n),
        }
    )


def test_psi_is_zero_for_identical_distributions_and_grows_with_shift():
    x = np.random.default_rng(1).normal(0, 1, 5000)
    assert calculate_psi(x, x) == pytest.approx(0.0, abs=1e-6)
    shifted = calculate_psi(x, x + 1.0)
    assert shifted > PSI_ALERT
    assert calculate_psi(x, x + 0.1) < shifted


def test_psi_handles_empty_and_constant_input():
    assert calculate_psi(np.array([]), np.array([1.0])) == 0.0
    assert calculate_psi(np.ones(50), np.ones(50)) == 0.0  # all values identical: no bins to compare


def test_similar_samples_are_not_flagged():
    report = run_drift_analysis(frame(seed=1), frame(seed=2))
    assert report["status"] == "healthy" and report["drifted_features_count"] == 0 and report["total_features_evaluated"] == 3


def test_a_real_shift_is_flagged():
    report = run_drift_analysis(frame(seed=1), frame(mean=450, rating_p=(0.3, 0.2, 0.2, 0.2, 0.1), seed=2))
    assert report["status"] == "drift_detected"
    assert report["feature_metrics"]["text_length"]["drift_detected"] and report["feature_metrics"]["rating"]["drift_detected"]


def test_tiny_but_statistically_significant_difference_is_not_flagged():
    """With many reviews the KS p-value is ~0 for trivial differences. The effect-size rule keeps that from being an alarm."""
    big_ref, big_cur = frame(n=60000, mean=300, seed=1), frame(n=60000, mean=303, seed=2)
    report = run_drift_analysis(big_ref, big_cur)
    text = report["feature_metrics"]["text_length"]
    assert text["ks_pvalue"] < 0.05 and text["ks_statistic"] < 0.1
    assert text["drift_detected"] is False


def test_real_comparison_sorts_by_date_not_file_order(tmp_path):
    """The reference must be the OLDER reviews even when the file is not in date order."""
    df = frame(n=500, seed=3).assign(
        timestamp=np.random.default_rng(4).permutation(np.arange(500)) * 86_400_000,
    )
    df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    path = tmp_path / "r.parquet"
    df.to_parquet(path)
    report = run_drift_analysis(parquet_path=path)
    assert report["reference_sample_size"] == 350 and report["current_sample_size"] == 150
    assert report["reference_period"]["to"] <= report["current_period"]["from"]
    assert report["reference_period"]["from"] == df["review_date"].min() and report["current_period"]["to"] == df["review_date"].max()


def test_too_little_data_is_reported_as_unavailable(tmp_path):
    path = tmp_path / "tiny.parquet"
    frame(n=50).assign(timestamp=1, review_date="2023-01-01").to_parquet(path)
    assert run_drift_analysis(parquet_path=path)["status"] == "unavailable"
    assert run_drift_analysis(parquet_path=tmp_path / "missing.parquet")["status"] == "unavailable"
