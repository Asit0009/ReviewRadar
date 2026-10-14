"""Pytest fixtures for the ReviewPulse test suite.

Most tests run against a small synthetic data set built by tests/data_factory.py (fast, and the right answers
are known). Tests marked `realdata` run against the real processed data and saved models and are skipped when
those files are not present.
"""

import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Make `import data_factory` work no matter where pytest is started from.
sys.path.insert(0, str(Path(__file__).parent))

import data_factory  # noqa: E402
from reviewpulse.api.app import create_app  # noqa: E402
from reviewpulse.config import Settings  # noqa: E402
from reviewpulse.pipeline.cleaner import clean_and_normalize_record  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "realdata: uses the real processed data / trained models (skipped if missing)")
    config.addinivalue_line("markers", "network: makes a real network request (skipped unless explicitly enabled)")


@pytest.fixture
def sample_raw_record():
    """Sample valid raw review record matching Amazon Reviews 2023 schema."""
    return {
        "rating": 5.0,
        "title": "Fantastic Sound Quality!",
        "text": "This audio cable works seamlessly. The connectors feel solid and well built.",
        "asin": "B000012345",
        "parent_asin": "B0000PARENT",
        "user_id": "U123456789",
        "timestamp": 1609459200000,
        "helpful_vote": 3,
        "verified_purchase": True,
    }


@pytest.fixture
def sample_processed_df(sample_raw_record):
    """Small sample DataFrame with standardized schema for unit testing."""
    records = []
    sentiments = [
        ("negative", 1.0, 0, "Terrible product, broke immediately after one day."),
        ("neutral", 3.0, 1, "It is okay, does the job but nothing extraordinary."),
        ("positive", 5.0, 2, "Absolutely love it, fantastic build quality and great value!"),
        ("positive", 4.0, 2, "Works well as described, good quality."),
        ("negative", 2.0, 0, "Poor performance and bad customer support."),
    ]
    for idx, (_, rating, _, text) in enumerate(sentiments):
        rec = sample_raw_record.copy()
        rec["rating"] = rating
        rec["text"] = text
        rec["title"] = f"Review {idx}"
        rec["asin"] = f"B0000{idx}"
        records.append(clean_and_normalize_record(rec, category="Musical_Instruments"))
    return pd.DataFrame(records)


# ---------- synthetic dataset ----------
@pytest.fixture(scope="session")
def fixture_settings(tmp_path_factory) -> Settings:
    return data_factory.write_dataset(tmp_path_factory.mktemp("reviewpulse_data"))


@pytest.fixture(scope="session")
def fixture_df(fixture_settings) -> pd.DataFrame:
    return pd.read_parquet(fixture_settings.reviews_parquet)


@pytest.fixture(scope="session")
def client(fixture_settings):
    """API client over the synthetic data, with a fake encoder and no Gemini key."""
    app = create_app(fixture_settings, encoder=data_factory.fake_encoder)
    with TestClient(app) as test_client:
        yield test_client


# ---------- real data ----------
@pytest.fixture(scope="session")
def real_settings() -> Settings:
    from reviewpulse.config import default_settings

    # The real key (if one is in .env) is blanked so no test can call, or be billed by, the live Gemini API.
    s = dataclasses.replace(default_settings(), gemini_api_key="")
    required = [s.reviews_parquet, s.metrics_json, s.models_dir / "logistic_regression.joblib", s.vector_store_dir / "index_info.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"real artifacts missing: {missing}")
    return s


@pytest.fixture(scope="session")
def real_client(real_settings):
    app = create_app(real_settings)
    with TestClient(app) as test_client:
        yield test_client
