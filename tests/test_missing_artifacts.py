"""A fresh clone has no data, models or index. The API must say so clearly and name the command that fixes it.

Nothing may answer with a 500 or a confusing error, and the commands in the messages must be the real ones.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reviewpulse.api.app import create_app
from reviewpulse.config import Settings
from reviewpulse.setup_steps import SETUP_COMMANDS, missing_steps


@pytest.fixture(scope="module")
def empty_client(tmp_path_factory):
    root = tmp_path_factory.mktemp("fresh_clone")
    settings = Settings(
        processed_dir=root / "processed", models_dir=root / "models", nlp_dir=root / "nlp", vector_store_dir=root / "vectors",
        frontend_dist_dir=root / "dist", categories=("Electronics",), gemini_api_key="",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        yield client


GET_ENDPOINTS = [
    "/api/v1/categories", "/api/v1/overview", "/api/v1/sentiment/trend", "/api/v1/ratings/distribution", "/api/v1/topics",
    "/api/v1/reviews", "/api/v1/models/metrics", "/api/v1/models/metrics/by-category", "/api/v1/monitoring/status",
]
POST_REQUESTS = [
    ("/api/v1/predict", {"text": "great product", "model_type": "logistic_regression"}),
    ("/api/v1/predict", {"text": "great product", "model_type": "linear_svc"}),
    ("/api/v1/predict", {"text": "great product", "model_type": "distilbert"}),
    ("/api/v1/assistant/chat", {"question": "what do people say about batteries?", "category": None, "top_k": 3}),
]


@pytest.mark.parametrize("url", GET_ENDPOINTS)
def test_get_endpoints_say_what_is_missing_and_how_to_build_it(empty_client, url):
    resp = empty_client.get(url)
    assert resp.status_code == 503, f"{url}: {resp.status_code} {resp.text[:200]}"
    assert "python -m reviewpulse." in resp.json()["detail"], f"{url} does not name a command: {resp.json()['detail']}"


@pytest.mark.parametrize("url,body", POST_REQUESTS)
def test_post_endpoints_say_what_is_missing_and_how_to_build_it(empty_client, url, body):
    resp = empty_client.post(url, json=body)
    assert resp.status_code == 503, f"{url}: {resp.status_code} {resp.text[:200]}"
    assert "python -m reviewpulse." in resp.json()["detail"]


def test_a_model_that_is_not_trained_is_not_replaced_by_another(empty_client):
    for model_type, step in [("logistic_regression", "sentiment_models"), ("distilbert", "distilbert_model")]:
        detail = empty_client.post("/api/v1/predict", json={"text": "great product", "model_type": model_type}).json()["detail"]
        assert SETUP_COMMANDS[step] in detail


def test_health_stays_up_and_lists_every_missing_step(empty_client):
    resp = empty_client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded" and not any(body["artifacts_ready"].values())
    assert body["setup_steps"] == {name: SETUP_COMMANDS[name] for name in SETUP_COMMANDS}
    assert body["total_reviews"] == 0 and body["retrieval"]["ready"] is False


def test_the_home_page_explains_how_to_build_the_dashboard(empty_client):
    resp = empty_client.get("/")
    assert resp.status_code == 200 and "npm run build" in resp.text


def test_health_has_no_setup_steps_when_everything_exists(client):
    """The shared test data set has data, classical models, topics and an index; only DistilBERT is absent."""
    body = client.get("/api/v1/health").json()
    assert set(body["setup_steps"]) == {"distilbert_model"}
    assert body["setup_steps"]["distilbert_model"] == SETUP_COMMANDS["distilbert_model"]


def test_missing_steps_helper():
    assert missing_steps({"parquet_data": True, "sentiment_models": False, "unknown": False}) == {"sentiment_models": SETUP_COMMANDS["sentiment_models"]}
    assert missing_steps({}) == {}


def test_setup_commands_run_in_build_order_and_name_real_modules():
    assert list(SETUP_COMMANDS) == ["parquet_data", "sentiment_models", "nmf_topics", "vector_search_faiss", "distilbert_model"]
    import importlib.util

    for command in SETUP_COMMANDS.values():
        for part in command.split("&&"):
            module = part.split("-m", 1)[1].split()[0]
            assert importlib.util.find_spec(module) is not None, f"{module} does not exist"


# ---------- locations are configurable ----------
def _config_paths(env):
    code = "import reviewpulse.config as c; print(c.DATA_DIR); print(c.DATA_RAW_DIR); print(c.MODELS_DIR); print(c.NLP_DIR); print(c.VECTOR_STORE_DIR); print(c.REPORTS_DIR); print(c.MANIFEST_PATH)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parent.parent), check=True)
    return [Path(line) for line in out.stdout.strip().splitlines()]


def test_data_artifact_and_report_folders_can_be_moved_with_environment_variables(tmp_path):
    import os

    env = {**os.environ, "REVIEWPULSE_DATA_DIR": str(tmp_path / "d"), "REVIEWPULSE_ARTIFACTS_DIR": str(tmp_path / "a"), "REVIEWPULSE_REPORTS_DIR": str(tmp_path / "r")}
    data, raw, models, nlp, vectors, reports, manifest = _config_paths(env)
    assert data == (tmp_path / "d").resolve() and raw == data / "raw" and manifest == data / "manifest.json"
    assert models == (tmp_path / "a").resolve() / "models" and nlp == (tmp_path / "a").resolve() / "nmf_topics" and vectors == (tmp_path / "a").resolve() / "vector_store"
    assert reports == (tmp_path / "r").resolve()


def test_default_locations_are_inside_the_repository_and_not_machine_specific(tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("REVIEWPULSE_")}
    repo = Path(__file__).resolve().parent.parent
    data, raw, models, nlp, vectors, reports, manifest = _config_paths(env)
    assert data == repo / "data" and models == repo / "artifacts" / "models" and reports == repo / "reports"
