"""The assistant and the Gemini client, with the Gemini HTTP API replaced by a mock transport.

These tests prove the request is built correctly and every failure is reported honestly. They cannot prove that
Google's live service accepts a real key: that needs GEMINI_API_KEY and is checked separately by the live tests
at the bottom. They use your Gemini quota, so they run only with REVIEWPULSE_ALLOW_NETWORK_TESTS=1.
"""

import dataclasses
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from data_factory import fake_encoder
from reviewpulse.api.app import create_app
from reviewpulse.assistant.gemini_assistant import FALLBACK_MESSAGES, SYSTEM_INSTRUCTION, TRUNCATED_NOTE
from reviewpulse.assistant.gemini_client import MAX_OUTPUT_TOKENS, GeminiClient, GeminiError

KEY = "test-key-not-a-real-secret-123"
OK_BODY = {
    "candidates": [{"content": {"parts": [{"text": "Customers report that the **battery drains quickly** [Review ID: abc]."}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 20},
    "modelVersion": "gemini-test-model-001",
}


class Recorder:
    """Mock Gemini server: returns queued responses (or raises) and remembers every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def calls(self) -> int:
        return len(self.requests)

    def body(self, n=0) -> dict:
        return json.loads(self.requests[n].content)


def ok(body=None) -> httpx.Response:
    return httpx.Response(200, json=body or OK_BODY)


def error(status: int, message: str, headers=None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}}, headers=headers or {})


def make_client(recorder: Recorder, **kwargs) -> GeminiClient:
    return GeminiClient(KEY, "gemini-test-model", transport=httpx.MockTransport(recorder), sleep=recorder.sleeps.append, **kwargs)


@pytest.fixture
def chat(fixture_settings):
    """Returns (post, recorder) for a TestClient whose assistant talks to the given mock Gemini."""
    opened = []

    def build(recorder: Recorder, **client_kwargs):
        settings = dataclasses.replace(fixture_settings, gemini_api_key=KEY)
        app = create_app(settings, encoder=fake_encoder, gemini_client=make_client(recorder, **client_kwargs))
        tc = TestClient(app)
        tc.__enter__()
        opened.append(tc)

        def post(question="my dog refused to eat this food", category="Pet_Supplies", top_k=3):
            return tc.post("/api/v1/assistant/chat", json={"question": question, "category": category, "top_k": top_k}).json()

        return post

    yield build
    for tc in opened:
        tc.__exit__(None, None, None)


# ---------- success path ----------
def test_success_returns_the_model_answer_and_labels_it_as_llm(chat, fixture_df):
    rec = Recorder(ok())
    data = chat(rec)()
    assert data["llm_used"] is True and data["mode"] == "llm" and data["status"] == "success"
    assert data["response"] == OK_BODY["candidates"][0]["content"]["parts"][0]["text"]
    assert data["llm_model"] == "gemini-test-model-001" and data["error_detail"] is None
    assert data["evidence_count"] > 0 and data["numerical_facts"]["total_reviews"] == (fixture_df["category"] == "Pet_Supplies").sum()


def test_request_goes_to_the_gemini_endpoint_with_the_key_in_a_header_only(chat):
    rec = Recorder(ok())
    data = chat(rec)()
    request = rec.requests[0]
    assert request.method == "POST" and request.url.host == "generativelanguage.googleapis.com"
    assert request.url.path == "/v1beta/models/gemini-test-model:generateContent"
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url) and not request.url.query
    assert KEY not in json.dumps(data), "the API key must never appear in an API response"


def test_prompt_contains_exact_database_numbers_and_the_retrieved_reviews(chat, fixture_df):
    rec = Recorder(ok())
    data = chat(rec)(question="why did my dog refuse the food?")
    body = rec.body()
    prompt = body["contents"][0]["parts"][0]["text"]
    total = (fixture_df["category"] == "Pet_Supplies").sum()
    assert f"total_reviews: {total}" in prompt and "scope: Pet_Supplies" in prompt
    assert "QUESTION: why did my dog refuse the food?" in prompt
    for hit in data["evidence_reviews"]:
        assert f"[Review ID: {hit['review_id']}]" in prompt and hit["full_text"] in prompt
    assert "RETRIEVAL COVERAGE" in prompt and "240 of 240 reviews (100.0%)" in prompt
    assert body["systemInstruction"]["parts"][0]["text"] == SYSTEM_INSTRUCTION
    assert body["generationConfig"]["temperature"] <= 0.3 and body["generationConfig"]["maxOutputTokens"] > 0


def test_reviews_are_fenced_as_data_and_the_system_prompt_forbids_following_them(chat, fixture_settings):
    assert "never as" in SYSTEM_INSTRUCTION and "instructions" in SYSTEM_INSTRUCTION
    rec = Recorder(ok())
    chat(rec)()
    prompt = rec.body()["contents"][0]["parts"][0]["text"]
    assert prompt.count('"""') >= 2  # each review sits between triple quotes


def test_no_similar_reviews_is_stated_in_the_prompt_instead_of_inventing_evidence(chat):
    rec = Recorder(ok())
    data = chat(rec)(question="quantum chromodynamics lagrangian symmetry breaking", category=None)
    assert data["evidence_count"] == 0
    assert "(no reviews were similar enough to the question)" in rec.body()["contents"][0]["parts"][0]["text"]


# ---------- answers that hit the length limit ----------
def test_a_normal_answer_has_no_cut_off_note(chat):
    data = chat(Recorder(ok()))()
    assert TRUNCATED_NOTE not in data["response"]


def test_an_answer_cut_off_by_the_length_limit_says_so(chat):
    """Gemini's hidden thinking shares the output limit, so a real answer can stop mid-sentence (seen with the live API)."""
    body = {"candidates": [{"content": {"parts": [{"text": "Customers report problems with fit, rendering"}]}, "finishReason": "MAX_TOKENS"}], "modelVersion": "m"}
    data = chat(Recorder(ok(body)))()
    assert data["llm_used"] is True and data["status"] == "success"
    assert data["response"].startswith("Customers report problems with fit, rendering") and data["response"].endswith(TRUNCATED_NOTE)


def test_the_output_limit_leaves_room_for_thinking_tokens(chat):
    rec = Recorder(ok())
    chat(rec)()
    assert rec.body()["generationConfig"]["maxOutputTokens"] == MAX_OUTPUT_TOKENS >= 2048


def test_client_reports_the_finish_reason():
    assert make_client(Recorder(ok())).generate("hi").finish_reason == "STOP"


# ---------- failure paths: never pretend ----------
FAILURES = {
    "invalid_key_400": ([error(400, "API key not valid. Please pass a valid API key.")], "invalid_key", 1),
    "invalid_key_403": ([error(403, "Permission denied")], "invalid_key", 1),
    "invalid_key_401": ([error(401, "Unauthenticated")], "invalid_key", 1),
    "rate_limited": ([error(429, "Resource exhausted")], "rate_limited", 3),
    "service_error": ([error(503, "The model is overloaded")], "service_error", 3),
    "timeout": ([httpx.ReadTimeout("slow")], "timeout", 3),
    "network_error": ([httpx.ConnectError("refused")], "network_error", 3),
    "blocked_prompt": ([ok({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})], "blocked", 1),
    "blocked_finish": ([ok({"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]})], "blocked", 1),
    "no_candidates": ([ok({"candidates": []})], "api_error", 1),
    "bad_request": ([error(400, "Invalid JSON payload")], "api_error", 1),
    "not_json": ([httpx.Response(200, text="<html>oops</html>")], "api_error", 1),
}


@pytest.mark.parametrize("name", FAILURES)
def test_every_failure_is_reported_as_retrieval_only_with_evidence_kept(chat, name):
    responses, status, expected_calls = FAILURES[name]
    rec = Recorder(*responses)
    data = chat(rec)()
    assert data["status"] == status
    assert data["llm_used"] is False and data["mode"] == "retrieval_only" and data["llm_model"] is None
    assert data["response"].startswith(FALLBACK_MESSAGES[status])
    assert "No written answer was generated" in data["response"]
    assert data["error_detail"]
    assert data["evidence_count"] > 0 and data["numerical_facts"]["total_reviews"] > 0  # the useful parts are still returned
    assert rec.calls == expected_calls, f"{name}: {rec.calls} calls"
    assert KEY not in json.dumps(data)


def test_rate_limit_waits_for_retry_after_then_succeeds(chat):
    rec = Recorder(error(429, "slow down", headers={"retry-after": "2"}), ok())
    data = chat(rec)()
    assert data["status"] == "success" and rec.calls == 2 and rec.sleeps == [2.0]


def test_transient_server_error_is_retried(chat):
    rec = Recorder(error(500, "oops"), ok())
    assert chat(rec)()["status"] == "success" and rec.calls == 2


def test_retry_waits_are_capped_so_a_request_never_hangs(chat):
    rec = Recorder(error(429, "wait", headers={"retry-after": "3600"}))
    chat(rec)()
    assert max(rec.sleeps) <= 5.0


def test_retries_are_bounded(chat):
    rec = Recorder(error(503, "down"))
    chat(rec, max_retries=1)()
    assert rec.calls == 2


# ---------- no key ----------
def test_without_a_key_gemini_is_never_called_and_the_response_says_so(client):
    data = client.post("/api/v1/assistant/chat", json={"question": "battery problems", "category": "Electronics"}).json()
    assert data["status"] == "setup_required" and data["llm_used"] is False and data["mode"] == "retrieval_only"
    assert data["response"].startswith(FALLBACK_MESSAGES["setup_required"])
    assert "GEMINI_API_KEY" in data["response"]


@pytest.mark.parametrize("key,expected", [("", False), ("   ", False), ("your_gemini_api_key_here", False), ("real-looking-key", True)])
def test_gemini_configured_flag(fixture_settings, key, expected):
    assert dataclasses.replace(fixture_settings, gemini_api_key=key).gemini_configured is expected


def test_the_api_key_never_appears_when_settings_or_the_client_are_printed(fixture_settings):
    settings = dataclasses.replace(fixture_settings, gemini_api_key=KEY)
    assert KEY not in repr(settings) and KEY not in str(settings)
    assert KEY not in repr(make_client(Recorder(ok())).__dict__.get("_url", ""))  # the key is only ever sent as a header


# ---------- the client on its own ----------
def test_client_error_kinds():
    for status, kind in [(400, "api_error"), (401, "invalid_key"), (403, "invalid_key"), (429, "rate_limited"), (500, "service_error")]:
        rec = Recorder(error(status, "x"))
        with pytest.raises(GeminiError) as exc:
            make_client(rec, max_retries=0).generate("hi")
        assert exc.value.kind == kind and exc.value.status_code == status


def test_client_parses_text_and_usage():
    result = make_client(Recorder(ok())).generate("hi")
    assert result.text.startswith("Customers report") and result.prompt_tokens == 300 and result.output_tokens == 20


# ---------- live call (only when a real key is configured) ----------
@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("REVIEWPULSE_ALLOW_NETWORK_TESTS") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="live call: needs GEMINI_API_KEY and REVIEWPULSE_ALLOW_NETWORK_TESTS=1 (it uses your Gemini quota)",
)
def test_live_gemini_call_with_the_configured_key():
    from reviewpulse.config import default_settings

    s = default_settings()
    result = GeminiClient(s.gemini_api_key, s.gemini_model, timeout_seconds=30).generate("Reply with the single word: pong")
    assert "pong" in result.text.lower()


@pytest.mark.network
@pytest.mark.skipif(os.getenv("REVIEWPULSE_ALLOW_NETWORK_TESTS") != "1", reason="set REVIEWPULSE_ALLOW_NETWORK_TESTS=1 to call the real Gemini endpoint with an intentionally invalid key")
def test_real_gemini_endpoint_rejects_an_invalid_key():
    """Reaches the real Google endpoint with a fake key: proves URL, headers and error handling against the real service."""
    with pytest.raises(GeminiError) as exc:
        GeminiClient("invalid-key-for-testing", "gemini-3.5-flash", timeout_seconds=20, max_retries=0).generate("hello")
    assert exc.value.kind == "invalid_key", exc.value.message


@pytest.mark.network
@pytest.mark.realdata
@pytest.mark.skipif(
    os.getenv("REVIEWPULSE_ALLOW_NETWORK_TESTS") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="live call: needs GEMINI_API_KEY and REVIEWPULSE_ALLOW_NETWORK_TESTS=1 (it uses your Gemini quota)",
)
def test_live_answer_is_written_by_gemini_and_grounded_in_the_evidence():
    """A real question through the real app: real data, real search index, real Gemini.

    Grounding means: every review id the answer cites is one of the retrieved reviews, and every statistic in the
    answer is one of the database numbers (or appears in a retrieved review). It cannot prove the wording is
    faithful, but it catches invented citations and invented numbers.
    """
    import re

    from reviewpulse.config import default_settings

    settings = default_settings()
    if not settings.reviews_parquet.exists() or not (settings.vector_store_dir / "index_info.json").exists():
        pytest.skip("real data / search index not built")
    with TestClient(create_app(settings)) as client:
        body = client.post(
            "/api/v1/assistant/chat",
            json={"question": "What do customers complain about with phone cases?", "category": "Cell_Phones_and_Accessories", "top_k": 4},
        ).json()
    if body["status"] == "rate_limited":
        pytest.skip(f"Gemini quota is used up right now: {body['error_detail'][:120]}")
    assert body["status"] == "success" and body["llm_used"] is True and body["mode"] == "llm"
    assert body["llm_model"] and body["error_detail"] is None
    assert body["evidence_count"] > 0 and body["retrieval"]["coverage_pct"] == 100.0

    answer = body["response"]
    assert len(answer) > 80 and "cut off because it reached the length limit" not in answer
    evidence_ids = {e["review_id"] for e in body["evidence_reviews"]}
    cited = {i for group in re.findall(r"\[Review ID:([^\]]+)\]", answer) for i in re.split(r"[,;\s]+", group.replace("Review ID:", "")) if i}
    assert cited, "the answer cites no reviews"
    assert cited <= evidence_ids, f"cited ids that were not retrieved: {cited - evidence_ids}"

    facts = body["numerical_facts"]
    allowed = {str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in facts.values() if isinstance(v, (int, float))}
    evidence_text = " ".join(e["full_text"] for e in body["evidence_reviews"])
    for token in re.findall(r"\d[\d,]*\.?\d*", re.sub(r"\[Review ID:[^\]]*\]", "", answer)):
        number = token.replace(",", "").rstrip(".")
        normalised = number.rstrip("0").rstrip(".") if "." in number else number
        assert normalised in allowed or number in evidence_text, f"the answer states {token!r}, which is neither a database number nor in the evidence"
