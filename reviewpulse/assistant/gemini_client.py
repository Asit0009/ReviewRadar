"""Minimal client for the Gemini `generateContent` REST API.

Why REST with httpx instead of an SDK: the older `google-generativeai` package is deprecated, and a plain
HTTP call keeps timeouts, retries and error handling explicit and testable (tests inject an
httpx.MockTransport, so no network or real key is needed to exercise every failure path).

The API key is read from the server environment and is only ever sent in the `x-goog-api-key` header. It is
never put in a URL, a log line or an API response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# Newer Gemini models "think" before answering and that thinking counts against this limit (a few hundred to
# a thousand tokens). A small limit cut real answers off mid-sentence, so it is generous; answers stay short
# because the system prompt asks for a short paragraph or a few bullets.
MAX_OUTPUT_TOKENS = 4096


@dataclass
class GeminiResult:
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None  # "STOP" is normal; "MAX_TOKENS" means the answer was cut off


class GeminiError(Exception):
    """A failed Gemini call. `kind` matches the assistant status values in the API schema."""

    def __init__(self, kind: str, message: str, status_code: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


def _error_message(response: httpx.Response) -> str:
    """Pull the human-readable message out of Google's JSON error body without echoing anything sensitive."""
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        message = error.get("message")
        if isinstance(message, str) and message:
            return message[:300]
    except ValueError:
        pass
    return f"HTTP {response.status_code}"


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 25.0,
        max_retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = GEMINI_BASE_URL,
    ):
        self._api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._url = f"{base_url}/models/{model}:generateContent"
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds), transport=transport)

    def close(self) -> None:
        self._client.close()

    def generate(self, prompt: str, system_instruction: Optional[str] = None, max_output_tokens: int = MAX_OUTPUT_TOKENS, temperature: float = 0.2) -> GeminiResult:
        """Send one prompt. Retries rate limits, 5xx errors, timeouts and network errors a limited number of times."""
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

        last_error: Optional[GeminiError] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self._url, json=body, headers=headers)
            except httpx.TimeoutException:
                last_error = GeminiError("timeout", "Gemini did not answer in time.")
            except httpx.HTTPError as exc:  # connection refused, DNS failure, TLS problems...
                last_error = GeminiError("network_error", f"Could not reach Gemini ({type(exc).__name__}).")
            else:
                if response.status_code == 200:
                    return self._parse(response)
                last_error = self._classify(response)
                if last_error.kind not in ("rate_limited", "service_error"):
                    raise last_error  # invalid key, bad request, blocked...: retrying cannot help
            if attempt < self.max_retries:
                # Short, capped waits: this runs inside a web request, so it must not hold the user for long.
                wait = min(last_error.retry_after or (0.5 * 2**attempt), 5.0)
                logger.warning("Gemini call failed (%s); retrying in %.1fs", last_error.kind, wait)
                self._sleep(wait)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _classify(response: httpx.Response) -> GeminiError:
        code = response.status_code
        message = _error_message(response)
        if code == 429:
            return GeminiError("rate_limited", "Gemini rate limit reached. " + message, code, _retry_after_seconds(response))
        if code in (401, 403) or (code == 400 and "api key" in message.lower()):
            return GeminiError("invalid_key", "Gemini rejected the API key. " + message, code)
        if code >= 500:
            return GeminiError("service_error", f"Gemini had a server error. {message}", code)
        return GeminiError("api_error", f"Gemini returned an error. {message}", code)

    def _parse(self, response: httpx.Response) -> GeminiResult:
        try:
            data = response.json()
        except ValueError as exc:
            raise GeminiError("api_error", "Gemini returned a response that was not JSON.", response.status_code) from exc

        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise GeminiError("blocked", f"Gemini declined to answer ({feedback['blockReason']}).", response.status_code)
        candidates = data.get("candidates") or []
        if not candidates:
            raise GeminiError("api_error", "Gemini returned no answer.", response.status_code)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            reason = candidate.get("finishReason", "unknown")
            kind = "blocked" if reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION") else "api_error"
            raise GeminiError(kind, f"Gemini returned an empty answer (finish reason: {reason}).", response.status_code)
        usage = data.get("usageMetadata") or {}
        return GeminiResult(
            text=text,
            model=data.get("modelVersion") or self.model,
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=candidate.get("finishReason"),
        )
