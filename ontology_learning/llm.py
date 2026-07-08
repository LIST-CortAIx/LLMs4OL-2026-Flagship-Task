"""HTTP client for OpenAI-compatible inference servers.

``LLMClient``
    Thin HTTP wrapper around an OpenAI-compatible /v1/chat/completions endpoint.
    JSON mode via response_format.
    <think> tag stripping for reasoning models.

Backend profiles
----------------
The same client talks to several backends; a ``profile`` selects which payload
fields are sent (some are vLLM/Qwen extensions other servers reject) and how
reasoning is requested:

    qwen     (default) vLLM + Qwen3 — full sampling extras + thinking
             (chat_template_kwargs / thinking_token_budget). Our own model.
    gpt-oss  vLLM GPT-OSS — sampling extras + ``reasoning_effort`` (low/medium/high).
    vllm     generic vLLM — sampling extras, no model-specific reasoning.
    openai   generic OpenAI / LiteLLM (e.g. Minimax) — OpenAI-standard params only
             (no top_k/min_p/repetition_penalty); reasons natively, <think> stripped.

Usage
-----
    client = LLMClient(base_url="http://node07:8000/v1", model="Qwen/Qwen3-9B")
    result = client.chat_json([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import time
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)

# Per-backend capabilities. ``vllm_sampling`` = send top_k/min_p/repetition_penalty
# (vLLM extensions); ``reasoning`` = how to request reasoning.
PROFILES: dict[str, dict[str, Any]] = {
    "qwen":    {"vllm_sampling": True,  "reasoning": "qwen"},
    "gpt-oss": {"vllm_sampling": True,  "reasoning": "effort"},
    "vllm":    {"vllm_sampling": True,  "reasoning": None},
    "openai":  {"vllm_sampling": False, "reasoning": None},
}


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLMClient:
    """Thin HTTP wrapper around a vLLM /v1/chat/completions endpoint.

    Args:
        base_url: Base URL of the server (e.g. "http://node07:8000/v1").
        model: Model identifier as served by the backend (path or name).
        api_key: API key for the endpoint ("EMPTY" works for open local servers).
        temperature: Sampling temperature. 0.0 = deterministic.
        max_tokens: Maximum tokens in the response (includes reasoning tokens).
        timeout: HTTP timeout in seconds per request attempt.
        max_retries: Number of retry attempts on transient errors.
        json_mode: Request valid JSON output via response_format. Some non-vLLM
            backends reject it — set False there and rely on JSON extraction.
        enable_thinking: (qwen profile) allow Qwen3 thinking tokens. Set False for
            extraction to save budget. Passed as chat_template_kwargs.
        profile: Backend profile — see module docstring / ``PROFILES``.
        verify_ssl: Verify TLS certs. Set False for self-signed https endpoints
            (e.g. Minimax via LiteLLM).
        reasoning_effort: (gpt-oss profile) "low" | "medium" | "high".
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 1.0,
        max_tokens: int = 32768,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        timeout: float = 300.0,
        max_retries: int = 3,
        json_mode: bool = True,
        enable_thinking: bool = True,
        thinking_budget: int | None = None,
        profile: str = "qwen",
        verify_ssl: bool = True,
        reasoning_effort: str | None = None,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.profile = profile
        self.reasoning_effort = reasoning_effort
        self._caps = PROFILES[profile]
        self._ssl_ctx = None if verify_ssl else ssl._create_unverified_context()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat request and return the assistant text content.

        Raises:
            LLMError: After exhausting retries.
        """
        for attempt in range(self.max_retries):
            try:
                return self._call(messages)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"LLM call failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                sleep = min(2.0**attempt, 30.0)
                logger.warning(
                    "LLM attempt %d/%d failed (%s) — retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, sleep,
                )
                time.sleep(sleep)
        raise LLMError("Unreachable")

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send a chat request and return the response parsed as a JSON dict.

        Raises:
            LLMError: If the call fails or the response is not valid JSON.
        """
        text = self.chat(messages)
        return _extract_json(text)

    def _call(self, messages: list[dict[str, str]]) -> str:
        # OpenAI-standard fields — accepted by every backend.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
        }
        # vLLM sampling extensions — only on backends that accept them.
        if self._caps["vllm_sampling"]:
            payload["top_k"] = self.top_k
            payload["min_p"] = self.min_p
            payload["repetition_penalty"] = self.repetition_penalty
        # vLLM with reasoning_parser=qwen3 sets enable_in_reasoning=False, meaning
        # structured output constraints apply ONLY to the answer portion after </think>,
        # not to thinking tokens. json_object mode is therefore safe with thinking enabled.
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Reasoning — backend-specific.
        if self._caps["reasoning"] == "qwen":
            if not self.enable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            elif self.thinking_budget is not None:
                payload["thinking_token_budget"] = self.thinking_budget
        elif self._caps["reasoning"] == "effort" and self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        data = _do_request(req, self.timeout, self._ssl_ctx)
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        content = choice["message"]["content"]
        if finish_reason == "length" or content is None:
            raise RuntimeError(
                f"LLM response truncated or empty (finish_reason={finish_reason!r},"
                f" content={'null' if content is None else 'truncated'})"
            )
        return content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _do_request(req: request.Request, timeout: float,
                ssl_ctx: ssl.SSLContext | None = None) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))  # type: ignore[no-any-return]
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} | {body}") from exc


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output.

    Strips thinking blocks (complete and truncated), markdown fences, and
    leading prose before attempting to parse.
    """
    text = text.strip()

    # Strip complete thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip truncated thinking block (no closing tag — max_tokens exhausted mid-think)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Find and parse the first complete JSON object, ignoring any trailing content
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object found in response: {text[:200]!r}")
    try:
        result, _ = json.JSONDecoder().raw_decode(text, start)
        if isinstance(result, dict):
            return result  # type: ignore[no-any-return]
        raise LLMError(f"JSON is not a dict: {type(result)}")
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"JSON parse failed: {exc} | text: {text[start : start + 200]!r}"
        ) from exc
