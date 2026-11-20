"""Ollama-first text generation with a bounded Gemini fallback.

No Streamlit or Google SDK dependency: both providers use their HTTP APIs.
Each request starts with Ollama, including after a previous fallback.
"""

import math
import os
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse

import requests


@dataclass(frozen=True)
class AISettings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    connect_timeout: float = 3.0
    ollama_timeout: float = 180.0
    gemini_timeout: float = 45.0
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str = field(default="", repr=False)

    @classmethod
    def from_mapping(cls, secrets=None):
        """Environment variables override optional Streamlit secrets."""
        secrets = secrets or {}

        def setting(name, default):
            return str(os.environ.get(name, secrets.get(name, default))).strip()

        def timeout(name, default):
            try:
                value = float(setting(name, default))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                return value
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a positive number of seconds.") from None

        return cls(
            ollama_base_url=setting("OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            ollama_model=setting("OLLAMA_MODEL", cls.ollama_model),
            connect_timeout=timeout("AI_CONNECT_TIMEOUT_SECONDS", cls.connect_timeout),
            ollama_timeout=timeout("OLLAMA_TIMEOUT_SECONDS", cls.ollama_timeout),
            gemini_timeout=timeout("GEMINI_TIMEOUT_SECONDS", cls.gemini_timeout),
            gemini_model=setting("GEMINI_MODEL", cls.gemini_model).removeprefix("models/"),
            gemini_api_key=setting("GEMINI_API_KEY", ""),
        )


@dataclass(frozen=True)
class AIResponse:
    text: str
    provider: str
    model: str
    fallback_reason: str = ""


class AIUnavailableError(RuntimeError):
    """Safe, actionable error when neither provider can answer."""


class ProviderError(RuntimeError):
    pass


def _post_json(url, *, payload, timeout, headers=None):
    # Never surface raw response bodies or exception URLs: they can contain keys.
    try:
        with requests.post(url, json=payload, timeout=timeout, headers=headers) as response:
            response.raise_for_status()
            data = response.json()
    except requests.Timeout:
        raise ProviderError("request timed out") from None
    except requests.ConnectionError:
        raise ProviderError("service could not be reached") from None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ProviderError(f"HTTP {status}") from None
    except (ValueError, requests.RequestException):
        raise ProviderError("invalid response or request configuration") from None
    if not isinstance(data, dict):
        raise ProviderError("invalid response format")
    return data


def _ollama(prompt, settings):
    try:
        endpoint = urlparse(settings.ollama_base_url)
    except ValueError:
        raise ProviderError("check OLLAMA_BASE_URL") from None
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or not settings.ollama_model:
        raise ProviderError("check OLLAMA_BASE_URL and OLLAMA_MODEL")
    data = _post_json(
        settings.ollama_base_url + "/api/generate",
        payload={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2m",
            "options": {"temperature": 0.2, "num_predict": 256, "num_ctx": 4096},
        },
        timeout=(settings.connect_timeout, settings.ollama_timeout),
    )
    answer = data.get("response")
    if data.get("error") or data.get("done") is not True or not isinstance(answer, str) or not answer.strip():
        raise ProviderError("model returned no usable answer")
    return answer.strip()


def _gemini(prompt, settings):
    if not settings.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY is not configured")
    if not settings.gemini_model:
        raise ProviderError("GEMINI_MODEL is not configured")
    model = quote(settings.gemini_model.removeprefix("models/"), safe="")
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": settings.gemini_api_key},
        payload={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        timeout=(settings.connect_timeout, settings.gemini_timeout),
    )
    try:
        candidate = data["candidates"][0]
        if candidate.get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            raise ProviderError("model did not return a usable answer")
        parts = candidate["content"]["parts"]
        answer = "\n".join(
            part["text"] for part in parts
            if isinstance(part.get("text"), str) and not part.get("thought", False)
        ).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("model returned no usable answer") from None
    if not answer:
        raise ProviderError("model returned no usable answer")
    return answer


def generate_response(prompt, settings):
    """Try Ollama once, then Gemini once; never cache an outage or an answer."""
    try:
        return AIResponse(_ollama(prompt, settings), "Ollama", settings.ollama_model)
    except ProviderError as exc:
        fallback_reason = str(exc)
    try:
        return AIResponse(_gemini(prompt, settings), "Gemini", settings.gemini_model, fallback_reason)
    except ProviderError as exc:
        raise AIUnavailableError(
            f"Ollama: {fallback_reason}. Gemini fallback: {exc}. "
            "Check Docker/Ollama and the configured model, or the Gemini API key, "
            "model and internet connection, then send your question again."
        ) from None
