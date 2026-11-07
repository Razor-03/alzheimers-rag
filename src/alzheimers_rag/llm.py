"""Small synchronous HTTP adapters. No SDK, model loading, retries or tool execution."""

import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMConfig


class ChatModel(Protocol):
    def complete(self, messages: list[dict], *, config: LLMConfig) -> str: ...


def _post_json(url, payload, headers, timeout):
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        # Do not print provider bodies, which may echo credentials or evidence.
        raise RuntimeError(f"LLM request failed (HTTP {exc.code})") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"LLM service unavailable ({type(exc).__name__})") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("LLM service returned invalid JSON") from None


class HTTPChatModel:
    def complete(self, messages, *, config):
        headers = {}
        key = os.getenv(config.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": config.model, "messages": messages, "stream": False}
        if config.provider == "ollama":
            endpoint = "/api/chat"
            payload.update(
                format="json",
                options={
                    "num_ctx": config.context_window,
                    "num_predict": config.max_output_tokens,
                    "temperature": 0,
                },
            )
        else:
            endpoint = "/chat/completions"  # base_url includes /v1 when required
            payload.update(
                max_completion_tokens=config.max_output_tokens,
                response_format={"type": "json_object"},
            )
        response = _post_json(
            config.base_url.rstrip("/") + endpoint, payload, headers, config.timeout
        )
        try:
            if config.provider == "ollama":
                if response.get("error") or response.get("done") is not True:
                    raise ValueError
                reason = response.get("done_reason")
                content = response["message"]["content"]
            else:
                choice = response["choices"][0]
                reason = choice["finish_reason"]
                if choice["message"].get("refusal"):
                    raise ValueError
                content = choice["message"]["content"]
            if reason != "stop" or not isinstance(content, str) or not content.strip():
                raise ValueError
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise RuntimeError("LLM response missing, refused, truncated or incomplete") from None
        return content
