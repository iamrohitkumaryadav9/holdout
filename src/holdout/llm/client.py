"""The LLM client protocol and its implementations.

``LLMClient.generate`` takes a fully rendered :class:`LLMRequest` and returns raw text plus
real token usage. Validation, repair, caching and budget live in
:mod:`holdout.llm.structured`, so every implementation is held to the same rules.

* :class:`GeminiClient` - google-genai SDK, JSON-schema structured output, capped thinking,
  the run seed passed as the generation seed (best effort, per the SDK).
* :class:`FakeLLM` - scripted responses for tests; never touches the network.
"""

import json
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from holdout.core.errors import LLMUnavailableError
from holdout.llm import config


@dataclass(frozen=True)
class LLMRequest:
    """Everything that determines a response (and hence the cache key)."""

    prompt_name: str
    prompt_version: str
    system: str
    user: str
    response_schema: Mapping[str, Any]
    seed: int
    temperature: float = config.TEMPERATURE
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS
    thinking_level: str = config.THINKING_LEVEL

    def parameters(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "thinking_level": self.thinking_level,
        }


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def supports_seed(self) -> bool: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------- Gemini


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class GeminiClient:
    """google-genai client. The API key is read from ``GEMINI_API_KEY`` only.

    ``sdk_client`` lets tests inject a stand-in for ``genai.Client`` (no network).
    """

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        *,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._sleep = sleep
        if sdk_client is not None:
            self._client = sdk_client
        else:
            key = os.environ.get(config.API_KEY_ENV)
            if not key:
                raise LLMUnavailableError(
                    f"{config.API_KEY_ENV} is not set; set it, or run with --offline"
                )
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_SECONDS * 1000),
            )
        self.model_verified = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_seed(self) -> bool:
        # GenerateContentConfig.seed exists in google-genai 3.4; the docs call it best
        # effort (same seed and parameters "make a best effort" to repeat a response).
        return True

    def verify_model(self) -> list[str]:
        """List available models; raise unless the configured model is among them."""
        names = sorted(
            str(getattr(m, "name", "")).removeprefix("models/") for m in self._client.models.list()
        )
        if self._model not in names:
            flash = [n for n in names if "flash" in n]
            raise LLMUnavailableError(
                f"model {self._model!r} is not available to this key; Flash models seen: {flash}"
            )
        self.model_verified = True
        return names

    def generate(self, request: LLMRequest) -> LLMResponse:
        from google.genai import types

        generation = types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json",
            response_json_schema=dict(request.response_schema),
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            seed=request.seed,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(request.thinking_level.upper())
            ),
        )
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=request.user, config=generation
                )
                break
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status in _TRANSIENT_STATUS and attempt < config.MAX_TRANSIENT_RETRIES:
                    self._sleep(2.0**attempt)
                    attempt += 1
                    continue
                raise LLMUnavailableError(
                    f"Gemini call failed ({type(exc).__name__}, status {status})"
                ) from exc
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=str(getattr(response, "text", "") or ""),
            usage=Usage(
                prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            ),
        )


# --------------------------------------------------------------------------- fakes

Responder = Callable[[LLMRequest], str | Mapping[str, Any]]


@dataclass
class FakeLLM:
    """Scripted responses per prompt name, in order; then an optional default responder.

    Usage is deterministic (characters / 4), so budget and manifest tests are exact.
    """

    scripts: dict[str, list[str | Mapping[str, Any] | Responder]] = field(default_factory=dict)
    default: Mapping[str, Responder] = field(default_factory=dict)
    #: Priced like the real model so budget tests are realistic; the manifest records the
    #: client class separately.
    model_name: str = config.DEFAULT_MODEL
    calls: list[LLMRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queues: dict[str, deque[str | Mapping[str, Any] | Responder]] = defaultdict(deque)
        for name, items in self.scripts.items():
            self._queues[name].extend(items)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def supports_seed(self) -> bool:
        return True

    def calls_for(self, prompt_name: str) -> list[LLMRequest]:
        return [c for c in self.calls if c.prompt_name == prompt_name]

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        queue = self._queues[request.prompt_name]
        if queue:
            item = queue.popleft()
        elif request.prompt_name in self.default:
            item = self.default[request.prompt_name]
        else:
            raise AssertionError(f"FakeLLM has no response for {request.prompt_name!r}")
        produced = item(request) if callable(item) else item
        text = produced if isinstance(produced, str) else json.dumps(produced)
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=(len(request.system) + len(request.user)) // 4,
                output_tokens=len(text) // 4,
                thinking_tokens=0,
            ),
        )
