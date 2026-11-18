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
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from holdout.core.errors import LLMConfigError
from holdout.llm import config
from holdout.llm.retry import RetryEvent, RetryPolicy, call_with_retries


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
    #: Requested service tier ("standard" or "flex"; None = the API default, standard). It
    #: changes price and latency, not the answer, so it is not part of the cache key.
    service_tier: str | None = None

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
    #: Prompt tokens served from a context cache (a subset of ``prompt_tokens``).
    cached_tokens: int = 0
    #: The tier that actually served the call ("standard", "flex", ...; None = not reported).
    tier: str | None = None

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage
    retries: int = 0


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def supports_seed(self) -> bool: ...

    def generate(
        self, request: LLMRequest, on_retry: Callable[[RetryEvent], None] | None = None
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- Gemini


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
        policy: RetryPolicy | None = None,
        rng: random.Random | None = None,
        timeout_seconds: float = config.REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._sleep = sleep
        self._policy = policy if policy is not None else RetryPolicy()
        self._rng = rng if rng is not None else random.Random()
        if sdk_client is not None:
            self._client = sdk_client
        else:
            key = os.environ.get(config.API_KEY_ENV)
            if not key:
                raise LLMConfigError(
                    f"{config.API_KEY_ENV} is not set; set it, or run with --offline"
                )
            from google import genai
            from google.genai import types

            # attempts=1 disables the SDK's own (tenacity) retries, and setting retry options
            # also turns off its implicit async retries: Holdout's policy owns every retry.
            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
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

    def verify_model(self, on_retry: Callable[[RetryEvent], None] | None = None) -> list[str]:
        """List available models (with retries); raise unless the model is among them."""
        listed, _ = call_with_retries(
            lambda: list(self._client.models.list()),
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            on_retry=on_retry,
            what="listing Gemini models",
        )
        names = sorted(str(getattr(m, "name", "")).removeprefix("models/") for m in listed)
        if self._model not in names:
            flash = [n for n in names if "flash" in n]
            raise LLMConfigError(
                f"model {self._model!r} is not available to this key; Flash models seen: {flash}"
            )
        self.model_verified = True
        return names

    def generate(
        self, request: LLMRequest, on_retry: Callable[[RetryEvent], None] | None = None
    ) -> LLMResponse:
        from google.genai import types

        structured = bool(request.response_schema)  # an empty schema asks for plain text
        generation = types.GenerateContentConfig(
            system_instruction=request.system or None,
            response_mime_type="application/json" if structured else None,
            response_json_schema=dict(request.response_schema) if structured else None,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            seed=request.seed,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(request.thinking_level.upper())
            ),
            service_tier=(
                types.ServiceTier(request.service_tier) if request.service_tier else None
            ),
            # The raw body: google-genai 3.4 parses only ``trafficType`` (Vertex AI only)
            # and drops the Gemini API's ``usageMetadata.serviceTier``, the served tier.
            should_return_http_response=True,
        )
        response, retries = call_with_retries(
            lambda: self._client.models.generate_content(
                model=self._model, contents=request.user, config=generation
            ),
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            on_retry=on_retry,
            what=f"Gemini call ({request.prompt_name})",
        )
        text, usage = parse_response(response)
        return LLMResponse(text=text, retries=retries, usage=usage)


def parse_response(response: Any) -> tuple[str, Usage]:
    """Text and usage from a ``generate_content`` response.

    With ``should_return_http_response`` the SDK returns only ``sdk_http_response``: the
    text is the answer's non-thought parts, and usage comes from ``usageMetadata``. The
    served tier is ``usageMetadata.serviceTier`` (Gemini API), else ``trafficType``
    (Vertex AI), else a response header naming the tier; None when none is reported
    (priced as standard). A parsed SDK response (tests, older paths) is read as before.
    """
    raw = getattr(response, "sdk_http_response", None)
    body = getattr(raw, "body", None) if raw is not None else None
    if body is None:
        usage = getattr(response, "usage_metadata", None)
        return str(getattr(response, "text", "") or ""), Usage(
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            tier=served_tier(
                getattr(usage, "service_tier", None) or getattr(usage, "traffic_type", None)
            ),
        )
    data = json.loads(body) if body else {}
    candidates = data.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
    meta = data.get("usageMetadata") or {}
    headers = {str(k).lower(): str(v) for k, v in (getattr(raw, "headers", None) or {}).items()}
    header_tier = next((v for k, v in headers.items() if "tier" in k), None)
    tier = served_tier(meta.get("serviceTier") or meta.get("trafficType") or header_tier)
    return text, Usage(
        prompt_tokens=int(meta.get("promptTokenCount", 0) or 0),
        output_tokens=int(meta.get("candidatesTokenCount", 0) or 0),
        thinking_tokens=int(meta.get("thoughtsTokenCount", 0) or 0),
        cached_tokens=int(meta.get("cachedContentTokenCount", 0) or 0),
        tier=tier,
    )


#: Reported tier names -> the tier that served the call. ``serviceTier`` (Gemini API)
#: and ``trafficType`` (Vertex AI) spell them differently.
_TIERS: dict[str, str] = {
    "ON_DEMAND": "standard",
    "ON_DEMAND_FLEX": "flex",
    "ON_DEMAND_PRIORITY": "priority",
    "STANDARD": "standard",
    "FLEX": "flex",
    "PRIORITY": "priority",
}


def served_tier(reported: Any) -> str | None:
    """The tier from a response's reported tier; None when not reported or unspecified
    (priced as standard, the conservative choice)."""
    if reported is None:
        return None
    name = str(getattr(reported, "value", reported)).upper().removeprefix("SERVICE_TIER_")
    if name in ("", "UNSPECIFIED", "TRAFFIC_TYPE_UNSPECIFIED"):
        return None
    return _TIERS.get(name, name.lower())


# --------------------------------------------------------------------------- fakes

Responder = Callable[[LLMRequest], str | Mapping[str, Any]]
ScriptItem = str | Mapping[str, Any] | Responder | BaseException


@dataclass
class FakeLLM:
    """Scripted responses per prompt name, in order; then an optional default responder.

    A script item may be an exception (e.g. ``TransientLLMError(503)``) to simulate an
    outage. Usage is deterministic (characters / 4), so budget and manifest tests are exact.
    """

    scripts: dict[str, list[ScriptItem]] = field(default_factory=dict)
    default: Mapping[str, Responder] = field(default_factory=dict)
    #: Priced like the real model so budget tests are realistic; the manifest records the
    #: client class separately.
    model_name: str = config.DEFAULT_MODEL
    calls: list[LLMRequest] = field(default_factory=list)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    #: Report implicit-cache hits as the prompt prefix shared with the previous call
    #: (characters / 4), like Gemini's implicit caching; off by default.
    simulate_cache: bool = False

    def __post_init__(self) -> None:
        self._previous_prompt: str | None = None
        self._queues: dict[str, deque[ScriptItem]] = defaultdict(deque)
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

    def generate(
        self, request: LLMRequest, on_retry: Callable[[RetryEvent], None] | None = None
    ) -> LLMResponse:
        """Produce the next scripted item. Exceptions in a script are raised; transient
        ones go through the same retry loop as the real client (without sleeping)."""

        def attempt() -> str:
            self.calls.append(request)
            queue = self._queues[request.prompt_name]
            if queue:
                item = queue.popleft()
            elif request.prompt_name in self.default:
                item = self.default[request.prompt_name]
            else:
                raise AssertionError(f"FakeLLM has no response for {request.prompt_name!r}")
            if isinstance(item, BaseException):
                raise item
            produced = item(request) if callable(item) else item
            return produced if isinstance(produced, str) else json.dumps(produced)

        text, retries = call_with_retries(
            attempt,
            policy=self.retry_policy,
            sleep=lambda seconds: None,
            rng=random.Random(0),
            on_retry=on_retry,
            what=f"fake call ({request.prompt_name})",
        )
        prompt = request.system + request.user
        cached = 0
        if self.simulate_cache and self._previous_prompt is not None:
            shared = 0
            for a, b in zip(self._previous_prompt, prompt, strict=False):
                if a != b:
                    break
                shared += 1
            cached = shared // 4
        self._previous_prompt = prompt
        return LLMResponse(
            text=text,
            retries=retries,
            usage=Usage(
                prompt_tokens=len(prompt) // 4,
                output_tokens=len(text) // 4,
                thinking_tokens=0,
                cached_tokens=cached,
                tier=request.service_tier or "standard",
            ),
        )
