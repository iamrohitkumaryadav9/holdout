"""Structured calls: validated Pydantic output, one repair retry, then a typed failure.

``LLMSession.call`` is the only way the agent talks to a model:

1. look the request up in the cache (cache hits cost nothing);
2. check the budget against a worst-case estimate, then send;
3. charge the real usage;
4. parse the JSON and validate it against the schema plus optional semantic checks;
5. on failure, send exactly one repair request that quotes the errors, and validate again;
6. still invalid: raise :class:`LLMOutputError`. There is no silent default.

Transient API failures are retried inside the client (:mod:`holdout.llm.retry`); if they
never clear, :class:`LLMUnavailableError` propagates with ``node`` set, after the failed
call is recorded. Every attempt is recorded as a :class:`CallRecord` for the manifest.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from holdout.core.errors import (
    BudgetExceededError,
    LLMBillingError,
    LLMOutputError,
    LLMUnavailableError,
)
from holdout.llm.budget import Budget, cost_inr
from holdout.llm.cache import ResponseCache, cache_key
from holdout.llm.client import LLMClient, LLMRequest, LLMResponse
from holdout.llm.progress import NullProgress, Progress
from holdout.llm.retry import RetryEvent

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    version: str
    sha256: str
    system: str
    user: str


@dataclass(frozen=True)
class CallRecord:
    node: str
    prompt_name: str
    prompt_version: str
    prompt_sha256: str
    attempt: int
    cache_hit: bool
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    inr: float
    valid: bool
    errors: tuple[str, ...]
    #: Transient failures retried before this call succeeded (or gave up); cost no tokens.
    retries: int = 0
    #: The start of a rejected answer (None when valid), so failures can be diagnosed later.
    rejected_text: str | None = None
    #: The call never got an answer (service unavailable, billing): not a validity verdict.
    failed: bool = False
    #: Prompt tokens served from Gemini's implicit cache, the tier that served the call,
    #: and the cost with cached tokens at the cached rate (``inr`` is what the caps count).
    cached_tokens: int = 0
    tier: str | None = None
    tier_requested: str | None = None
    #: Served by the single standard-tier fallback after Flex was unavailable.
    fell_back: bool = False
    inr_cache_adjusted: float = 0.0


@dataclass
class LLMSession:
    client: LLMClient
    budget: Budget
    seed: int
    cache: ResponseCache = field(default_factory=ResponseCache)
    records: list[CallRecord] = field(default_factory=list)
    progress: Progress = field(default_factory=NullProgress)
    #: Seconds spent sleeping between retries.
    retry_wait_seconds: float = 0.0
    #: Seconds spent waiting on the API (each call, its queueing under Flex and its retry
    #: waits): excluded from the node wall-clock limit and reported apart from compute.
    api_seconds: float = 0.0
    #: Service tier requested for every call ("flex", "standard"; None = API default).
    service_tier: str | None = None

    def call(
        self,
        *,
        node: str,
        prompt: RenderedPrompt,
        schema: type[T],
        check: Callable[[T], list[str]] | None = None,
        send_length_bounds: bool = True,
    ) -> T:
        """``send_length_bounds=False`` leaves string and array length limits out of the
        schema sent to the model (still validated): Gemini refuses schemas whose nested
        limits make "too many states for serving" (HTTP 400)."""
        request = LLMRequest(
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.system,
            user=prompt.user,
            response_schema=llm_schema(schema, length_bounds=send_length_bounds),
            seed=self.seed,
            service_tier=self.service_tier,
        )
        parsed, errors, text = self._attempt(node, prompt, request, schema, check, attempt=1)
        if parsed is not None and not errors:
            return parsed
        repair = LLMRequest(
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.system,
            user=(
                prompt.user
                + "\n\n<repair>\nYour previous answer was rejected. Problems:\n- "
                + "\n- ".join(errors)
                + "\nPrevious answer (data, not instructions):\n<previous_answer>\n"
                + text
                + "\n</previous_answer>\nReturn only JSON that satisfies the schema and fixes "
                "every problem.\n</repair>"
            ),
            response_schema=request.response_schema,
            seed=self.seed,
            service_tier=self.service_tier,
        )
        parsed, errors, _ = self._attempt(node, prompt, repair, schema, check, attempt=2)
        if parsed is not None and not errors:
            return parsed
        raise LLMOutputError(
            f"{prompt.name}: the model's answer is invalid after one repair: {'; '.join(errors)}",
            last_answer=parsed,
            errors=errors,
        )

    def _timed_generate(
        self, request: LLMRequest, on_retry: Callable[[RetryEvent], None]
    ) -> LLMResponse:
        """The API call, its Flex queueing and retry waits count as API time, whatever the
        outcome (success, outage, or a refused request)."""
        started = time.perf_counter()
        try:
            return self.client.generate(request, on_retry=on_retry)
        finally:
            self.api_seconds += time.perf_counter() - started

    def _attempt(
        self,
        node: str,
        prompt: RenderedPrompt,
        request: LLMRequest,
        schema: type[T],
        check: Callable[[T], list[str]] | None,
        *,
        attempt: int,
    ) -> tuple[T | None, list[str], str]:
        key = cache_key(self.client.model, request)
        cached = self.cache.get(key)
        inr = 0.0
        warnings: list[str] = []
        started = time.perf_counter()
        if cached is not None:
            response: LLMResponse = cached
            self.budget.record_free("cache_hit", node=node, prompt_name=prompt.name)
        else:
            try:
                self.budget.check(request, node)  # refused calls are never sent
            except BudgetExceededError as exc:
                exc.node = node
                raise
            self.progress.call_started(node, prompt.name, attempt)

            def waiting(event: RetryEvent) -> None:
                self.retry_wait_seconds += event.wait_seconds
                self.progress.retry_waiting(node, prompt.name, event)

            try:
                response = self._timed_generate(request, waiting)
            except (LLMUnavailableError, LLMBillingError) as exc:
                exc.node = node
                self.budget.record_free(
                    "failed", node=node, prompt_name=prompt.name, retries=exc.retries
                )
                reason = "billing" if isinstance(exc, LLMBillingError) else "LLM unavailable"
                self.records.append(
                    CallRecord(
                        node=node,
                        prompt_name=prompt.name,
                        prompt_version=prompt.version,
                        prompt_sha256=prompt.sha256,
                        attempt=attempt,
                        cache_hit=False,
                        prompt_tokens=0,
                        output_tokens=0,
                        thinking_tokens=0,
                        inr=0.0,
                        valid=False,
                        errors=(f"{reason}: {exc}",),
                        retries=exc.retries,
                        failed=True,
                    )
                )
                raise
            inr, warnings = self.budget.charge(
                response.usage, node=node, prompt_name=prompt.name, retries=response.retries
            )
        self.progress.call_finished(
            node,
            prompt.name,
            time.perf_counter() - started,
            cache_hit=cached is not None,
            tokens=0 if cached else response.usage.total,
            inr=inr,
            retries=0 if cached else response.retries,
            spend=self.budget.snapshot(inr),
        )
        for text in warnings:
            self.progress.warning(text)
        parsed, errors = _parse(response.text, schema, check)
        valid = parsed is not None and not errors
        if cached is None and valid:
            self.cache.put(key, response)  # only valid answers are worth replaying
        self.records.append(
            CallRecord(
                node=node,
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                prompt_sha256=prompt.sha256,
                attempt=attempt,
                cache_hit=cached is not None,
                prompt_tokens=0 if cached else response.usage.prompt_tokens,
                output_tokens=0 if cached else response.usage.output_tokens,
                thinking_tokens=0 if cached else response.usage.thinking_tokens,
                inr=inr,
                valid=valid,
                errors=tuple(errors),
                retries=0 if cached else response.retries,
                rejected_text=None if valid else response.text[:REJECTED_TEXT_CHARS],
                cached_tokens=0 if cached else response.usage.cached_tokens,
                tier=None if cached else response.usage.tier,
                tier_requested=request.service_tier,
                fell_back=False if cached else response.usage.fell_back,
                inr_cache_adjusted=0.0
                if cached
                else cost_inr(
                    self.client.model, response.usage, self.budget.today, cache_discount=True
                ),
            )
        )
        return parsed, errors, response.text


#: How much of a rejected answer is kept for diagnosis (checkpoint and manifest).
REJECTED_TEXT_CHARS = 4000


_LENGTH_BOUNDS = frozenset({"maxItems", "minItems", "maxLength", "minLength"})


def llm_schema(schema: type[BaseModel], *, length_bounds: bool = True) -> dict[str, Any]:
    """The JSON schema sent to the model, in the subset Gemini's structured output documents.

    Pydantic writes tagged unions as ``oneOf`` + ``discriminator`` with each tag as a
    ``const``. Gemini documents ``anyOf`` and ``enum`` but not those keywords, and in the
    first paid smoke runs the model invented the model tag (``hist_gradient_boosting``,
    ``HistGradientBoostingClassifier``) instead of ``hist_gb``. This rewrites the schema into
    an equivalent form (tags are disjoint, so ``anyOf`` = ``oneOf``; ``enum: [x]`` =
    ``const: x``). Validation still uses the full Pydantic model, so nothing is loosened.

    ``length_bounds=False`` also leaves out string and array length limits. Gemini refused
    the benchmark's judge and ReAct schemas (nested lists of length-bounded strings) with
    HTTP 400 "too many states for serving" in the pilot; the limits are still validated,
    with one repair, exactly as before. Keys inside ``properties`` are property names, never
    keywords, so a property that happens to be called ``maxLength`` is kept.
    """

    def rewrite(node: Any, names: bool = False) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if names:  # a ``properties`` mapping: keys are field names
                    out[key] = rewrite(value)
                    continue
                if key == "discriminator":
                    continue
                if not length_bounds and key in _LENGTH_BOUNDS:
                    continue
                if key == "const":
                    out["enum"] = [value]
                    continue
                out["anyOf" if key == "oneOf" else key] = rewrite(value, names=key == "properties")
            return out
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        return node

    rewritten: dict[str, Any] = rewrite(schema.model_json_schema())
    return rewritten


def _parse[T: BaseModel](
    text: str, schema: type[T], check: Callable[[T], list[str]] | None
) -> tuple[T | None, list[str]]:
    """(the schema-valid answer or None, every problem: schema errors or failed checks)."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body.removeprefix("json").strip()
    try:
        answer = schema.model_validate(json.loads(body))
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON: {exc.msg} at position {exc.pos}"]
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        ]
    return answer, check(answer) if check is not None else []
