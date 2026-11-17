"""Structured calls: validated Pydantic output, one repair retry, then a typed failure.

``LLMSession.call`` is the only way the agent talks to a model:

1. look the request up in the cache (cache hits cost nothing);
2. check the budget against a worst-case estimate, then send;
3. charge the real usage;
4. parse the JSON and validate it against the schema plus optional semantic checks;
5. on failure, send exactly one repair request that quotes the errors, and validate again;
6. still invalid: raise :class:`LLMOutputError`. There is no silent default.

Every attempt is recorded as a :class:`CallRecord` for the manifest.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from holdout.core.errors import LLMOutputError
from holdout.llm.budget import Budget
from holdout.llm.cache import ResponseCache, cache_key
from holdout.llm.client import LLMClient, LLMRequest, LLMResponse

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


@dataclass
class LLMSession:
    client: LLMClient
    budget: Budget
    seed: int
    cache: ResponseCache = field(default_factory=ResponseCache)
    records: list[CallRecord] = field(default_factory=list)

    def call(
        self,
        *,
        node: str,
        prompt: RenderedPrompt,
        schema: type[T],
        check: Callable[[T], list[str]] | None = None,
    ) -> T:
        request = LLMRequest(
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.system,
            user=prompt.user,
            response_schema=schema.model_json_schema(),
            seed=self.seed,
        )
        answer, errors, text = self._attempt(node, prompt, request, schema, check, attempt=1)
        if answer is not None:
            return answer
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
        )
        answer, errors, _ = self._attempt(node, prompt, repair, schema, check, attempt=2)
        if answer is not None:
            return answer
        raise LLMOutputError(
            f"{prompt.name}: the model's answer is invalid after one repair: {'; '.join(errors)}"
        )

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
        if cached is not None:
            response: LLMResponse = cached
        else:
            self.budget.check(request, node)
            response = self.client.generate(request)
            inr = self.budget.charge(response.usage, node=node, prompt_name=prompt.name)
        answer, errors = _parse(response.text, schema, check)
        if cached is None and answer is not None:
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
                valid=answer is not None,
                errors=tuple(errors),
            )
        )
        return answer, errors, response.text


def _parse[T: BaseModel](
    text: str, schema: type[T], check: Callable[[T], list[str]] | None
) -> tuple[T | None, list[str]]:
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
    problems = check(answer) if check is not None else []
    return (None, problems) if problems else (answer, [])
