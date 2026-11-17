"""On-disk response cache: development re-runs cost nothing.

Key = sha256 of (model, prompt name, prompt version, rendered system and user text,
response schema, generation parameters including the run seed). Two run seeds therefore
never share a cached response. Entries hold the response text and its original usage;
never the API key (it is not part of any request).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from holdout.llm import config
from holdout.llm.client import LLMRequest, LLMResponse, Usage


def cache_key(model: str, request: LLMRequest) -> str:
    payload = {
        "model": model,
        "prompt_name": request.prompt_name,
        "prompt_version": request.prompt_version,
        "system": request.system,
        "user": request.user,
        "schema": request.response_schema,
        "parameters": request.parameters(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def default_cache_dir() -> Path:
    return config.holdout_home() / "cache"


@dataclass
class ResponseCache:
    directory: Path = field(default_factory=default_cache_dir)
    enabled: bool = True

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(text=data["text"], usage=Usage(**data["usage"]))

    def put(self, key: str, response: LLMResponse) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": response.text,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.output_tokens,
                "thinking_tokens": response.usage.thinking_tokens,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
