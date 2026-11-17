"""Progress reporting: timestamped lines for nodes, LLM calls and retry waits.

``holdout agent --verbose`` and the smoke script print them; everything else uses
:class:`NullProgress`. Lines never contain prompts, data or the API key.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from holdout.llm.retry import RetryEvent


class Progress(Protocol):
    def node_started(self, node: str) -> None: ...

    def node_finished(self, node: str, seconds: float) -> None: ...

    def call_started(self, node: str, purpose: str, attempt: int) -> None: ...

    def call_finished(
        self,
        node: str,
        purpose: str,
        seconds: float,
        *,
        cache_hit: bool,
        tokens: int,
        inr: float,
        retries: int,
    ) -> None: ...

    def retry_waiting(self, node: str, purpose: str, event: RetryEvent) -> None: ...


class NullProgress:
    def node_started(self, node: str) -> None:
        del node

    def node_finished(self, node: str, seconds: float) -> None:
        del node, seconds

    def call_started(self, node: str, purpose: str, attempt: int) -> None:
        del node, purpose, attempt

    def call_finished(
        self,
        node: str,
        purpose: str,
        seconds: float,
        *,
        cache_hit: bool,
        tokens: int,
        inr: float,
        retries: int,
    ) -> None:
        del node, purpose, seconds, cache_hit, tokens, inr, retries

    def retry_waiting(self, node: str, purpose: str, event: RetryEvent) -> None:
        del node, purpose, event


class LineProgress:
    """Writes one timestamped line per event through ``write`` (e.g. ``print``)."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write

    def _line(self, text: str) -> None:
        self._write(f"[{datetime.now():%H:%M:%S}] {text}")

    def node_started(self, node: str) -> None:
        self._line(f"node {node}: start")

    def node_finished(self, node: str, seconds: float) -> None:
        self._line(f"node {node}: done in {seconds:.1f} s")

    def call_started(self, node: str, purpose: str, attempt: int) -> None:
        suffix = " (repair)" if attempt > 1 else ""
        self._line(f"  gemini {purpose}{suffix} [{node}]: calling")

    def call_finished(
        self,
        node: str,
        purpose: str,
        seconds: float,
        *,
        cache_hit: bool,
        tokens: int,
        inr: float,
        retries: int,
    ) -> None:
        if cache_hit:
            self._line(f"  gemini {purpose} [{node}]: cached response, {seconds:.1f} s, Rs 0.00")
            return
        retried = f", {retries} retr{'y' if retries == 1 else 'ies'}" if retries else ""
        self._line(
            f"  gemini {purpose} [{node}]: done in {seconds:.1f} s, {tokens} tokens, "
            f"Rs {inr:.3f}{retried}"
        )

    def retry_waiting(self, node: str, purpose: str, event: RetryEvent) -> None:
        what = "timeout" if event.kind == "timeout" else f"HTTP {event.status}"
        hint = " (server Retry-After)" if event.retry_after is not None else ""
        self._line(
            f"  gemini {purpose} [{node}]: {what} on attempt {event.attempt}; "
            f"retrying in {event.wait_seconds:.1f} s{hint}"
        )
