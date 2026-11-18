"""Build the benchmark systems from the pinned configuration."""

from collections.abc import Sequence

from bench.config import BenchConfig
from bench.systems.base import System
from bench.systems.flaml_system import FlamlSystem
from bench.systems.holdout_system import HoldoutSystem
from bench.systems.llm_judge import install
from bench.systems.react import ReactSystem
from holdout.agent.driver import ClientFactory
from holdout.llm.client import GeminiClient, LLMClient


def gemini_factory(model: str) -> ClientFactory:
    def make() -> LLMClient:
        client = GeminiClient(model=model)
        client.verify_model()
        return client

    return make


def build_systems(
    names: Sequence[str],
    config: BenchConfig,
    *,
    client_factory: ClientFactory | None = None,
) -> dict[str, System]:
    """The named systems. ``client_factory`` replaces Gemini (tests use FakeLLM)."""
    factory = client_factory
    systems: dict[str, System] = {}
    for name in names:
        if name == "holdout":
            systems[name] = HoldoutSystem("holdout", client_factory=factory)
        elif name == "audits_only":
            systems[name] = HoldoutSystem("audits_only")
        elif name == "llm_judge":
            systems[name] = HoldoutSystem("llm_judge", client_factory=factory, customize=install)
        elif name == "react":
            systems[name] = ReactSystem(
                factory or gemini_factory(config.model),
                max_steps=config.react_max_steps,
                observation_chars=config.react_observation_chars,
            )
        elif name == "flaml":
            systems[name] = FlamlSystem(time_budget=config.flaml_time_budget)
        else:
            raise ValueError(f"unknown system {name!r}")
    return systems
