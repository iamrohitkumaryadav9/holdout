"""Fixed benchmark configurations for tests of pricing logic, so the tests do not depend on
whichever tier or margin the live bench/config.toml pins after an amendment."""

from dataclasses import replace

from bench.config import BenchConfig, load_bench_config


def config_with(
    tier: str = "flex", tiers: dict[str, str] | None = None, margin: float = 0.20
) -> BenchConfig:
    """The live config with the tier, per-system overrides and margin replaced."""
    return replace(load_bench_config(), service_tier=tier, service_tiers=dict(tiers or {}),
                   cost_margin=margin)  # fmt: skip
