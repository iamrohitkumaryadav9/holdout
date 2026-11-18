"""The benchmark's pinned configuration (``bench/config.toml``)."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from holdout.llm import config as llm_config

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
ALL_SYSTEMS = ("holdout", "audits_only", "llm_judge", "react", "flaml")


@dataclass(frozen=True)
class BenchConfig:
    model: str
    service_tier: str
    run_cap_inr: float
    systems: tuple[str, ...]
    flaml_time_budget: float
    react_max_steps: int
    react_observation_chars: int
    cost_margin: float
    pilot_results: Path
    sha256: str


def load_bench_config(path: Path = CONFIG_PATH) -> BenchConfig:
    """Parse and validate the configuration.

    Raises:
        ValueError: on an unknown system, a model without a price, a per-run cap that is
            not Holdout's (every system must have the same cap), or a negative margin.
    """
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    cfg = BenchConfig(
        model=str(raw["model"]),
        service_tier=str(raw["service_tier"]),
        run_cap_inr=float(raw["run_cap_inr"]),
        systems=tuple(raw["systems"]),
        flaml_time_budget=float(raw["flaml_time_budget"]),
        react_max_steps=int(raw["react_max_steps"]),
        react_observation_chars=int(raw["react_observation_chars"]),
        cost_margin=float(raw["cost_margin"]),
        pilot_results=Path(raw["pilot_results"]),
        sha256=hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest(),
    )
    unknown = sorted(set(cfg.systems) - set(ALL_SYSTEMS))
    if unknown:
        raise ValueError(f"unknown systems in {path.name}: {unknown}")
    if cfg.service_tier not in llm_config.SERVICE_TIERS:
        raise ValueError(f"unknown service tier {cfg.service_tier!r}")
    table = llm_config.FLEX_PRICE_TABLE if cfg.service_tier == "flex" else llm_config.PRICE_TABLE
    if cfg.model not in table:
        raise ValueError(f"{cfg.model!r} has no {cfg.service_tier} price in holdout.llm.config")
    if cfg.run_cap_inr != llm_config.RUN_CAP_INR:
        raise ValueError("run_cap_inr must equal Holdout's RUN_CAP_INR (one cap for all)")
    if cfg.cost_margin < 0:
        raise ValueError("cost_margin must not be negative")
    return cfg
