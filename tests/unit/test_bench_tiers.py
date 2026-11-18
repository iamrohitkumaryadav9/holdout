"""Amendment 2: the Gemini tier per system (holdout and llm_judge on Standard, react on
Flex), resolved per unit by the runner and priced per system by the pre-flight."""

from datetime import date
from pathlib import Path

import pytest

from bench.budget import Measured, estimate
from bench.config import CONFIG_PATH, load_bench_config
from bench.run import tier_of

DAY = date(2026, 11, 20)
PILOT = {s: Measured(2.0, 3, DAY, "standard") for s in ("holdout", "llm_judge", "react")}


def test_the_config_pins_standard_for_holdout_and_the_judge_and_flex_for_react() -> None:
    cfg = load_bench_config()
    assert cfg.tier_for("holdout") == "standard"
    assert cfg.tier_for("llm_judge") == "standard"
    assert cfg.tier_for("react") == "flex"
    assert cfg.model == "gemini-3.8-flash"  # one model for every tier


def test_tiers_resolve_per_system() -> None:
    tiers = {"holdout": "standard", "react": "flex"}
    assert tier_of(tiers, "holdout") == "standard"
    assert tier_of(tiers, "react") == "flex"
    assert tier_of(tiers, "flaml") is None
    assert tier_of("flex", "holdout") == "flex"
    assert tier_of(None, "react") is None


def test_unknown_systems_and_tiers_in_overrides_are_refused(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for bad, message in (('{ holdout = "standard", nobody = "flex" }', "unknown systems"),
                         ('{ holdout = "priority" }', "unknown service tier")):  # fmt: skip
        path = tmp_path / "config.toml"
        path.write_text(text.replace('{ holdout = "standard", llm_judge = "standard" }', bad),
                        encoding="utf-8")  # fmt: skip
        with pytest.raises(ValueError, match=message):
            load_bench_config(path)


def test_the_preflight_prices_each_system_at_its_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    cfg = load_bench_config()
    report = estimate({"holdout": 10, "llm_judge": 10, "react": 10}, cfg, purpose="benchmark",
                      measured=PILOT, today=DAY, fallback_share=0.5)  # fmt: skip
    by = {e.system: e.total_inr for e in report.estimates}
    standard = 10 * 2.0 * 1.2
    assert by["holdout"] == pytest.approx(standard)  # Standard: no fallback discount
    assert by["llm_judge"] == pytest.approx(standard)
    assert by["react"] == pytest.approx(standard * (0.5 * 0.5 + 0.5 * 1.0))  # Flex + fallback
    assert "priced at standard" in report.render()
    assert "priced at flex" in report.render()
