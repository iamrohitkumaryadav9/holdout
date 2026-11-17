"""A8: messy numbers, category variants and sentinels are warnings with exact actions."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from holdout.audit.checks import a08_formats as a8
from holdout.audit.types import Severity
from tests.audit_harness import SEEDS, trap_context, twin_context


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_format_issues_are_warnings_with_exact_actions(seed: int) -> None:
    flags = {
        (f.columns[0], f.evidence.issue): f
        for f in a8.run(trap_context("messy_formats", seed)).flags
    }  # type: ignore[union-attr]
    assert set(flags) == {
        ("income", "numeric_strings"),
        ("segment", "category_variants"),
        ("age", "sentinel"),
    }
    assert all(f.severity is Severity.WARNING for f in flags.values())
    parse = flags[("income", "numeric_strings")].fix.action  # type: ignore[union-attr]
    assert parse.kind == "parse_numeric"
    assert {",", "rs"} <= set(parse.strip_tokens)  # type: ignore[union-attr]
    mapping = flags[("segment", "category_variants")].fix.action.mapping  # type: ignore[union-attr]
    assert mapping["RETAIL"] == "retail"
    assert mapping["  Retail "] == "retail"
    sentinel = flags[("age", "sentinel")].fix.action  # type: ignore[union-attr]
    assert sentinel.value == -999.0  # type: ignore[union-attr]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["messy_formats", "direct_leak", "time_order", "group_leakage"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    assert a8.run(twin_context(trap, seed)).flags == []


def _with_column(values: list[object]) -> list[str]:
    ctx = twin_context("messy_formats", 0)
    n = len(ctx.dev)
    column = pd.Series((values * (n // len(values) + 1))[:n], index=ctx.dev.index)
    dev = ctx.dev.assign(probe=column)
    return [f.evidence.issue for f in a8.run(dataclasses.replace(ctx, dev=dev)).flags]  # type: ignore[union-attr]


def test_percent_and_unit_strings_parse() -> None:
    assert _with_column(["12%", "7.5%", "40%"]) == ["numeric_strings"]
    assert _with_column(["12 kg", "7 kg", "1,200 kg"]) == ["numeric_strings"]


def test_plain_categories_and_codes_are_not_numbers() -> None:
    assert _with_column(["red", "green", "blue"]) == []
    assert _with_column(["A1", "B2", "C3"]) == []


def test_minus_one_inside_the_range_is_not_a_sentinel() -> None:
    rng = np.random.default_rng(0)
    assert _with_column(list(rng.integers(-3, 4, 50))) == []
