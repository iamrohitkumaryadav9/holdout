"""A2: lexicon hints, signal-gated severity, and quiet clean twins."""

import pytest

from holdout.audit.checks import a01_single_feature as a1
from holdout.audit.checks import a02_post_outcome as a2
from holdout.audit.hints import ColumnSample, HintRequest, LexiconHintProvider, tokens
from holdout.audit.types import Severity
from tests.audit_harness import trap_context, twin_context
from tests.tiers import SEEDS


def test_tokens_split_every_naming_style() -> None:
    assert tokens("refundIssued") == ["refund", "issued"]
    assert tokens("Final-Settlement amount") == ["final", "settlement", "amount"]


def test_lexicon_hints_names_values_and_target_tokens() -> None:
    request = HintRequest(
        columns=[
            ColumnSample("refund_issued", ["0", "1"]),
            ColumnSample("churn_date", ["2024-01-01"]),
            ColumnSample("ticket_status", ["open", "resolved"]),
            ColumnSample("age", ["31", "45"]),
            ColumnSample("churned", ["0", "1"]),
        ],
        target="churned",
        goal_text="predict churned customers",
    )
    hints = {h.column: h for h in LexiconHintProvider().suggest(request)}
    assert set(hints) == {"refund_issued", "churn_date", "ticket_status"}
    assert "refund" in hints["refund_issued"].terms
    assert "churn" in hints["churn_date"].terms
    assert any("post-outcome statuses" in r for r in hints["ticket_status"].reasons)


def test_shared_target_token_without_timing_word() -> None:
    request = HintRequest(
        columns=[ColumnSample("units_returned_later", []), ColumnSample("units_in_stock", [])],
        target="units_sold",
        goal_text="",
    )
    hints = {h.column: h for h in LexiconHintProvider().suggest(request)}
    assert any("shares ['units']" in r for r in hints["units_in_stock"].reasons)


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_leak_is_a_warning_and_hinted(seed: int) -> None:
    ctx = trap_context("direct_leak", seed)
    outcome, hinted = a2.run(ctx, a1.measure(ctx))
    assert "refund_issued" in hinted
    flag = next(f for f in outcome.flags if f.columns == ["refund_issued"])
    assert flag.severity is Severity.WARNING  # A2 alone is never critical
    assert flag.evidence.kind == "hint"


def test_hint_without_signal_stays_info() -> None:
    ctx = trap_context("direct_leak", 0)
    outcome, _ = a2.run(ctx, None)  # no A1 result: no signal evidence
    assert all(f.severity is Severity.INFO for f in outcome.flags)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["direct_leak", "noisy_proxy", "group_leakage"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    ctx = twin_context(trap, seed)
    outcome, _ = a2.run(ctx, a1.measure(ctx))
    assert [f for f in outcome.flags if f.severity is not Severity.INFO] == []


def test_noisy_proxy_gets_no_name_hint() -> None:
    ctx = trap_context("noisy_proxy", 0)
    _, hinted = a2.run(ctx, None)
    assert "signal_strength" not in hinted
