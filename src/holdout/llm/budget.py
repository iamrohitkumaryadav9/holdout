"""Spend accounting in rupees, from real token counts, with run and project caps.

Each call is checked *before* it is sent, against a worst-case estimate (the prompt's
characters / 4 plus the full ``max_output_tokens``, priced as output), so a single call can
never push spend past a cap. After the call it is charged with the real counts from the
response's usage metadata. The project total lives in an append-only ledger outside the
repository (``~/.holdout/spend_ledger.jsonl``) so it persists across runs and processes.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from holdout.core.errors import BudgetExceededError
from holdout.llm import config
from holdout.llm.client import LLMRequest, Usage


def cost_inr(model: str, usage: Usage, on: date) -> float:
    """Rupees for a call; thinking tokens are billed at the output rate."""
    price = config.price_for(model, on)
    usd = (
        usage.prompt_tokens * price.input_usd_per_million
        + (usage.output_tokens + usage.thinking_tokens) * price.output_usd_per_million
    ) / 1_000_000
    return usd * config.USD_TO_INR


def estimate_inr(model: str, request: LLMRequest, on: date) -> float:
    """Worst-case cost before sending: every allowed output token is used."""
    worst = Usage(
        prompt_tokens=(len(request.system) + len(request.user)) // 4 + 1,
        output_tokens=request.max_output_tokens,
        thinking_tokens=0,
    )
    return cost_inr(model, worst, on)


def default_ledger_path() -> Path:
    return config.holdout_home() / "spend_ledger.jsonl"


@dataclass
class SpendLedger:
    """Append-only JSON-lines ledger of every paid call, across runs and processes."""

    path: Path = field(default_factory=default_ledger_path)

    def total_inr(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line)["inr"])
        return total

    def append(self, entry: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


@dataclass
class Budget:
    run_id: str
    model: str
    # Read at construction (not import) so configuration changes apply to new budgets.
    run_cap_inr: float = field(default_factory=lambda: config.RUN_CAP_INR)
    project_cap_inr: float = field(default_factory=lambda: config.PROJECT_CAP_INR)
    ledger: SpendLedger = field(default_factory=SpendLedger)
    spent_run_inr: float = 0.0
    today: date = field(default_factory=lambda: datetime.now(UTC).date())

    def check(self, request: LLMRequest, node: str) -> float:
        """Raise before a call that could exceed a cap; return the worst-case estimate."""
        estimate = estimate_inr(self.model, request, self.today)
        if self.spent_run_inr + estimate > self.run_cap_inr:
            raise BudgetExceededError(
                f"run cap: Rs {self.spent_run_inr:.2f} spent, the next call ({node}) could "
                f"cost up to Rs {estimate:.2f}, cap Rs {self.run_cap_inr:.2f}"
            )
        project = self.ledger.total_inr()
        if project + estimate > self.project_cap_inr:
            raise BudgetExceededError(
                f"project cap: Rs {project:.2f} spent across runs, the next call ({node}) "
                f"could cost up to Rs {estimate:.2f}, cap Rs {self.project_cap_inr:.2f}"
            )
        return estimate

    def charge(self, usage: Usage, *, node: str, prompt_name: str) -> float:
        """Record the real cost of a call (in the run and the project ledger)."""
        inr = cost_inr(self.model, usage, self.today)
        self.spent_run_inr += inr
        self.ledger.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self.run_id,
                "node": node,
                "prompt": prompt_name,
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                "thinking_tokens": usage.thinking_tokens,
                "inr": inr,
            }
        )
        return inr
