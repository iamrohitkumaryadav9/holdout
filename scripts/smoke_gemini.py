"""Live smoke test against the real Gemini API (manual; never run in CI).

Runs two autopilot agent runs - the churn example and a planted direct-leak trap - and
prints, per run: the model, calls per node, prompt/output/thinking tokens, rupees at paid
Flash pricing (even on the free tier), cache hits, wall time, and whether the leak was
caught. Requires GEMINI_API_KEY in the environment.

    python -m scripts.smoke_gemini              # fresh calls (no response cache)
    python -m scripts.smoke_gemini --cache      # allow cached responses
"""

import argparse
import os
import sys
import time
from pathlib import Path

from bench.traps import build_case
from holdout.agent.driver import AgentOutcome, run_agent
from holdout.agent.graph import AgentOptions
from holdout.llm import config
from holdout.llm.client import GeminiClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", action="store_true", help="allow cached LLM responses")
    parser.add_argument("--out", type=Path, default=Path("smoke_runs"))
    args = parser.parse_args()
    if not os.environ.get(config.API_KEY_ENV):
        print(f"{config.API_KEY_ENV} is not set.", file=sys.stderr)
        return 2

    client = GeminiClient()
    available = client.verify_model()
    flash = [name for name in available if "flash" in name]
    print(f"model {config.DEFAULT_MODEL}: available (Flash models visible: {', '.join(flash)})")
    print(
        f"thinking_level={config.THINKING_LEVEL}, max_output_tokens={config.MAX_OUTPUT_TOKENS}, "
        f"prices as of {config.PRICES_AS_OF}, {config.USD_TO_INR} INR/USD "
        f"(as of {config.EXCHANGE_RATE_AS_OF}); {config.PRICE_NOTE}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    leak_csv = args.out / "direct_leak.csv"
    leak = build_case("direct_leak", 0)
    leak.df.to_csv(leak_csv, index=False)
    runs = [
        ("churn example", Path("examples/churn_classification.csv"), "churned", None, None),
        ("direct_leak trap", leak_csv, leak.target, leak.goal_text, "refund_issued"),
    ]
    failures = 0
    for label, csv, target, goal, leaked in runs:
        started = time.perf_counter()
        outcome = run_agent(
            AgentOptions(
                csv=str(csv),
                target=target,
                out_dir=str(args.out / "runs"),
                goal=goal,
                mode="autopilot",
                use_cache=args.cache,
            )
        )
        failures += _print(label, outcome, time.perf_counter() - started, leaked)
    return 1 if failures else 0


def _print(label: str, outcome: AgentOutcome, seconds: float, leaked: str | None) -> int:
    manifest = outcome.manifest
    print(f"\n== {label}: {outcome.status} in {seconds:.1f} s ({outcome.run_dir})")
    if manifest is None or manifest.agent is None:
        print(f"   stopped: {outcome.stop_reason}")
        return 1
    agent = manifest.agent
    print(f"   model {agent.model} via {agent.llm_client}; verified={agent.model_verified}")
    for node, usage in agent.usage_by_node.items():
        print(
            f"   {node:16} calls {usage.calls:2}  cached {usage.cache_hits:2}  prompt "
            f"{usage.prompt_tokens:6}  output {usage.output_tokens:5}  thinking "
            f"{usage.thinking_tokens:5}  Rs {usage.inr:.3f}"
        )
    print(
        f"   total Rs {agent.total_inr:.3f}; cache hits {agent.cache_hits}; "
        f"substitutions {len(agent.substitutions)}"
    )
    holdout = (
        manifest.holdout if isinstance(manifest.holdout, str) else f"{manifest.holdout.score:.4f}"
    )
    print(f"   validation {manifest.validation_score:.4f}; holdout {holdout}")
    print(f"   stop: {agent.stop_reason}")
    if leaked is None:
        return 0
    first_round = agent.rounds[0].flags
    caught = any(f.startswith("A1 critical") and leaked in f for f in first_round)
    removed = leaked in manifest.plans[-1].plan.dropped
    print(f"   leak caught: {'yes' if caught else 'NO'} (A1 critical on {leaked!r})")
    print(f"   removed from the final plan: {'yes' if removed else 'NO'}")
    return 0 if caught and removed else 1


if __name__ == "__main__":
    raise SystemExit(main())
