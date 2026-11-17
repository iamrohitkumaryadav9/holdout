"""Live smoke test against the real Gemini API (manual; never run in CI).

Runs two autopilot agent runs - the churn example and a planted direct-leak trap - and
prints, per run: the model, calls per node, prompt/output/thinking tokens, rupees at paid
Flash pricing (even on the free tier), cache hits, wall time, and whether the leak was
caught. Requires GEMINI_API_KEY in the environment.

A timestamped progress line is printed for every graph node, every Gemini call and every
retry wait. If Gemini stays unavailable (429 / 5xx / timeouts after every retry), the run
pauses with a checkpoint, the remaining runs are skipped, and the script still prints what
completed plus the resume command; it exits with 5.

    python -m scripts.smoke_gemini                            # fresh calls (no cache)
    python -m scripts.smoke_gemini --cache                    # allow cached responses
    python -m scripts.smoke_gemini --model gemini-3.6-flash   # another priced model
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

from bench.traps import build_case
from holdout.agent.driver import PAUSED_LLM, AgentOutcome, run_agent
from holdout.agent.graph import AgentOptions
from holdout.core.errors import LLMConfigError, LLMRequestError, LLMUnavailableError
from holdout.llm import config
from holdout.llm.client import GeminiClient
from holdout.llm.progress import LineProgress

EXIT_LLM_UNAVAILABLE = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", action="store_true", help="allow cached LLM responses")
    parser.add_argument("--out", type=Path, default=Path("smoke_runs"))
    parser.add_argument(
        "--model",
        default=None,
        help=f"model for this smoke run only (default {config.DEFAULT_MODEL}); "
        "must be priced in holdout.llm.config",
    )
    args = parser.parse_args()
    if not os.environ.get(config.API_KEY_ENV):
        print(f"{config.API_KEY_ENV} is not set.", file=sys.stderr)
        return 2
    model = args.model or config.DEFAULT_MODEL
    progress = LineProgress(lambda line: print(line, flush=True))

    try:
        config.require_priced(model, date.today())
        client = GeminiClient(model=model)
        progress.node_started("verify_model")
        started = time.perf_counter()
        available = client.verify_model(
            on_retry=lambda event: progress.retry_waiting("verify_model", "list models", event)
        )
        progress.node_finished("verify_model", time.perf_counter() - started)
    except LLMUnavailableError as exc:
        print(f"\nGemini is unavailable: {exc}. No run was started; try again later.")
        return EXIT_LLM_UNAVAILABLE
    except (LLMConfigError, LLMRequestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    flash = [name for name in available if "flash" in name]
    override = " (--model)" if args.model else ""
    print(f"model {model}{override}: available (Flash models visible: {', '.join(flash)})")
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
    finished: list[tuple[str, AgentOutcome, float, str | None]] = []
    paused: AgentOutcome | None = None
    for label, csv, target, goal, leaked in runs:
        print(f"\n-- {label}", flush=True)
        started = time.perf_counter()
        outcome = run_agent(
            AgentOptions(
                csv=str(csv),
                target=target,
                out_dir=str(args.out / "runs"),
                goal=goal,
                mode="autopilot",
                use_cache=args.cache,
                model=args.model,
            ),
            progress=progress,
        )
        finished.append((label, outcome, time.perf_counter() - started, leaked))
        if outcome.status == PAUSED_LLM:
            paused = outcome
            break

    failures = 0
    for label, outcome, seconds, leaked in finished:
        if outcome.status == PAUSED_LLM:
            _print_paused(label, outcome, seconds)
        else:
            failures += _print(label, outcome, seconds, leaked)
    if paused is not None:
        skipped = [label for label, *_ in runs[len(finished) :]]
        if skipped:
            print(f"\n   skipped (Gemini unavailable): {', '.join(skipped)}")
        return EXIT_LLM_UNAVAILABLE
    return 1 if failures else 0


def _print_paused(label: str, outcome: AgentOutcome, seconds: float) -> None:
    record = outcome.paused
    assert record is not None
    print(f"\n== {label}: {outcome.status} in {seconds:.1f} s ({outcome.run_dir})")
    print(f"   model {record.model}; completed nodes: {', '.join(record.completed_nodes) or '-'}")
    for node, usage in record.usage_by_node.items():
        print(
            f"   {node:16} calls {usage.calls:2}  cached {usage.cache_hits:2}  prompt "
            f"{usage.prompt_tokens:6}  output {usage.output_tokens:5}  thinking "
            f"{usage.thinking_tokens:5}  Rs {usage.inr:.3f}  retries {usage.retries}"
        )
    print(f"   total Rs {record.total_inr:.3f}; retries {record.retries}")
    print(f"   {record.message}")


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
