"""ReAct baseline: Gemini with a Python sandbox, asked to audit its own methodology.

A charitable baseline: same model, budget cap, goal and development rows as Holdout, the
same issue categories and severity scale, a fuller sandbox (scikit-learn, LightGBM) and 60 s
for model-fitting steps. Up to 15 steps; each step is one structured LLM call through
Holdout's ``LLMSession`` (same caps, retries, ledger). The full history is sent every step
(never summarised); each tool observation is truncated to 2,000 characters, head and tail,
and observation sizes are recorded so the cost of the history is visible.

The prompt (``prompts/react.md``) is published in full in docs/benchmark.md.
"""

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from bench.systems.base import (
    FinalPipeline,
    NodeQuality,
    ReportedIssue,
    SystemContext,
    SystemName,
    SystemOutput,
)
from bench.systems.sandbox import (
    MODEL_FILE,
    prepare_workdir,
    run_step,
    truncate_observation,
)
from bench.systems.unpickle import UnsafePickleError, restricted_loads
from holdout.agent.prompts import PromptTemplate
from holdout.audit.checks.a11_causal import check_causal_claims
from holdout.audit.types import Category, Severity
from holdout.core.errors import BudgetExceededError, LLMOutputError, RunPausingError
from holdout.llm.budget import Budget
from holdout.llm.cache import ResponseCache
from holdout.llm.client import LLMClient
from holdout.llm.structured import LLMSession, RenderedPrompt

MAX_STEPS: Final = 15
OBSERVATION_CHARS: Final = 2000
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "react.md"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReactIssue(_Strict):
    category: Category
    severity: Severity
    columns: list[str] = Field(default_factory=list, max_length=20)
    explanation: str = Field(min_length=3, max_length=500)


class ReactFinal(_Strict):
    issues: list[ReactIssue] = Field(max_length=20)
    dropped_columns: list[str] = Field(max_length=200)
    split: str = Field(min_length=3, max_length=100)
    deduplicated: bool
    validation_metric: str = Field(min_length=2, max_length=40)
    validation_score: float
    report: str = Field(min_length=20, max_length=3000)


class ReactStep(_Strict):
    thought: str = Field(min_length=1, max_length=2000)
    action: Literal["python", "final"]
    code: str | None = Field(default=None, max_length=8000)
    final: ReactFinal | None = None


def load_template(max_steps: int = MAX_STEPS) -> PromptTemplate:
    raw = PROMPT_PATH.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    header = re.match(r"\A---\n(?P<h>.*?)\n---\n", text, re.DOTALL)
    sections = dict(re.findall(r"<(system|user)>\n(.*?)\n</\1>", text, re.DOTALL))
    if header is None or set(sections) != {"system", "user"}:
        raise ValueError("react prompt: missing header or sections")
    fields = {
        k.strip(): v.strip()
        for k, v in (line.split(":", 1) for line in header.group("h").splitlines() if ":" in line)
    }
    return PromptTemplate(
        name=fields["name"],
        version=fields["version"],
        purpose=fields["purpose"],
        sha256=hashlib.sha256(raw).hexdigest(),
        system=sections["system"].replace("{{max_steps}}", str(max_steps)),
        user=sections["user"],
    )


def _columns_summary(ctx: SystemContext) -> dict[str, Any]:
    return {
        "rows": len(ctx.dev),
        "columns": {str(c): str(t) for c, t in ctx.dev.dtypes.items()},
    }


def _step_check(step: ReactStep) -> list[str]:
    if step.action == "python" and not (step.code and step.code.strip()):
        return ["a python step needs code"]
    if step.action == "final" and step.final is None:
        return ["a final step needs the final answer"]
    return []


class ReactSystem:
    name: SystemName = "react"
    uses_llm = True

    def __init__(
        self,
        client_factory: Any,
        max_steps: int = MAX_STEPS,
        observation_chars: int = OBSERVATION_CHARS,
    ) -> None:
        self.client_factory = client_factory
        self.max_steps = max_steps
        self.observation_chars = observation_chars

    def run(self, ctx: SystemContext) -> SystemOutput:
        client: LLMClient = self.client_factory()
        session = LLMSession(
            client=client,
            budget=Budget(run_id=ctx.unit_id, model=client.model, run_cap_inr=ctx.run_cap_inr),
            seed=ctx.seed,
            cache=ResponseCache(enabled=False),
            progress=ctx.progress,
        )
        workdir = ctx.workdir / "sandbox"
        prepare_workdir(workdir, ctx.dev)
        template = load_template(self.max_steps)
        history: list[dict[str, Any]] = []
        raw_chars = shown_chars = 0
        sent_chars = raw_sent_chars = 0  # observations re-sent with the history, all steps
        raw_lengths: list[int] = []
        final: ReactFinal | None = None
        stop: str | None = None
        started = time.perf_counter()
        for step_no in range(1, self.max_steps + 1):
            sent_chars += sum(len(h["observation"]) for h in history)
            raw_sent_chars += sum(raw_lengths)
            rendered = template.render(
                goal=ctx.goal,
                target={"target": ctx.target, "task": ctx.task.value},
                columns=_columns_summary(ctx),
                history=history or "none yet",
                step=str(step_no),
                max_steps=str(self.max_steps),
            )
            prompt = RenderedPrompt(
                rendered.name, rendered.version, rendered.sha256, rendered.system, rendered.user
            )
            try:
                step = session.call(
                    node="react", prompt=prompt, schema=ReactStep, check=_step_check
                )
            except BudgetExceededError as exc:
                if exc.scope == "project":
                    raise  # the benchmark pauses; nothing about this system's quality
                stop = f"the per-run budget cap stopped the run at step {step_no}"
                break
            except RunPausingError:
                raise  # unavailable / billing: the runner pauses and reruns the unit later
            except LLMOutputError as exc:
                stop = f"invalid step {step_no} after one repair: {exc}"
                break
            if step.action == "final":
                final = step.final
                break
            assert step.code is not None
            result = run_step(step.code, workdir)
            text = (
                result.output
                if result.status == "ok"
                else (f"[{result.status}] {result.detail or ''}\n{result.output}")
            )
            shown, _ = truncate_observation(text, self.observation_chars)
            raw_chars += len(text)
            shown_chars += len(shown)
            raw_lengths.append(len(text))
            history.append(
                {
                    "step": step_no,
                    "thought": step.thought,
                    "code": step.code,
                    "status": result.status,
                    "observation": shown,
                }
            )
        else:
            stop = f"no final answer within {self.max_steps} steps"
        output = self._output(
            ctx, session, final, stop, workdir, raw_chars, shown_chars, history, started
        )
        output.extra["observation_tokens_sent_estimate"] = sent_chars // 4
        # What the history would have cost without truncation (same steps, same code).
        output.extra["observation_tokens_sent_untruncated_estimate"] = raw_sent_chars // 4
        return output

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)  # a ReAct transcript has no checkpoint: rerun the unit

    def _output(
        self,
        ctx: SystemContext,
        session: LLMSession,
        final: ReactFinal | None,
        stop: str | None,
        workdir: Path,
        raw_chars: int,
        shown_chars: int,
        history: list[dict[str, Any]],
        started: float,
    ) -> SystemOutput:
        records = session.records
        firsts = [r for r in records if r.attempt == 1 and not r.failed]
        model_path = workdir / MODEL_FILE
        estimator, model_error = None, None
        if model_path.exists():
            try:
                estimator = restricted_loads(model_path.read_bytes())
            except UnsafePickleError as exc:
                model_error = str(exc)
        else:
            model_error = "no model was submitted"
        report = final.report if final is not None else None
        issues = (
            [
                ReportedIssue(
                    category=i.category,
                    severity=i.severity,
                    columns=list(i.columns),
                    source="react",
                    message=i.explanation,
                )
                for i in final.issues
            ]
            if final is not None
            else []
        )
        status = "completed" if final is not None else "stopped"
        return SystemOutput(
            status=status,  # type: ignore[arg-type]
            issues=issues,
            pipeline=(
                FinalPipeline(
                    dropped_columns=list(final.dropped_columns),
                    metric=final.validation_metric,
                    split=final.split,
                    deduplicated=final.deduplicated,
                    model="react",
                )
                if final is not None
                else None
            ),
            estimator=estimator,
            validation_metric=final.validation_metric if final else None,
            validation_score=final.validation_score if final else None,
            report_text=report,
            report_a11_flags=len(check_causal_claims(report)) if report else 0,
            cost_inr=sum(r.inr for r in records),
            calls=len(records),
            retries=sum(r.retries for r in records),
            first_attempt={
                "react": NodeQuality(first_attempts=len(firsts), valid=sum(r.valid for r in firsts))
            },
            repairs=sum(r.attempt == 2 for r in records),
            error=stop or model_error,
            extra={
                "steps": len(history) + (1 if final is not None else 0),
                "observation_chars_raw": raw_chars,
                "observation_chars_shown": shown_chars,
                "observation_tokens_shown_estimate": shown_chars // 4,
                "observation_tokens_raw_estimate": raw_chars // 4,
                "prompt_tokens": sum(r.prompt_tokens for r in records),
                "output_tokens": sum(r.output_tokens for r in records),
                "thinking_tokens": sum(r.thinking_tokens for r in records),
                "model_error": model_error,
                "seconds_in_loop": round(time.perf_counter() - started, 1),
                "step_statuses": [h["status"] for h in history],
            },
        )
