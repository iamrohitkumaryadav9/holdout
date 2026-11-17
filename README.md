# Holdout

Holdout is a self-auditing data-science agent for tabular CSVs. LLM data-science agents
have been shown to leak validation data, and to leak more as they get more capable
([Silent Failures in Agentic Data Science](docs/design.md#problem)). Holdout's position is
that correctness should be enforced by code, not by a prompt. The holdout set is split off,
written to disk and hashed before anything computes a feature-target statistic. Proposals
(from Gemini in agent runs, or an offline default) are typed `ModelPlan`s with bounded choices;
deterministic code validates them and compiles them into leakage-safe scikit-learn
pipelines. After cross-validation, twelve audit checks inspect the pipeline on the
development rows for leakage, contaminated splits and misleading metrics. A critical
finding blocks the single, logged holdout evaluation. Every run exports a standalone script
that reproduces its result without Holdout installed.

**Status: Week 3 of 6.** The deterministic core, the audit layer, and the agent: a Gemini-backed
LangGraph agent that proposes plans, explains audit flags, applies their fixes and writes a
report whose every number comes from run state, with checkpointed resume, an autopilot mode,
spend caps and an optional EDA sandbox. No UI yet; benchmark results come in Week 4.

## Quickstart (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Or with uv: `uv venv --python 3.12`, activate it as above, then `uv pip install -e ".[dev]"`.

```powershell
# Two small synthetic datasets: churn (classification) and store sales (time-ordered)
holdout make-example --out examples

# Classification: stratified split, dummy baseline, default plan, audit, one holdout evaluation
holdout run --csv examples\churn_classification.csv --target churned --unseal

# Time-ordered data declared as tabular: the time-order warning before sealing, then the
# audit (A6) finds the time leak and marks this run's holdout invalid
holdout run --csv examples\store_sales_timeseries.csv --target units_sold

# The same data with --data-type timeseries: time-ordered tabular data
# (time-based split; forecasting is not supported yet)
holdout run --csv examples\store_sales_timeseries.csv --target units_sold --data-type timeseries --unseal --ack-warnings

# Your own plan (see examples\plan_churn_lightgbm.json for the contract)
holdout run --csv examples\churn_classification.csv --target churned --plan examples\plan_churn_lightgbm.json

# Re-run the audits of an existing run (never unseals)
holdout audit --run runs\<run_id>

# Reproduce a run's CV score without Holdout
python runs\<run_id>\pipeline.py --csv examples\churn_classification.csv

# Tests: the fast tier (default, under 4 minutes: 1 seed, no 50k-row tests) ...
pytest
# ... and the full tier (every test and seed, as CI runs it; add -m "not slow" to skip
# the 50k-row performance tests)
pytest --tier full --cov

# Lint and types
ruff check src tests scripts bench
mypy src/holdout bench
```

On Linux or macOS, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

## The agent

```powershell
# Offline: the same graph with the default plan, audits and a template report; no API calls
holdout agent --csv examples\churn_classification.csv --target churned --offline

# With Gemini (set your key in the environment first; it is never written anywhere)
$env:GEMINI_API_KEY = "..."
holdout agent --csv examples\churn_classification.csv --target churned --goal "predict churn"

# Unattended: every question answered by the logged autopilot policy
holdout agent --csv examples\churn_classification.csv --target churned --autopilot

# A run paused at a question (exit code 4) or interrupted continues from its checkpoint
holdout agent --resume <run_id>

# Options: --eda (sandboxed analysis snippets), --no-cache, --trace (OpenTelemetry spans)
```

`holdout run` stays the deterministic, LLM-free path. In agent runs Gemini proposes the
plan, explains audit flags and writes the report's prose; code validates every answer,
applies every audit fix, re-audits, decides whether the holdout may be unsealed, and fills
every number in the report. Spend is capped at Rs 15 per run and Rs 1,200 per project
(ledger in `~/.holdout/spend_ledger.jsonl`); responses are cached in `~/.holdout/cache`.

**Privacy.** Only the schema, summary statistics and at most 5 sample rows (development rows,
never holdout rows) are sent to Gemini; never the full dataset. Under Gemini's free tier,
Google may use inputs to improve its products: for private data use the paid tier or
`--offline`.

**EDA sandbox.** With `--eda`, up to four model-written snippets run on a copy of the
development rows in an isolated process with an import allowlist, a 20 s timeout, a 1 GB
memory limit and a 64 KB output cap. It limits damage from mistakes; it is not a security
boundary.

**Live smoke test** (manual, uses your key, never run in CI):

```powershell
$env:GEMINI_API_KEY = "..."
python -m scripts.smoke_gemini
```
## The audit

Every run audits the cross-validated pipeline on development rows only. The twelve checks
are described, with their blind spots, in [docs/audits.md](docs/audits.md). This is the
audit table from `report.md` for the `direct_leak` trap, where `refund_issued` is derived
from the target:

```powershell
python -c "from bench.traps import build_case; build_case('direct_leak', 0).df.to_csv('direct_leak.csv', index=False)"
holdout run --csv direct_leak.csv --target converted --unseal
```

| check | severity | columns | message | fix |
|---|---|---|---|---|
| A1 | critical | `refund_issued` | 'refund_issued' alone reaches roc_auc 0.969 (dummy 0.500, full plan 0.977): 98% of the plan's lift. A2 also points to leakage, so treat it as a leak. | drop 'refund_issued' |
| A2 | warning | `refund_issued` | 'refund_issued' may be recorded after the outcome (name contains outcome-timing word(s) ['refund']) and it carries signal; check when its value becomes known. | drop 'refund_issued' |
| A10 | info | - | The plan scores 0.977 roc_auc (95% of the way from dummy to perfect). Not a flag by itself; the audit ran in deep mode (all features up to 1000, up to 50000 rows, more group and time candidates). | deep mode applied; see the other checks |

The unseal is refused (exit code 3):

```
critical flag(s): A1 refund_issued (override only with --accept-risk "reason")
1 warning(s) not acknowledged (use --ack-warnings)
```

Rules for unsealing:

- warnings need `--ack-warnings`;
- a critical flag can be overridden only with `--accept-risk "reason"`, which is logged and
  shown at the top of the report;
- a contaminated holdout (rows, entities or time shared with development data) is never
  evaluated: start a new run with the corrected split the audit suggests;
- an incomplete audit also blocks unsealing.

Other run options:

- `--goal "..."` states what you want to learn;
- `--entities new|known` says whether predictions are for new entities or for entities
  already in the data.

## What a run writes

`runs/<run_id>/` contains:

- `pipeline.py`: a standalone script that rebuilds the split and pipeline and prints the CV
  score. It imports only pandas, numpy, scikit-learn and lightgbm.
- `run_manifest.json`: data sha256, profile summary, split spec and reasoning, run seed and
  derived seeds, every plan tried with its CV result, the audit, the unseal decision, the
  thresholds file's sha256, the holdout result (or `"not unsealed"`), the optimism gap,
  package versions and the git commit.
- `report.md`: a short human-readable summary with the audit table.
- `audit.json`: the full audit report with evidence (the latest audit of the run).
- `holdout.parquet`, `holdout_seal.json`, `holdout_access.jsonl`: the sealed holdout, its
  state, and a log of every access attempt.
- `holdout_row_digests.json`: keyed digests of the holdout's feature rows, used only to
  count duplicates across the seal (no values, no labels).
- agent runs only: `checkpoints.sqlite` (LangGraph checkpoints for `--resume`; state only, no
  data, no key) and `agent_options.json`; `eda/` holds sandbox figures when `--eda` is used.

## Layout

```
src/holdout/core/     types, config, seeds, errors, run state, portable kernel
src/holdout/data/     loading, target-blind profiling, split proposal, the seal
src/holdout/plan/     ModelPlan contract, validation, metrics, compiler, defaults
src/holdout/audit/    the twelve checks, thresholds, audit runner, unseal gate
src/holdout/export/   pipeline.py generator, manifest and report
src/holdout/llm/      Gemini client, structured calls, budget, response cache
src/holdout/agent/    LangGraph agent, prompts, planner, report rules, autopilot policy
src/holdout/sandbox/  EDA sandbox (AST allowlist, isolated process, limits)
src/holdout/cli.py    holdout run / holdout audit / holdout agent / holdout make-example
bench/traps/          trap generators with ground truth and clean twins (not shipped)
tests/                unit, property (hypothesis), integration and performance tests
docs/                 design.md (guarantees and decisions), audits.md (the checks)
```

## Windows note

If an import fails with "An Application Control policy has blocked this file", Windows
Smart App Control is blocking a recently released binary wheel. The pinned versions in
`pyproject.toml` are mature releases that pass its reputation check; see
[docs/design.md](docs/design.md#environment). To reproduce the exact tested environment:

```powershell
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```
