# Holdout

Holdout is a self-auditing data-science agent for tabular CSVs. LLM data-science agents
have been shown to leak validation data, and to leak more as they get more capable
([Silent Failures in Agentic Data Science](docs/design.md#problem)). Holdout's position is
that correctness should be enforced by code, not by a prompt. The holdout set is split off,
written to disk and hashed before anything computes a feature-target statistic. It can be
scored once per run, through one logged call. Proposals (today an offline default, later an
LLM) are typed `ModelPlan`s with bounded choices; deterministic code validates them,
compiles them into leakage-safe scikit-learn pipelines, and exports a standalone script that
reproduces the result without Holdout installed.

**Status: Week 1 of 6.** Offline core: profiling, split proposal, sealed holdout, typed
plans, cross-validation, export and CLI. No LLM, audit checks or UI yet.

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

# Classification: stratified split, dummy baseline, default plan, one holdout evaluation
holdout run --csv examples\churn_classification.csv --target churned --unseal

# Time-ordered data declared as tabular: prints the time-order warning and a time-split alternative
holdout run --csv examples\store_sales_timeseries.csv --target units_sold

# The same data split by time
holdout run --csv examples\store_sales_timeseries.csv --target units_sold --data-type timeseries --unseal

# Your own plan (see examples\plan_churn_lightgbm.json for the contract)
holdout run --csv examples\churn_classification.csv --target churned --plan examples\plan_churn_lightgbm.json

# Reproduce a run's CV score without Holdout
python runs\<run_id>\pipeline.py --csv examples\churn_classification.csv

# Tests, lint, types
pytest --cov
ruff check src tests scripts
mypy src/holdout
```

On Linux or macOS, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

## What a run writes

`runs/<run_id>/` contains:

- `pipeline.py`: a standalone script that rebuilds the split and pipeline and prints the CV
  score. It imports only pandas, numpy, scikit-learn and lightgbm.
- `run_manifest.json`: data sha256, profile summary, split spec and reasoning, run seed and
  derived seeds, every plan tried with its CV result, the holdout result (or
  `"not unsealed"`), the optimism gap, package versions and the git commit.
- `report.md`: a short human-readable summary.
- `holdout.parquet`, `holdout_seal.json`, `holdout_access.jsonl`: the sealed holdout, its
  state, and a log of every access attempt.

## Layout

```
src/holdout/core/     types, config, seeds, errors, run state, portable kernel
src/holdout/data/     loading, target-blind profiling, split proposal, the seal
src/holdout/plan/     ModelPlan contract, validation, metrics, compiler, defaults
src/holdout/export/   pipeline.py generator, manifest and report
src/holdout/cli.py    holdout run / holdout make-example
tests/                unit, property (hypothesis) and integration tests
docs/design.md        guarantees, how each is tested, and design decisions
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
