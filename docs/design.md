# Holdout design (Week 1)

## Problem

"The Silent Failures in Agentic Data Science" (Li, Xia, Gomm, Santos, Wolff, Hulsebos; CWI;
arXiv:2610.14115, Oct 2026) reports that LLM data-science agents leak validation data, and
that the leakage rises with model capability. A more capable agent is better at finding a
higher validation score, including the illegitimate ways to get one.

> Reference note: this citation was found through search results and is given as found.
> Check it against its arXiv page (https://arxiv.org/abs/2610.14115) before citing it
> further.

Holdout's answer is that correctness is enforced in code. A proposer (the offline default
today, an LLM later) suggests; deterministic code decides.

## Seal before see

The order of operations is the main guarantee:

1. **Load** the CSV and hash its bytes (sha256). Rows with a missing target are dropped and counted.
2. **Profile** the schema without looking at the target. Per-column facts only; the only
   target statistics are its own distribution (task hint, class frequencies for
   stratification).
3. **Propose a split** from the profile: time, group, stratified or random, with reasons and
   warnings.
4. **Validate the plan** against the profile. A rejected plan stops the run *before*
   anything is sealed.
5. **Seal.** The split happens inside `SealedHoldout.seal`, which writes the holdout to
   `holdout.parquet`, records its sha256, and returns only the development rows. No caller
   ever holds the holdout DataFrame.
6. **Cross-validate** the dummy baseline, then the plan, on development rows only, with folds
   that mirror the holdout split (group -> `GroupKFold`, time -> `TimeSeriesSplit`).
7. **Unseal (optional, once).** Fit on all development rows; `SealedHoldout.evaluate` checks
   the hash, scores once, and logs the call. A second call raises
   `HoldoutAlreadyUsedError` unless `force=True`, and the forced call is logged too.

The seal is a **process-level** guarantee, not cryptographic security: someone with file
access can read the parquet file or start a fresh run. It stops accidental and silent reuse,
and makes deliberate reuse visible in `holdout_access.jsonl` (per run) and
`holdout_ledger.jsonl` (across runs, keyed by data hash; warns rather than blocks).

## Why the agent fills a typed plan instead of writing code

Free-form modelling code can fit a scaler on all rows, tune on the test fold, or read the
holdout, and none of it is visible until it has already happened. A `ModelPlan` can only
*choose* among options whose implementations are known to be leakage-safe:

- discriminated unions for models and feature selection; each model has its own bounded
  hyperparameter sub-model; unknown fields are rejected;
- every dropped column needs a reason; every choice needs a one-line rationale;
- shape validation (pydantic) needs no data; context validation (`validate_plan`) returns
  typed `PlanIssue`s (code, field, message) that a proposer can act on, e.g. "accuracy on a
  10% minority class".

The compiler places every learned step (imputers, scaler, encoder, selector, model) inside
one scikit-learn `Pipeline`, so each CV fold fits a fresh clone on its training rows only.
`target_cv` uses `TargetEncoder`, which cross-fits internally.

## How each guarantee is tested

| Guarantee | Test |
|---|---|
| Profile computes no feature-target statistic | `tests/property/test_profile_invariance.py`: the profile is identical after shuffling the target (hypothesis, 4 datasets), plus a positive control showing a correlation statistic *would* change |
| No holdout row in any CV fold | `tests/property/test_cv_properties.py` (hypothesis over data, seeds, all strategies) |
| Every fit sees only its fold's training rows; no fit ever sees a holdout row; evaluating the holdout fits nothing | `tests/integration/test_leakage_spy.py`: a spy transformer records hidden row ids at every fit (4 strategies x 2 plans covering every learned step), plus a negative control where a deliberately leaky CV loop is caught |
| Same seed, same split, same CV scores (exact) | `tests/property/test_cv_properties.py`, `tests/property/test_split_properties.py` |
| Group split never shares a group; time split has `max(dev time) <= min(holdout time)` | `tests/property/test_split_properties.py` |
| One evaluation per run; tamper detection; every access logged | `tests/unit/test_seal.py` |
| Bounded hyperparameters; accuracy rejected on a 10% minority | `tests/unit/test_plan_schema.py` |
| Exported `pipeline.py` reproduces the CV score without Holdout | `tests/integration/test_export_reproduces.py`: runs the script with `python -I` in a subprocess, matches to 1e-9 (3 strategies, 3 models) |
| CLI works end to end, offline | `tests/integration/test_cli.py` on the `make-example` CSVs |

## Decisions where the spec left room (the more rigorous option was taken)

- **One kernel, embedded verbatim.** `holdout/core/portable.py` holds the split, pipeline
  builder and scorers, and imports only the standard library, pandas, numpy, scikit-learn
  and lightgbm (an AST test enforces this). The exporter embeds its source in `pipeline.py`,
  so the exported script cannot drift from what Holdout ran.
- **The exported script re-applies the split** and discards the holdout rows unseen, because
  the CV score was computed on development rows only. It warns if the CSV's sha256 differs.
- **Holdout state is persisted** (`holdout_seal.json`), so a new process cannot reset the
  evaluation count. A failed scoring attempt still counts as a use, because holdout rows
  reached the pipeline.
- **The plan is validated before sealing**, so a rejected plan leaves no sealed run behind.
- **Model-task pairs are enforced**: `logreg` is classification-only and `ridge` is
  regression-only (`RidgeClassifier` has no probabilities, so AUC metrics would be invalid).
- **Explicit drops.** Datetime, free-text and id-like columns must be dropped with a reason;
  the Week 1 pipeline does not featurize them, and silently ignoring them would hide choices.
- **Determinism.** LightGBM runs with `deterministic=True, force_row_wise=True, n_jobs=2`;
  `HistGradientBoosting` runs with `early_stopping=False` (its automatic early stopping
  carves a data-size-dependent internal split). Seeds come from `derive_seed` (sha256).
- **Metric conventions.** The binary positive class is `classes_[1]` (scikit-learn's
  convention); multiclass `roc_auc` is one-vs-rest and `f1` is macro; `pr_auc` is rejected
  for multiclass. Scores are reported in their natural sign (RMSE is positive).
- **Group split `test_size` is a fraction of groups**, not rows (`GroupShuffleSplit`); the
  manifest records the actual row counts.
- **`apply_split(df, spec, target)` takes the target** (needed for stratification), and
  `propose_split` takes the run seed as a keyword argument. Both are small signature
  additions to the spec.

## Environment

The numeric stack is pinned to mature releases (numpy 2.3, pandas 2.3, scikit-learn 1.7,
LightGBM 4.6, pyarrow 21). On the development laptop, Windows Smart App Control blocked
DLLs from the newest wheels (numpy 2.5, pyarrow 26, and the native extension of recent
hypothesis releases). Older, widely installed builds pass its reputation check. CI installs
from `pyproject.toml`; `requirements.lock` records the exact local environment.

## Seams for later weeks

- `runner.execute_run` takes a plan path or falls back to `default_plan`. An LLM proposer
  plugs in as another source of `ModelPlan`s, with `validate_plan` issues fed back to it.
- `RunObserver` decouples progress reporting from the CLI (a Streamlit UI implements it).
- The audit checks will read `run_manifest.json`, `holdout_access.jsonl` and the ledger.
