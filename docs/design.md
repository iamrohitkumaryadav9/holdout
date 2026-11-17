# Holdout design (Weeks 1-2)

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
7. **Audit** (Week 2) the cross-validated pipeline with twelve checks on development rows
   only (see [audits.md](audits.md)). Flags carry typed evidence and a typed fix.
8. **Unseal (optional, once, gated).** Only if the audit allows it: fit on all development
   rows; `SealedHoldout.evaluate` checks the hash, scores once, and logs the call. A second
   call raises `HoldoutAlreadyUsedError` unless `force=True`, and the forced call is logged
   too.

The seal is a **process-level** guarantee, not cryptographic security: someone with file
access can read the parquet file or start a fresh run. It stops accidental and silent reuse,
and makes deliberate reuse visible in `holdout_access.jsonl` (per run) and
`holdout_ledger.jsonl` (across runs, keyed by data hash; warns rather than blocks).

## The audit layer (Week 2)

After the plan's CV, `holdout.audit.runner.run_audits` runs twelve checks in a fixed order
on the development rows and returns an `AuditReport` (also written to `audit.json` and
embedded in the manifest). [audits.md](audits.md) lists each check's method, severity
rules, fix and blind spots. Four rules shape the layer:

**Unseal gate.** Unsealing is decided in the run orchestrator, not the CLI, so programmatic
callers get the same rules (`holdout.audit.gate`):

1. a contaminated holdout (`holdout_invalid`) is never evaluated; nothing overrides it;
2. an incomplete audit (a check errored) blocks unsealing; not overridable, because an
   unknown is not an accepted risk;
3. critical flags block unsealing unless the user writes a reason with
   `--accept-risk "reason"`; the reason is logged in the manifest and shown at the top of
   `report.md`;
4. warnings must be acknowledged with `--ack-warnings`, even when a risk is accepted.

A refused `--unseal` still writes every artifact and exits with code 3.

**Holdout integrity rule.** Some problems are not in the pipeline but in the split itself:
development rows duplicated in the holdout (A4 above its share threshold), entities shared
between development and holdout (A5 critical), or a holdout drawn without regard to time
(A6 critical). Then the sealed holdout is contaminated and its score would confirm the leak
instead of exposing it. Such a run is marked `holdout_invalid` in the seal state itself, so
`SealedHoldout.evaluate` refuses even with `force=True`. The CLI tells the user to start a
**new** run with the corrected split (the flag's `ChangeSplit` fix). That new run's report
carries a caveat: its holdout was re-drawn after the data had been inspected, so its score
may be slightly optimistic. The caveat comes from the cross-run ledger, which now records
`sealed` and `invalidated` events as well as evaluations.

**A4 exception (the only holdout-side read).** Duplicates across the seal cannot be found
from development rows alone. At seal time, `SealedHoldout.seal` stores, for each holdout
row, an HMAC-SHA256 digest of its normalised *feature* row (floats rounded to 6 significant
digits, strings stripped and lower-cased; never the label) under a random per-run key kept
in the seal state. Plain hashes of low-entropy rows could be reversed by enumeration; keyed
digests cannot without the key. The audit's only access is
`SealedHoldout.holdout_overlap(dev_features)`, which returns counts (development rows
matching, holdout rows matching, holdout rows), never digests, values or labels.
`tests/unit/test_seal_week2.py` asserts what is stored.

**A1 severity reasoning.** Some clean data is near-perfectly predictable from one honest
feature (odor in the mushroom data). Statistics alone cannot tell that from a leak, so A1
alone is at most a warning and becomes critical only when corroborated by A2 (name or values
suggest a post-outcome column) or A3 (the column is ID-like). A train/validation lift gap is
deliberately not a corroborator: a genuine leak generalises within development CV, so it
shows no gap; a gap signals memorisation, which is A3's job.

**Provisional thresholds.** Every threshold lives in `holdout/audit/thresholds.toml` with a
comment, loads into strict typed models (a typo fails), and the file's sha256 is recorded in
every manifest. The values are provisional: they will be tuned in Week 4 on development trap
tasks only, never on evaluation tasks.

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
| Bounded hyperparameters; accuracy on a 10% minority flagged before sealing | `tests/unit/test_plan_schema.py` |
| Each audit check fires on its planted case at the expected severity and stays quiet on the clean twin, across 5 seeds | `tests/unit/audit/test_a*.py` |
| Every trap is detected end to end; no clean twin raises a critical flag | `tests/integration/test_traps_e2e.py` |
| Easy-but-clean data gets at most an A1 warning | `tests/unit/audit/test_runner.py` |
| An invalid holdout refuses `--unseal`, `--accept-risk` and `force=True` | `tests/integration/test_audit_cli.py`, `tests/unit/test_seal_week2.py` |
| An incomplete audit blocks unsealing; a skipped A11 does not | `tests/unit/audit/test_runner.py`, `tests/integration/test_audit_cli.py` |
| The A4 seal-time digests expose no holdout value or label | `tests/unit/test_seal_week2.py` |
| A full run with audits on 50k rows stays under 90 s, deep mode included | `tests/performance/test_audit_budget.py` (slow; Ubuntu CI) |
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

Week 2:

- **Accuracy on an imbalanced target moved from a plan error to an audit critical.** Week 1
  rejected it before sealing; now it is a pre-seal `PlanIssue` warning and an A7 critical.
  The run proceeds so the report can show accuracy next to balanced accuracy and PR AUC
  (secondary CV metrics for plan and dummy), and unsealing stays blocked unless the risk is
  accepted with a reason. Metric/task mismatches and `pr_auc` on multiclass stay hard errors
  (the score cannot be computed).
- **Noise-calibrated split comparisons (A5, A6).** On ~1k rows with a weak model, the
  relative drop under a stricter splitter is noisy: on clean classification twins its
  standard deviation was ~0.16 with a true drop of zero. Each comparison therefore re-runs
  the reference under a second shuffle; a drop counts only if it clears both the threshold
  and 3x the shuffle-to-shuffle difference.
- **Size-matched control for A6.** Forward-chaining folds train early folds on little data,
  so they score lower even on clean data. A6 compares them with the same folds on a shuffled
  order, so the learning-curve effect cancels.
- **Sort confound (A6).** Row-order autocorrelation is blamed on time only if no ordinary
  feature explains the order (near-monotonic in row position, or a categorical in contiguous
  blocks); otherwise it is a warning and never invalidates. Only integer sequences count as
  time indices; a continuous feature is "monotonic" only because the file is sorted by it.
- **Entity scope.** `--entities new|known` is stored in `SplitSpec.entity_scope` (excluded
  from the split fingerprint, since it does not change which rows are held out). Known
  entities make a row-level split legitimate: A5 records info and never invalidates.
  Unanswered is treated as new. Structural entity columns (repeats plus entity-constant
  columns, datetime columns excluded) raise the question before sealing.
- **A2 and A3 "signal" uses the normalized score, not relative lift.** Relative lift is
  noise-dominated when the full plan is weak (a random ID once reached 19% of a weak plan's
  lift by chance).
- **A1's relative-lift rule needs a strong plan** (at least halfway from dummy to perfect).
  Carrying most of a small lift is what an honest dominant feature looks like.
- **A3 considers only integer or string columns ID-like by structure.** Continuous
  measurements are naturally near-unique.
- **Float rounding for row hashing is a core constant** (6 significant digits, recorded in
  each seal), not an audit threshold, because the holdout's digests are computed at seal
  time, before any audit runs.
- **`run_audits` takes its context explicitly** (`baseline`, `profile`, `spec`, `target`,
  `overlap_source`, ...) in addition to the spec's `(run_state, dev_df, plan, cv_result)`.
- **A12 measures spread against the claimed lift** (std / |mean - dummy|), not only the plain
  coefficient of variation (recorded too), which is scale-dependent and meaningless near
  AUC 0.5 or R2 0.
- **Trap design.** `id_with_signal` issues IDs in target order and then shuffles the file (a
  target-sorted file makes the ID monotonic, which Week 1 already drops). `time_order`
  bases have features that drift with time: with the date removed and no time information
  in any feature, a random split cannot exploit time either, so there would be no leak to
  find. Traps are pure functions grouped in three modules (`leaks`, `structure`, `quality`)
  with a registry (`catalog`), and were built before the checks so that each check's tests
  use the same planted cases the Week 4 benchmark will.
- **`holdout audit --run`** rebuilds the development rows by re-applying the recorded split
  to the recorded CSV, verified by the CSV's sha256 and a development-row fingerprint stored
  at seal time; it never reads the holdout file. The latest audit replaces `audit.json`; the
  manifest keeps the audit made at run time.

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
- `HintProvider` (A2) is the seam for an LLM hint provider (Week 3); the offline
  `LexiconHintProvider` stays the fallback.
- `check_causal_claims` (A11) is ready for the generated report (Week 3).
- `bench/traps` provides planted cases with ground truth for the Week 4 benchmark.
