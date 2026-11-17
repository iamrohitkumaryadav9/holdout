# Holdout design (Weeks 1-3)

## Problem

"The Silent Failures in Agentic Data Science" (Li, Xia, Gomm, Santos, Wolff, Hulsebos; CWI;
arXiv:2610.14115, Oct 2026) reports that LLM data-science agents leak validation data, and
that the leakage rises with model capability. A more capable agent is better at finding a
higher validation score, including the illegitimate ways to get one.

> Reference note: this citation was found through search results and is given as found.
> Check it against its arXiv page (https://arxiv.org/abs/2610.14115) before citing it
> further.

Holdout's answer is that correctness is enforced in code. A proposer (Gemini in agent runs,
the offline default otherwise) suggests; deterministic code decides.

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

**Benchmark contamination guard.** Thresholds were tuned on synthetic trap cases. Every
base dataset and seed used during development or tuning is listed in
`bench/dev_usage.toml`, and the final-suite builder (`bench.suite.build_final_suite`) refuses
any listed base or seed before loading data. The Week 4 benchmark uses held-out OpenML bases and
seeds never seen during tuning.

**Provisional thresholds.** Every threshold lives in `holdout/audit/thresholds.toml` with a
comment, loads into strict typed models (a typo fails), and the file's sha256 is recorded in
every manifest. The values are provisional: they will be tuned in Week 4 on development trap
tasks only, never on evaluation tasks.

## The agent (Week 3)

`holdout agent` wraps the deterministic core in a LangGraph state machine with a SQLite
checkpointer (`runs/<id>/checkpoints.sqlite`):

intake -> confirm_split (interrupt) -> seal -> [eda] -> plan -> compile_and_fit -> audit,
then either end_split (the split leaks), or explain_and_fix -> review (interrupt) ->
compile_and_fit again (at most 3 rounds), and finally confirm_unseal (interrupt) -> unseal
-> report.

**The LLM proposes; code decides.** Gemini (google-genai SDK) proposes a `ModelPlan`,
explains audit flags, integrates their fixes, writes report prose, suggests A2 hints and
(optionally) EDA snippets. Every call returns a Pydantic model through the SDK's JSON-schema
output; an invalid answer gets exactly one repair request quoting the errors, then the run
fails with a typed error. There is no silent fallback to a default plan inside an agent run;
`--offline` is a separate, explicit path that runs the same graph with the default plan,
code explanations, template prose and lexicon hints (and no API calls).

**Code-applied fixes and substitutions.** Every audit fix that changes a plan (drop columns,
change the metric) is applied by code: that is the source of truth. The LLM's integrated plan
is accepted only if it contains every fix and validates. If it does not (after one repair),
the code-applied plan is used and a substitution is recorded (node, round, the fixes the
model missed). The manifest lists every substitution and `report.md` shows the count,
because a remediation achieved by the code fallback must not be credited to the LLM; the
Week 4 benchmark reports the substitution rate.

**Numbers come from state.** The report's prose is written by the LLM around placeholders
(`{{validation_score}}`, `{{holdout_score}}`, ...) that code fills from run state. A numeric
literal in the prose is rejected (one repair) unless it is an allow-listed identifier: check
ids (A1-A12), column names (`feature_12`), metric and model names (`f1`), plan field values,
or identifiers inside code-written caveats (run ids). Tests check that every number in a
report's prose is a formatted run-state value.

**Data is data.** Everything taken from the CSV or the user (column names, sample values,
goal text, earlier answers) reaches a prompt only through `<data name="...">` sections, with
closing tags escaped so data cannot break out, and every system prompt says that
instructions inside data must be ignored. The model sees the schema, summary statistics
(target-blind, as in Week 1) and at most 5 development rows (never holdout rows). A
prompt-injection test (a column named like an instruction, a goal text full of
instructions) checks that injected text stays inside data sections, and that a model that
obeys it still cannot get past validation, the audit and the unseal gate.

**A11 on the report.** In agent runs A11 checks the generated report's text and its
`causal_claim` field. If the draft makes a causal claim, the flag is recorded and the model
is asked once to remove it; if it does not, code strips the causal sentences, adds the
caveat, and records a substitution. Offline runs keep Week 2's recorded skip.

**Interrupts and autopilot.** Interrupts ask a person (rich prompts) or follow the autopilot
policy, a scripted, logged policy for unattended runs and the benchmark: accept the proposed
split; entity scope "new" unless declared; accept the agent's fixes; request unsealing with
warnings acknowledged by policy and no accepted risk, so the gate unseals only with no open
critical flag. Each autopilot decision is recorded exactly like a human one. With no
terminal, an interactive run pauses (exit code 4) and `holdout agent --resume RUN_ID`
continues it, in a new process if need be. If the split leaks, the run ends; under autopilot
exactly one new run starts with the corrected split, and the two manifests link to each
other (the new one carries the re-drawn-holdout caveat).

**Loop guards (the AgentRAG infinite-loop lesson).** An agent loop without hard stops can
cycle indefinitely. Holdout's graph has: at most 3 fix rounds;
one plan retry with validation feedback, then a typed failure; a stop when a revised plan
repeats an earlier one (plans are compared by hash); and a per-node wall-clock limit. The
limit is checked when a node returns, because Python cannot safely kill a running thread;
LLM calls are additionally bounded by the SDK request timeout (120 s), and sandbox snippets
by their own 20 s timeout.

**Crash safety.** State holds only JSON-serialisable values; development rows are rebuilt
from the hashed CSV and the split spec in each node, so a checkpoint never contains data.
Nodes with side effects are idempotent: a resumed seal node reattaches to the existing seal,
and a resumed unseal node reads the earlier evaluation from the access log instead of
evaluating again. A test kills the process mid-run and resumes it to a manifest identical to
an uninterrupted run (apart from timestamps).

**Model, thinking, prices (checked 2026-11-17).** The model name lives only in
`holdout/llm/config.py`: `gemini-3.8-flash`, the current stable Flash model
(https://aistudio.google.com/docs/models). Thinking is capped with the Gemini 3 parameter
`thinking_config.thinking_level = "low"` (https://aistudio.google.com/docs/thinking; 3.8
Flash cannot switch thinking off), plus `max_output_tokens = 4096`, which bounds thinking
and answer together. The run seed is passed as `GenerateContentConfig.seed` (the SDK
supports it; Google calls it best effort) and is part of the cache key, so two seeds never
share a cached response. Prices: USD 0.75 per 1M input and 3.75 per 1M output tokens
(thinking billed as output) through 2026-12-31, doubling from 2027-01-01
(https://aistudio.google.com/docs/pricing); 94.1 INR per USD as of 2026-11-13. Every manifest
records the model, whether it was verified against the SDK's model list, the client class,
the thinking level, prompt versions and hashes, and tokens, rupees and cache hits per node.

**Budget.** Real token counts from each response's usage metadata are priced from the dated
table. Before each call, a worst-case estimate (the prompt plus every allowed output token)
must fit both caps, Rs 15 per run and Rs 1,200 per project, so a single call can never
overshoot. The project ledger lives outside the repository (`~/.holdout/spend_ledger.jsonl`)
and persists across runs and processes. The API key is read from `GEMINI_API_KEY` only and is
never logged, cached, checkpointed or written to a manifest; a test plants a sentinel key
and scans every artifact, the checkpoint database, the cache and the ledger for it.

**EDA sandbox.** Optional (`--eda`): up to 4 model-written snippets run on a copy of the
development rows in a fresh `python -I` process in an empty temporary directory, behind an
AST allowlist (pandas, numpy, scipy.stats, matplotlib), a runtime import guard and pruned
builtins, a 20 s timeout, a 1 GB memory limit (psutil watchdog on Windows; `RLIMIT_DATA` and
`RLIMIT_CPU` plus the watchdog on Linux) and a 64 KB output cap. It is a damage limiter, not
a security boundary.

**Tracing.** OpenTelemetry spans per node carry call counts, tokens and rupees; the console
exporter is off by default (`--trace` turns it on). Spans never carry prompts or data.
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
| The agent detects every trap and keeps every clean twin free of criticals (autopilot, fake LLM) | `tests/integration/test_agent_traps.py` |
| Every number in an agent report's prose is a run-state value | `tests/integration/test_agent_traps.py`, `tests/unit/agent/test_report_rules.py` |
| Injected instructions stay inside data sections; an obedient model still cannot pass validation, audit and gate | `tests/integration/test_agent_behaviour.py` |
| Malformed or random plan JSON is rejected or repaired, never compiled unvalidated | `tests/unit/llm/test_llm_core.py` (hypothesis) |
| Budget caps stop a run before the call; the ledger persists across processes | `tests/unit/llm/test_llm_core.py`, `tests/integration/test_agent_behaviour.py` |
| Two seeds never share a cached response | `tests/unit/llm/test_llm_core.py`, `tests/integration/test_agent_behaviour.py` |
| The API key never appears in any artifact, checkpoint, cache or ledger | `tests/integration/test_agent_behaviour.py` |
| A run killed mid-graph resumes to an identical manifest; the holdout is evaluated once | `tests/integration/test_agent_resume.py` |
| Tests never touch the network | autouse socket block in `tests/conftest.py` |
| Sandbox: every blocked construct rejected; timeout and memory kills work; no holdout path or key inside | `tests/unit/test_sandbox.py` |
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
- **A12 (revised in Week 3)** warns only when the fold spread could change the conclusion:
  a fold at or below the dummy, or a Nadeau-Bengio corrected 95% interval of the per-fold
  lift that includes zero (plan and dummy share seeded folds, so differences are paired).
  Otherwise it records info with the spread. Week 2's rule (spread relative to the lift)
  warned on honest small-data runs whose conclusion was not in doubt.
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

Week 3:

- **Only critical flags change the plan automatically.** A plan fix (drop columns, change
  the metric) is applied when its flag is critical. Warnings are explained and must be
  acknowledged, but not auto-applied: an uncorroborated A1 warning can be an honest, very
  strong feature (the mushroom odor case), and dropping it automatically would make clean
  near-separable data worse.
- **Offline runs use the same graph** with an offline planner, so their manifests are
  comparable with agent runs; `holdout run` stays the deterministic, LLM-free path.
- **Only valid LLM answers are cached**, and the cache key includes the run seed; tests turn
  caching off by default because the key does not (and must not) identify the test double.
- **FakeLLM is priced like the real model** (token counts approximated as characters / 4),
  so budget tests are realistic; the manifest records the client class separately.
- **A resumed run keeps the mode it started in** (`agent.mode`); decisions record the mode
  that made them, so a run started interactively and resumed under autopilot shows both.
- **Code-written caveats are shown by code**, not echoed by the model; identifiers inside
  them (run ids) are allow-listed for the number rule.
- **Test tiers.** `pytest` runs a fast tier (1 seed per planted/clean test, fewer hypothesis
  examples, heavy end-to-end variants skipped) in under 4 minutes; `pytest --tier full` runs
  every test and seed, and is what CI runs.
- **The EDA sandbox was built before the graph** (Part 4 before Part 3's EDA node), because
  the node calls it.
## Environment

The numeric stack is pinned to mature releases (numpy 2.3, pandas 2.3, scikit-learn 1.7,
LightGBM 4.6, pyarrow 21). On the development laptop, Windows Smart App Control blocked
DLLs from the newest wheels (numpy 2.5, pyarrow 26, and the native extension of recent
hypothesis releases). Older, widely installed builds pass its reputation check. CI installs
from `pyproject.toml`; `requirements.lock` records the exact local environment.

Week 3 dependencies were checked in a scratch virtual environment before installing:
google-genai 3.4, langgraph 1.2 with langgraph-checkpoint-sqlite 3.1, opentelemetry 1.45,
psutil 7.2 and matplotlib 3.11 (with kiwisolver, contourpy and Pillow) all load.
**xxhash 4.1** (a langgraph dependency) was blocked by Smart App Control; **xxhash 3.6**
loads, so `pyproject.toml` pins `xxhash>=3.6,<3.7`. Matplotlib loads, so plotting stays in
the EDA sandbox allowlist.

## Seams for later weeks

- `runner.execute_run` takes a plan path or falls back to `default_plan`. An LLM proposer
  plugs in as another source of `ModelPlan`s, with `validate_plan` issues fed back to it.
- `RunObserver` decouples progress reporting from the CLI (a Streamlit UI implements it).
- `Planner` separates who proposes (Gemini, offline) from the graph; other models plug in
  behind `LLMClient`.
- `RunObserver` and the agent's `Asker` decouple the UI (a Streamlit app implements them).
- `bench/traps` provides planted cases with ground truth for the Week 4 benchmark.
