# Pre-registration: Holdout benchmark v1

**Status: registered.** This document is binding from the commit tagged `prereg-v1`.
Nothing that affects results may change without a dated amendment at the end of this file.
Each amendment says what changed, why, and whether any final result had been seen.

- Written and reviewed 2026-11-18, before any system has run on `final_v1` or
  `real_world_v1`.
- Definitions of systems, suites, holdouts and scoring: [benchmark.md](benchmark.md).
- Design decisions and the one-time tuning: [design.md](design.md).

## 1. Hypotheses

The unit of every test and CI is the **task**. Seeds are averaged within a task, or, for
McNemar, a task counts as detected when it is detected in both seeds. Anonymised variants
are never pooled with originals.

| | hypothesis | data | primary metric | test | decision rule |
|---|---|---|---|---|---|
| **H1** | Holdout's detection recall on planted traps exceeds ReAct's. | `final_v1`, 28 trap tasks | detection in both seeds, per task | exact McNemar (binomial on discordant tasks), two-sided | **supported** if Holdout has more discordant wins and the Holm-adjusted p < 0.05 |
| **H2** | Holdout's critical false-alarm rate on clean controls is at most 10%. | 12 clean twins in `final_v1` + 4 real-world clean controls = **16 clean tasks** | a task is a non-pass if any seed or variant raised a critical issue **or failed** | none: decided by the rate rule; the exact one-sided binomial p (H0: rate ≥ 10%) and the Clopper–Pearson upper bound are descriptive only | **pass** if the observed rate ≤ 10%, i.e. at most 1 of 16 |
| **H3** | Holdout's optimism gap is smaller than ReAct's. | `final_v1`, every task where both systems have a gap | **primary: `optimism_gap_deploy`**; secondary: `optimism_gap_iid` | paired sign-flip permutation test on the per-task difference (Holdout − ReAct), two-sided | **supported** if the mean difference is negative and the Holm-adjusted p < 0.05 |
| **H4** | Deterministic audits beat the LLM judge on detection. | `final_v1`, 28 trap tasks | as H1, `holdout` vs `llm_judge` | as H1 | as H1 |
| **H5** *(exploratory)* | Real-world detection does not depend on column names. | `real_world_v1` issues, original vs anonymised | per-issue detection rate (seeds averaged) | paired bootstrap 95% CI of original − anonymised | "no material dependence" if the CI lies within ±0.20; outside the Holm family |

**H4 is the ablation.** `llm_judge` is Holdout with only its audit layer replaced by a
Gemini verifier. Same planner, compiler, fix loop and unseal gate. The comparison
isolates the audit layer.

**Multiplicity.**
- **Family.** Holm–Bonferroni over **H1, H3 and H4** at α = 0.05.
- **H2 is not in the family.** It is decided by its pre-stated observed-rate rule, so no
  p-value plays a role in its decision. This corrects the earlier "Holm covers H1–H4",
  which was written before H2 became a rate rule.
- **H5** is exploratory.

**H2's bound and p, stated as decided.** With 16 clean tasks, the one-sided 95%
Clopper–Pearson upper bound cannot fall below about 17% even with zero false alarms. The
exact binomial p cannot fall below 0.9¹⁶ = 0.185. Both are **reported next to the rate as
descriptive statistics only**.

**Estimates reported with every hypothesis.**
- **H1, H4:** the mean per-task difference in detection rate, with a paired bootstrap 95% CI
  (10,000 resamples, seed 0).
- **H2:** the rate, with its Clopper–Pearson bound and binomial p (descriptive).
- **H3:** the mean per-task gap difference, with a paired bootstrap CI.
- **H5:** the mean difference with its CI.

**Implementation.** The analysis code is fixed at this tag: `bench/hypotheses.py` (H1–H5,
Holm) and `bench/score.py` (detection, remediation, false alarms, gaps, McNemar,
Clopper–Pearson, bootstrap). It is tested on hand-built results in
`tests/unit/test_bench_hypotheses.py` and `tests/unit/test_bench_scoring.py`.
`python -m bench.analyze` prints the H1–H5 table first, then each suite's unit failure
rates.

## 2. Secondary and descriptive measures (no hypothesis tests)

- **Detection per trap type**, and "either seed" McNemar for every pair of systems.
- **Detection at critical severity.** For the real-world issues this is the secondary
  level; their primary level is warning.
- **Remediation rate.** The final pipeline actually fixed the issue.
- **False alarms** at warning level and at any level (info included).
- **Holdout scores** relative to the dummy and to FLAML, on both holdouts. Real-world
  cases marked `holdout_scored = false` (Bank Marketing, Titanic) are excluded.
- **Cost and quality:** cost and time per run, calls, retries, first-attempt validity,
  repairs, substitutions, Flex tier served, cached share.
- **Timeouts per system.** If any ReAct sandbox step timed out, the report says so at the
  top as a possible handicap.
- **A11 flag rate** on reports of non-causal tasks. This is an upper bound on its
  false-positive rate; the flagged sentences are listed for human labels.

## 3. Scoring rules fixed here

- **Detection.**
  - *Planted traps:* the expected category is reported at severity ≥ the trap's minimum.
  - *Real-world issues:* one of the issue's categories is reported at ≥ warning, naming
    one of its columns when it has any. See benchmark.md, "Scoring".
- **Failed units count against the system everywhere.** A unit fails when a system
  crashes or a node exceeds its compute limit.
  - *On a trap:* the unit counts as not detected and not remediated.
  - *On a clean task:* for H2 and every false-alarm metric (critical, warning, any), it
    counts as a **non-pass**. Failing therefore cannot shrink the denominator or improve
    the false-alarm rate.
  - *Reporting:* each system's unit failure rate is shown near the top of every suite
    section of `docs/results.md`.
- **Paused units are not results.** The run is repeated until every unit is completed,
  stopped or failed.
- **Stopped units are scored as they are.** A unit stops when it hits the per-run cap or
  produces no final answer, and is scored on whatever issues it reported.
- **No unit is re-run after it completes, stops or fails**, except under an amendment for
  a harness bug. `--retry-failed` exists for that case only, and every use is listed in
  the amendments.
- **Neutralisation rules** for `holdout_deploy` are per trap type
  (benchmark.md, "The two holdouts").
  - The real-world deploy handling was recorded in `real_world_v1.toml` before any system
    ran on it.
  - Bank Marketing and Titanic are scored on detection and remediation only.
  - Electricity and Covertype use the later time block.
  - Diabetes uses unseen patients.
- **`time_order`** in `final_v1` is **semi-synthetic** and is labelled so everywhere.

## 4. Expected misses, stated before the run

The one-time dev review (design.md, "One-time tuning") found two blind spots of the
deterministic path. They are stated here so they cannot be explained away afterwards.

- **Semi-synthetic `time_order`: A6 is expected to miss it.** Its lag-1 row-order gate
  cannot see a smooth trend. On dev, z was 0.65 and 1.21. With the gate forced open, the
  forward-chaining drop was 7%, below the 20% critical rule. This affects `audits_only`
  and Holdout's audit layer. Holdout can still catch it if its LLM planner chooses a time
  split, but no such result is assumed.
- **`causal_question`: `audits_only` cannot detect it.** It does not read the goal.
- **Neutral-named leaks** (`noisy_proxy`) get only an A1 warning, so they are not removed
  automatically. Their optimism gap is expected to resemble FLAML's for `audits_only`.
- **A11's recall is untested going in.** It could not be measured on the dev suite: none of
  the 14 dev reports contained a causal claim. Its performance on `causal_question` in
  the final run is therefore untested. Only its false-positive behaviour was checked:
  there were no flags on those 14 reports.

## 5. What is frozen

| item | value |
|---|---|
| suite `final_v1` (sha256, LF) | `d4de5207759396fc4888e4552eb8792168c240638603e875d0b92f5cf3049b8b` |
| suite `real_world_v1` | `df2f25beee8375ecae432a1a4ed4a67d850f8931fcbf73877c46c59544b8658e` |
| `real_world_v1` anonymisation | `f00d3c97e2bd68c2249620588ae55428427e4549f00c1c1282e038c20130a020` |
| suite `dev_v1` (not part of the evaluation) | `cb17bb2702d08b7e3924dc812ed6baeae533f8c9542861852c0e15a1e072f9cf` |
| audit thresholds `src/holdout/audit/thresholds.toml` | `99a8905a1004207b9c314a99d6ab10da88be232a5786fb35819e04e4348287d6` |
| A11 lexicon `src/holdout/audit/checks/a11_causal.py` (LF) | `068868a9a90b4e59c8c22682860a1906e6998b0592d067eae0c8a81bd2140e97` |
| benchmark config `bench/config.toml` (LF) | `86192a1953de043db479abc2bb215c639d65cc7503d1f603373f59e75fd08b37` |
| model | `gemini-3.8-flash`, temperature 0, thinking level low, seed passed |
| service tier | **Flex** for holdout, llm_judge and react (15-minute request timeout) |
| per-run cap | Rs 15 per agent run; a Holdout-family unit can spend up to two caps (one follow-up run) |
| ReAct | at most 15 steps; 20 s per step, 60 s per model-fitting step; observations truncated to 2,000 characters, head and tail; append-only prompt |
| FLAML | 60 s budget; LightGBM, random forest, extra trees, L1 logistic regression |
| seeds | `final_v1`: 101, 202; `real_world_v1`: 101, 202 |
| code | the commit tagged `prereg-v1` |

**Prompts** (version and sha256 of the file as committed):

| prompt | version | sha256 |
|---|---|---|
| agent `plan` | 2 | `8cd3f04cc460fdddb3453acbbebe6d12dffa770833f23fc4d7f80f9403ddccb0` |
| agent `eda` | 1 | `bbe674190b984343dc3f16a806cb12375968824bebe10dd75a2ec1485d6d329c` |
| agent `explain_fix` | 2 | `2432e069e9937b89e937bfbc39c37ddcbf9e61badc49b0ef4fe62357277b4898` |
| agent `report` | 2 | `dbca98b269108fc9358ce226d9743d1e1aded008b12c038e71e323ca9f33da27` |
| agent `hints` | 1 | `cc224cfeb13c6e316e34f137cdfafa327b6b4a00f30b22536ddbc516af50b282` |
| bench `react` | 2 | `24db1e7bc51b9b8e55dfebc8180da35b20c51ecca99ec22ee4516cfdfcafd57f` |
| bench `judge` | 1 | `2035f27e7bd2aaa729fbb1249de0c010e8d12c8d1d3f2e88564854af713d5dac` |
| bench `judge_report` | 1 | `d1b35eddb585c715dd3d580fa6632aeec10badbba13f5a5f7070b46c8e1e76c7` |

Every run manifest records the thresholds hash and each prompt's version and hash, so any
drift is visible per result.

## 6. How the final run is executed

**Machine:** the benchmark laptop, 8 logical cores, Windows 11, Python 3.12.15 (`.venv`).

**Environment:**
- `GEMINI_API_KEY` is read from the environment only.
- `HOLDOUT_PROJECT_CAP_INR=1000`.
- `HOLDOUT_TRUST_CACHE_DISCOUNT` is unset, so caps use list prices.
- The prepaid balance is recorded with `holdout spend --set-balance` before the start.

**Parallelism and CPU fairness:**
- **LLM units:** `--workers 4`. Each worker process is limited to 8 // 4 = **2 native
  threads** through the OpenMP, MKL, OpenBLAS, numexpr and Accelerate variables, plus
  `LOKY_MAX_CPU_COUNT`. threadpoolctl caps libraries already loaded. ReAct's sandboxes
  inherit the limit.
- **CPU-only units** (flaml, audits_only) run afterwards in a separate pass, one at a time,
  alone on the machine.
- **Recorded per result:** the worker count, the thread limit and the core count.

**Commands, in this order.** Rerun a command unchanged after any pause until it reports no
pending units.

```text
python -m bench.run --suite final_v1 --purpose benchmark --workers 4 --plan
python -m bench.run --suite final_v1 --purpose benchmark --workers 4
python -m bench.run --suite real_world_v1 --purpose benchmark --workers 4
python -m bench.analyze --results results --out docs/results.md --charts docs/img
```

**Budget safety during the run:**
- **Whole-benchmark pre-flight.** Any run on either final suite first checks the whole
  benchmark, both suites and every system, against the cap and the prepaid balance, with
  headroom for the units in flight. It refuses to start unless everything fits.
- **Unit-start guard.** A unit starts only if every limit still covers its worst case
  (Rs 15 for ReAct, Rs 30 for holdout and llm_judge) plus the worst case of every unit
  already running.
- **No overshoot.** Spend can never exceed the cap.
- **If the guard stops the run before the end,** it is resumed only after a top-up recorded
  as an amendment. No hypothesis is evaluated on a partial run.

## 7. Cost projection

**Pilot basis** (3 dev tasks, seed 11, 2026-11-18, standard tier), rescaled to Flex (× 0.5),
plus the 20% margin, × 116 units per system:

| system | measured Rs/run (standard) | projected (Flex, +20%) |
|---|---|---|
| holdout | 1.084 | Rs 75.44 |
| llm_judge | 1.179 | Rs 82.04 |
| react | 8.100 | Rs 563.78 |
| audits_only, flaml | 0 | Rs 0 |
| **total** | | **Rs 721.26** |

- **Headroom.** The unit-start guard needs Rs 120 for 4 workers × Rs 30, so the pre-flight
  requires **Rs 841.26**.
- **Available** on 2026-11-18: Rs 957.83 under the Rs 1,000 cap, after Rs 42.17 spent.
  The prepaid balance left is about Rs 1,159.50. The pre-flight passes.
- **Without ReAct's observation truncation**, the pilot implies Rs 739.74.
- **Cache-adjusted:** about Rs 714.6. Implicit caching served 1.9% of ReAct's prompt
  tokens in the one Flex check, so it is not relied on.
- **Thinner Flex-measured basis:** one holdout unit and one ReAct unit gave Rs 0.546 and
  Rs 1.837 per run at Flex prices, but that ReAct run took 8 steps against a pilot mean of 13.
- **Prices** as of 2026-11-17, USD→INR 94.1. Gemini 3.8 Flash prices **double on
  2027-01-01**, so the run must finish before then.

## 8. Deviations from the original Week 4 plan, decided before this registration

- **H2's pass rule** is the observed rate (≤ 10%). The Clopper–Pearson bound and the
  binomial p are descriptive.
- **H5** is exploratory, and Holm covers H1, H3 and H4. This was corrected at review: the
  earlier "H1–H4" predates H2 becoming a rate rule.
- **ReAct:**
  - 60 s for model-fitting steps;
  - observations truncated to 2,000 characters (never summarised);
  - an append-only prompt (v2), with the same content as v1.
- **Two holdouts** (iid and deploy). H3's primary is the deploy gap.
- **Flex tier** for all LLM systems, with four parallel workers for LLM units and a
  separate single-worker pass for CPU-only units.
- **Unit worst case:** the guard counts a Holdout-family unit as two per-run caps.
- **Failed units** count against the system: as misses on traps, and as non-passes on
  clean tasks (section 3).

## Amendments

### Amendment 1 (2026-11-18): run infrastructure after a Flex outage

**Reason.** The final run started on 2026-11-18 and was stopped by hand (Ctrl+C) after
repeated HTTP 503 responses on the Flex tier.
- **Observed rates.** Between 11:17 and 15:01 UTC, 73 of 91 Gemini attempts failed (80%),
  almost all with HTTP 503. 11 of 29 calls failed all 6 attempts (38%).
- **Why progress stalled.** The runner stopped starting LLM units after the first pause, so
  LLM progress was near zero: holdout 0, react 0 and llm_judge 1 completed.
- **What finished.** The CPU-only pass ran: audits_only 76 completed and 4 failed, flaml 72
  completed and 8 failed.
- **No final results were analysed** before this amendment. No hypothesis statistic was
  computed.

**Changes.**
1. **Flex patience.** A call requested on Flex gets up to **10 attempts** with exponential
   backoff and full jitter (base 5 s, each wait capped at 300 s), honouring `Retry-After`
   (capped at 300 s). This follows Google's Flex guidance. Standard-tier calls keep the
   previous policy (6 attempts, base 2 s, cap 60 s).
2. **One standard fallback per call.** A Flex call that exhausts its 10 attempts is sent
   once more, **unchanged**, on the standard tier with the standard policy: same model
   (`gemini-3.8-flash`, never another), same prompt, same parameters, same seed.
   - **Recorded per call:** the tier requested, the tier served and whether it fell back.
   - **Priced** at the tier that served it.
   - **Reported:** fallback counts per system and per unit in the results and in
     `bench/analyze.py`.
3. **Dropped connections are transient.** A connection closed before any HTTP response
   (`httpx.RemoteProtocolError` and similar) is retried like a 503. Previously it failed the
   unit at once. One llm_judge unit failed this way during the outage
   (`direct_leak@qsar_biodeg`, seed 202).
4. **The runner keeps going.** A unit that still cannot be served pauses only itself; other
   units keep starting.
   - **Circuit breaker:** once 3 units have paused within 10 minutes, new LLM units wait
     10 minutes, then the run continues.
   - **Final sweep:** after the passes, one sweep resumes every paused unit.
   - **Exit:** the runner exits with a summary of completed, stopped, paused and failed
     units. A budget-guard stop still exits 6.
5. **Budget.** The unit worst case stays at the caps (Rs 15; Rs 30 for holdout and
   llm_judge), so the unit-start guard is unchanged.
   - The pre-flight prices the **observed fallback share** at the standard rate, the rest at
     Flex.
   - The share is fallback calls ÷ calls over the completed LLM units of the final suites,
     re-measured at every resume. It is assumed to be 0% until 10 LLM units have completed.
   - If fallbacks push spend to the Rs 1,000 cap, the guard stops the run (exit 6). The run
     then resumes only after a top-up recorded as a further amendment, and no hypothesis is
     judged on a partial run (section 6).
6. **Harness bug, FLAML column names.** LightGBM inside FLAML refuses feature names with
   JSON special characters. magic_telescope's columns end in `:`, so all 8 magic_telescope
   FLAML units failed.
   - The FLAML adapter now renames such columns for FLAML only, identically at fit and
     prediction, and records how many it renamed.
   - The data, the features and FLAML's search are otherwise unchanged.

**Reruns.** Already completed units remain valid and are not rerun. Two groups of failed
units are rerun, because their failures came from the harness bugs fixed above:
- the 8 FLAML magic_telescope units (`LightGBMError: Do not support special JSON
  characters in feature name`);
- the 1 llm_judge unit (`RemoteProtocolError`).

They are selected by error text with `--retry-failed-matching`, and their failed lines stay
in the results files. The 4 audits_only failures are **not** rerun. They are genuine node
timeouts of the deterministic path on house_16h, run alone with no contention: 616–1,023 s
of compute against the pre-registered 600 s limit. They count as failures as
pre-registered.

**What this does not change.** These changes affect only retry timing, the billing tier of
calls that fall back, the runner's scheduling, and two harness defects. They do not change:
- the model, the prompts, the generation parameters or the outputs a call can return;
- the suites, the thresholds, the systems' logic, or the scoring and hypotheses
  (sections 1–5).

The frozen hashes in section 5 are unchanged. `prereg-v1` stays where it is; this amendment
is tagged `prereg-v1-amend1`.

**Resume command:**

```text
python -m bench.run --suite final_v1 --purpose benchmark --workers 4 --retry-failed-matching "RemoteProtocolError|special JSON characters in feature name"
python -m bench.run --suite real_world_v1 --purpose benchmark --workers 4
python -m bench.analyze --results results --out docs/results.md --charts docs/img
```

Superseded in part by Amendment 2: the tiers and the cap changed. The commands are
unchanged.

### Amendment 2 (2026-11-18): service tier per system

**Reason.** Flex was unavailable for most requests. During the stopped run, 73 of 91 Gemini
attempts (80%) failed, almost all with HTTP 503. With Amendment 1's patience, a Flex call
that keeps failing waits up to about 20 minutes of backoff before its standard fallback.
holdout and llm_judge make several short calls per unit, so on Flex most of their units
would wait out the full backoff on most calls. No final results had been analysed when this
amendment was made.

**Change.** The Gemini service tier is now set per system for the remaining
final-benchmark units (`bench/config.toml`, `service_tiers`):

| system | tier | retries |
|---|---|---|
| holdout | **Standard** | 6 attempts, base 2 s, cap 60 s |
| llm_judge | **Standard** | as holdout |
| react | **Flex** | Amendment 1: 10 attempts, base 5 s, cap 300 s, then one Standard fallback per call |

- **Unchanged:** the same model (`gemini-3.8-flash`), prompts, generation parameters and
  seeds, suites, thresholds, scoring and hypotheses. The tier changes latency and price
  only, not model outputs, so completed units stay valid and are not rerun.
- **Per-call records:** every call still records the tier requested and served and is
  priced at the serving tier.
- **Timing:** time per run stays compute time only, with API wait reported separately, so
  the different tiers do not distort timing comparisons.
- **Pre-flight:** each system is priced at its own tier. The observed fallback share
  (Amendment 1) is measured over the Flex systems only (react) and priced at Standard.

**Cost impact** (pilot basis, +20%, 2026-11-18, whole pending benchmark):

| system | Amendment 1 (all Flex) | Amendment 2 |
|---|---|---|
| holdout (116 units) | Rs 75.44 | **Rs 150.88** (Standard) |
| llm_judge (115 pending) | Rs 81.33 | **Rs 162.67** (Standard) |
| react (116 units) | Rs 563.78 | Rs 563.78 (Flex; 0% fallback assumed until 10 react units complete) |
| **total** | Rs 720.55 | **Rs 877.33** |

- **Cap and balance.** The project cap is raised to **Rs 1,150** (`HOLDOUT_PROJECT_CAP_INR=1150`).
  The pre-flight requires Rs 877.33 plus Rs 120 of headroom for 4 units in flight, which
  is Rs 997.33. Available is Rs 1,104.39 under the cap and about Rs 1,150.39 of prepaid
  balance, so it fits.
- **Sensitivity.** Each 10% of ReAct calls served by the Standard fallback adds about
  Rs 56. The pre-flight stops fitting at a fallback share of about 19%. If that happens,
  the unit-start guard ends the run at the cap (exit 6) and a further amendment is needed
  to continue (section 6).

**Frozen values.** This changes `bench/config.toml`. Its sha256 (LF) is now
`1d233cfa20422d6412b0c8c7f3a470dd681ea8073318a86c4079ca450f7d2936`, replacing the hash in
section 5 for the remaining units. Every other hash in section 5 is unchanged. This
amendment is tagged `prereg-v1-amend2`; `prereg-v1` and `prereg-v1-amend1` stay where they
are.

**Resume command** (environment as in section 6, but with `HOLDOUT_PROJECT_CAP_INR=1150`):

```text
python -m bench.run --suite final_v1 --purpose benchmark --workers 4 --retry-failed-matching "RemoteProtocolError|special JSON characters in feature name"
python -m bench.run --suite real_world_v1 --purpose benchmark --workers 4
python -m bench.analyze --results results --out docs/results.md --charts docs/img
```
