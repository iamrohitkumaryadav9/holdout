# Pre-registration: Holdout benchmark v1

**Status: DRAFT for review. Not yet tagged.** This document becomes binding when it is
committed and tagged `prereg-v1`. After the tag, nothing that affects results may change
without a dated amendment at the end of this file that says what changed, why, and whether
any final result had been seen.

- Written 2026-11-18, before any system has run on `final_v1` or `real_world_v1`.
- Definitions of systems, suites, holdouts and scoring: [benchmark.md](benchmark.md).
- Design decisions and the one-time tuning: [design.md](design.md).

## 1. Hypotheses

The unit of every test and CI is the **task**. Seeds are averaged within a task, or, for
McNemar, a task counts as detected when it is detected in both seeds. Anonymised variants
are never pooled with originals.

| | hypothesis | data | primary metric | test | decision rule |
|---|---|---|---|---|---|
| **H1** | Holdout's detection recall on planted traps exceeds ReAct's. | `final_v1`, 28 trap tasks | detection in both seeds, per task | exact McNemar (binomial on discordant tasks), two-sided | **supported** if Holdout has more discordant wins and the Holm-adjusted p < 0.05 |
| **H2** | Holdout's critical false-alarm rate on clean controls is at most 10%. | 12 clean twins in `final_v1` + 4 real-world clean controls = **16 clean tasks** | a task alarms if any seed or variant raised a critical issue | observed rate; exact one-sided binomial p for H0: rate ≥ 10% | **pass** if the observed rate ≤ 10%, i.e. at most 1 of 16 |
| **H3** | Holdout's optimism gap is smaller than ReAct's. | `final_v1`, every task where both systems have a gap | **primary: `optimism_gap_deploy`**; secondary: `optimism_gap_iid` | paired sign-flip permutation test on the per-task difference (Holdout − ReAct), two-sided | **supported** if the mean difference is negative and the Holm-adjusted p < 0.05 |
| **H4** | Deterministic audits beat the LLM judge on detection. | `final_v1`, 28 trap tasks | as H1, `holdout` vs `llm_judge` | as H1 | as H1 |
| **H5** *(exploratory)* | Real-world detection does not depend on column names. | `real_world_v1` issues, original vs anonymised | per-issue detection rate (seeds averaged) | paired bootstrap 95% CI of original − anonymised | "no material dependence" if the CI lies within ±0.20; outside the Holm family |

**H4 is the ablation.** `llm_judge` is Holdout with only its audit layer replaced by a
Gemini verifier. Same planner, compiler, fix loop and unseal gate. The comparison
isolates the audit layer.

**Multiplicity.**
- **Family.** Holm–Bonferroni over **H1–H4** at α = 0.05.
- **H2's p.** H2 is decided by its observed-rate rule. Its exact binomial p still enters
  the Holm family. That makes the correction for H1, H3 and H4 more conservative, never
  less.
- **H5** is exploratory.

**H2's bound, stated as decided.** With 16 clean tasks, the one-sided 95% Clopper–Pearson
upper bound cannot fall below about 17% even with zero false alarms (and H2's binomial p
cannot fall below 0.9¹⁶ = 0.185). The bound is therefore **reported next to the rate but
not used as the pass condition**.

**Estimates reported with every hypothesis.**
- **H1, H4:** the mean per-task difference in detection rate, with a paired bootstrap 95% CI
  (10,000 resamples, seed 0).
- **H2:** the rate and its Clopper–Pearson bound.
- **H3:** the mean per-task gap difference, with a paired bootstrap CI.
- **H5:** the mean difference with its CI.

**Implementation.** The analysis code is fixed at this tag: `bench/hypotheses.py` (H1–H5,
Holm) and `bench/score.py` (detection, remediation, false alarms, gaps, McNemar,
Clopper–Pearson, bootstrap). It is tested on hand-built results in
`tests/unit/test_bench_hypotheses.py` and `tests/unit/test_bench_scoring.py`.
`python -m bench.analyze` prints the H1–H5 table first.

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
- **Failed units count as not detected and not remediated.** A unit fails when a system
  crashes or a node exceeds its compute limit. Without this rule, a crashing system would
  escape its misses. Failed units are excluded from false-alarm counts: a crash cannot
  raise an alarm. The number of failed units per system is reported.
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

- **H2's pass rule** is the observed rate (≤ 10%); the Clopper–Pearson bound is reported.
- **H5** is exploratory, and Holm covers H1–H4.
- **ReAct:**
  - 60 s for model-fitting steps;
  - observations truncated to 2,000 characters (never summarised);
  - an append-only prompt (v2), with the same content as v1.
- **Two holdouts** (iid and deploy). H3's primary is the deploy gap.
- **Flex tier** for all LLM systems, with four parallel workers for LLM units and a
  separate single-worker pass for CPU-only units.
- **Unit worst case:** the guard counts a Holdout-family unit as two per-run caps.
- **Failed units** count as not detected (section 3).

## Amendments

None.
