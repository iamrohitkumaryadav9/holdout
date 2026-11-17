# Trap generators

Each trap plants one known data-science failure in a synthetic base dataset and records
its ground truth (`TrapTruth`). Each has a **clean twin**: the same base data and seed with
only the planted mechanism removed, used to measure false alarms.

```python
from bench.traps import build_case, clean_twin

case = build_case("group_leakage", seed=0)
twin = clean_twin(case)
```

Everything is deterministic per seed. The bases (`synthetic.py`) are built so that a clean
base is quiet under every audit: neutral names, no duplicates, no entity structure and no
time order unless asked for.

## Detection rule

A trap counts as **detected** when the audit raises a flag of `expected_category` at a
severity of at least `min_expected_severity`. A clean twin passes when it raises **no
critical flag** (end to end) and, per check, nothing above info for the check it targets.

## Traps and their minimum severity

| trap | planted problem | expected checks | category | min severity | why that minimum |
|---|---|---|---|---|---|
| `direct_leak` | `refund_issued` = target with 3% flips | A1 + A2 | target_leakage | critical | statistics (A1) and the name (A2) agree, which is the corroboration A1 needs |
| `noisy_proxy` | `signal_strength` = target + noise, neutral name | A1 | target_leakage | warning | no name or ID evidence: A1 alone cannot tell a proxy from an honest strong feature, so it is a warning by design |
| `id_with_signal` | `record_number` issued in target order, file shuffled | A3 (+ A1) | id_signal | critical | an identifier must never predict the target |
| `cross_split_duplicates` | 15% of rows duplicated (exact or float noise < 1e-9) | A4 | duplicate_rows | critical | far above the 1% share threshold, and it contaminates the holdout |
| `group_leakage` | entity-level target bias, entity-constant features, neutral `account` key | A5 | group_leakage | critical | the score depends on memorising entities; the holdout shares them |
| `time_order` | drift, date removed (or renamed), rows in time order | A6 | time_leakage | critical | training on the future; the holdout ignores time |
| `imbalance_accuracy` | minority at 4%, metric accuracy | A7 | metric_mismatch | critical | accuracy cannot separate a model from the majority-class guess |
| `messy_formats` | `"Rs 1,200"` strings, case/space variants, -999 sentinels | A8 | format_issue | warning | quality problems, not leaks; A8 never escalates |
| `causal_question` | the goal asks whether X causes Y | A11 | causal_claim | warning | a claim to remove, not a leak; detected on the generated report from Week 3 (string-level in Week 2) |

Clean twins: `direct_leak`, `noisy_proxy`, `cross_split_duplicates`, `messy_formats` and
`causal_question` use the untouched base; `id_with_signal` keeps the ID column but issues
it in random order; `group_leakage` keeps the entities and entity-constant features but no
entity-level effect; `time_order` uses the drift-free base; `imbalance_accuracy` keeps the
accuracy metric on the original, reasonably balanced data.

## Contamination guard

The audit thresholds were tuned on these synthetic bases with seeds 0-4. Every base and
seed used during development or tuning is registered in `bench/dev_usage.toml`. The final
benchmark (Week 4) is built with `bench.suite.build_final_suite`, which **refuses** any entry
whose base or seed is registered there, before loading anything; `tests/unit/test_suite_guard.py`
proves it. The final benchmark therefore uses held-out OpenML bases and seeds never seen during
tuning. Register a new base or seed in `dev_usage.toml` *before* using it in development.
