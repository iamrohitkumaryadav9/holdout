# Post-hoc exploratory: audits_only v1.1 vs v1.0

> **Not part of the pre-registered evaluation.** The pre-registered results are
> [results.md](results.md) (v1.0, unchanged). Every v1.1 fix was derived from errors found
> on the benchmark itself ([error_analysis.md](error_analysis.md)), so v1.1 numbers on the
> benchmark suites are **optimistic**. Only the offline system (`audits_only`) was
> re-run: no API calls were made, so the agent (`holdout`) and the LLM judge were not
> re-evaluated, and **the Gemini planner path of v1.1 (data findings answered by the model)
> is untested live**. It is covered by fake-LLM tests only.

## What changed

v1.1 adds data findings recorded before planning and kept for the whole run, a no-masking
rule for entity and time findings, stricter A3 (ID-like) and A5 (entity key) candidates,
deduplication of development rows as a plan action, and a goal-level causal check with a
code-enforced decline. Details: [design.md](design.md#v11-post-benchmark-fixes) and
[audits.md](audits.md#v11-changes-post-benchmark-not-pre-registered).

## Method

- Same harness, suites, seeds, scoring rules ([bench/score.py](../bench/score.py)), and
  frozen `thresholds.toml`. v1.1 adds `thresholds_v11.toml` on top.
- **dev_v1 is the primary comparison.** Both versions were run fresh there, on the same
  machine and on the same evening (v1.0 from a `git worktree` of the `v1.0.0` tag). The
  dev suite was used while building v1.0 (one tuning pass), but none of the v1.1 fixes
  were derived from it.
- **final_v1 and real_world_v1 are optimistic: the fixes were derived from errors on this
  suite.** Their v1.0 numbers are the pre-registered `results/` files. They are not
  re-run, because those files must not change.
- v1.1 results: `results_v11/v1_1/<suite>/audits_only.jsonl`; v1.0 on dev:
  `results_v11/v1_0/dev_v1/audits_only.jsonl`.
- Tables: `python -m bench.compare --suite <suite> --before <v1.0 root> --after <v1.1 root>`.
  They list every per-seed change: detection lost or gained, remediation lost or gained,
  and false alarm new or gone on a clean control. A failed unit counts as a miss on a trap
  and as a false alarm on a clean control, as pre-registered.

## dev_v1 (primary)

| | v1.0 | v1.1 |
|---|---|---|
| runs (failed) | 24 (0) | 24 (0) |
| detection recall (task = unit) | 77.8% | 88.9% |
| detected at critical | 55.6% | 55.6% |
| remediated (scorable runs) | 62.5% | 62.5% |
| clean controls with a critical false alarm | 0/3 | 0/3 |
| clean controls with a warning-or-critical false alarm | 1/3 | 1/3 |

| change | task | seed |
|---|---|---|
| detection gained | causal_question@credit_approval (causal_question) | 11 |
| detection gained | causal_question@credit_approval (causal_question) | 12 |

No detection or remediation was lost, and no false alarm is new. The remaining clean warning
is unchanged in both versions: A1 on `A9` of `credit_approval` (file order). The only gain
comes from the causal goal check (fix 6), which v1.0 never ran offline. Nine tasks and
three clean controls are too few to measure a false-alarm change here.

## final_v1 (optimistic: the fixes were derived from errors on this suite)

| | v1.0 | v1.1 |
|---|---|---|
| runs (failed) | 80 (4) | 80 (2) |
| detection recall (task = unit) | 75.0% | 89.3% |
| detected at critical | 57.1% | 57.1% |
| remediated (scorable runs) | 64.0% | 64.0% |
| clean controls with a critical false alarm | 4/12 | 1/12 |
| clean controls with a warning-or-critical false alarm | 5/12 | 1/12 |

| change | task | seed | issues behind it (the version that alarmed) |
|---|---|---|---|
| detection gained | causal_question@puma32h | 101, 202 | goal-level A11 |
| detection gained | causal_question@telco_churn | 101, 202 | goal-level A11 |
| detection gained | causal_question@wilt | 101, 202 | goal-level A11 |
| detection gained | messy_formats@house_16h | 101, 202 | v1.0 failed: the audit node timed out (756 s, 711 s > 600 s) |
| false alarm gone (critical and warning) | causal_question:clean@telco_churn | 101 | A5 critical `number_vmail_messages`; A3 critical `phone_number` |
| false alarm gone (critical and warning) | cross_split_duplicates:clean@cpu_act | 202 | A3 critical `freeswap` |
| false alarm gone (critical and warning) | noisy_proxy:clean@steel_plates_fault | 101, 202 | A3 critical `V3`, `V4`, `V8` |
| false alarm gone (warning) | group_leakage:clean@telco_churn | 101 | A5 warning `number_vmail_messages` |

**Lost detections: none. Lost remediations: none. New false alarms: none.**

- **messy_formats@house_16h is not a better format check.** v1.0's audit timed out because
  A5 cross-validated numeric pseudo-keys. v1.1's A5 finds no entity-like column, so the audit
  finishes (606 s and 582 s for the whole unit) and A8's existing warning is reported. The
  gain depends on the 600 s compute limit.
- **The remaining clean false alarm (1/12) is a failure, not a flag.**
  group_leakage:clean@house_16h times out in both versions: in `compile_and_fit` in v1.0, and
  in `profile_fit` (the same fit, moved earlier) in v1.1. Every completed clean control on
  final_v1 is free of warnings in v1.1.
- **The duplicate traps did not change.** On cross_split_duplicates, 16.5-17.2% of the rows in
  Holdout's internal sealed holdout duplicate a development row (all six runs), so A4 still
  invalidates the holdout (scope `before_split`) and the run ends. Deduplicating development rows (fix 5) cannot help those
  traps.
- Total audits_only compute fell from 5,355 s to 4,808 s. The probe audit is reused for the
  first plan, and A5 tests fewer candidates.

## real_world_v1 (optimistic: the fixes were derived from errors on this suite)

| | v1.0 | v1.1 |
|---|---|---|
| runs (failed) | 36 (0) | 36 (0) |
| detection recall (task = unit) | 57.1% | 57.1% |
| detected at critical | 3.6% | 3.6% |
| remediated (scorable runs) | 17.9% | 3.6% |
| clean controls with a critical false alarm | 1/4 | 1/4 |
| clean controls with a warning-or-critical false alarm | 4/4 | 4/4 |

| change | task | seed |
|---|---|---|
| remediation lost | titanic/original (body_post_outcome) | 101, 202 |
| remediation lost | titanic/anonymised (body_post_outcome) | 101, 202 |

- **The titanic `body` remediation is lost.** `body` (the body-identification number,
  recorded only for victims) is a real post-outcome leak. v1.0 dropped it for the wrong
  reason: A3 called it an identifier, which made A1 critical. v1.1's A3 requires an
  identifier pattern (sequence, code shape, row order or key-like name), which `body` does
  not have. A1 still detects it as a warning, so detection is unchanged, but nothing
  critical drops it. This is a direct cost of the A3 fix.
- Detections and the clean controls' flags are identical. Mushroom `odor` (A1), credit_g (A1
  and A2) and kr_vs_kp (A6) still warn, and abalone A6 is still critical on seed 101. None of
  these was targeted by a fix. The anonymised Diabetes identifier (`col_1`), which v1.1's A3
  could no longer call ID-like, cost no scored detection.

## Trade-offs

- **A3 precision for recall.** The fix removes the `freeswap`, `V3/V4/V8` and
  `phone_number` false alarms but loses the accidental titanic `body` remediation. It also
  stops flagging identifiers that have an anonymised name and no sequence or code shape.
- **A5 coverage.** A numeric entity key with a neutral name is no longer tested.
- **Fix 5 rarely fires on its own.** With a random seal, development data with many
  duplicates nearly always overlaps the sealed holdout by at least 1%, and that invalidates
  the holdout first. Deduplication helps when duplicates across the seal are rare, or when
  the planner chooses it for a warning.
- **The probe costs a fit when the planner diverges.** When the LLM's first plan differs
  from the probe, v1.1 cross-validates and audits twice. This is untested live, and its cost
  in API terms is zero (no extra LLM call), but compute rises.
- **Untested live:** the Gemini planner's `findings_handled` answers, its repair on a
  missing answer, and its response to the masking rule.
