# Error analysis (post-hoc, exploratory)

> **Post-hoc and exploratory.** This analysis was written after the pre-registered results
> were known, to explain them. It changes no pre-registered result, suite, threshold,
> prompt, scoring rule or hypothesis, and its exploratory statistics do not replace any
> pre-registered test. The confirmatory results are in [results.md](results.md), and the
> registration and its amendments are in [preregistration.md](preregistration.md).

**Data.**
- The final results include the Amendment 4 rerun of 12 ReAct units on 2026-11-19, and
  the latest result per unit is used.
- Each Holdout unit was traced through its run manifests: Gemini's plan in every round
  and every audit flag in every round of the first run and of any follow-up run.
- No API calls were made.

## Summary

1. **No bug in the scoring code.** For every unit traced, the scorer collected every flag
   the system raised.
   - **Two harness bugs did change ReAct's results:** an empty model file failed whole
     units and discarded their answers, and an HTTP 502 was never retried. Amendment 4
     fixed both and reran the 12 affected units, and ReAct now has no failed units.
   - **One smaller harness defect remains:** reruns reused old sandbox folders and read a
     stale empty model file (section 5.3). It can affect at most 2 ReAct remediation
     outcomes and no hypothesis.
2. **Holdout detects less than its own audit layer** (McNemar 1 vs 9, p = 0.02). The
   main reason is that Gemini's plan silently avoids the problem.
   - In 16 of the 20 discordant units the plan dropped the identifier or leak column, or
     chose another metric than the accuracy the goal asked for. The audit then reported
     only info or a warning, below the trap's minimum severity.
   - This is a genuine result under the pre-registered definition (detection means
     reporting the issue). It also means Holdout does not tell the user about problems
     its planner avoided.
3. **Group leakage is masked when the entity key is dropped.**
   - Gemini drops the entity column (`account`, Diabetes's `patient_nbr`) but keeps a
     random split.
   - A5 can no longer see the entity, so the leakage is neither detected nor fixed.
   - This is a genuine design flaw, and it accounts for all of Holdout's remediation gap
     to audits_only on `final_v1` (4 units).
4. **Remediation on the other traps.**
   - **Cross-split duplicates:** no Holdout-family system ever removes them. Holdout flags
     them critical, but no code applies the deduplication fix.
   - **`noisy_proxy` and `time_order`:** misses already stated before the run.
   - **Real-world issues:** Holdout remediates more than audits_only (15 of 28 against 5).
5. **Every critical false alarm is a genuine audit weakness, not a scoring error.**
   Holdout fails H2 with 5 of 16 clean tasks, and the same checks fire for audits_only.
   - A3 calls continuous columns that are almost all unique "identifiers" (false
     positives).
   - A5 treats a count column as an entity key (false positive).
   - A6 flags abalone's non-random row order as time leakage (debatable).
6. **H3 is dominated by units, not by optimism.**
   - 99.999% of the absolute Holdout−ReAct difference comes from regression tasks scored
     in raw RMSE, and one task, noisy_proxy@house_16h, dominates those.
   - Exploratory, scale-free checks disagree. Holdout has the smaller signed gap in 25 of
     36 tasks (sign test p = 0.029). The absolute gaps (20 vs 16, p = 0.62) and the
     dummy-normalised gaps (median difference −0.006) show no difference.
7. **ReAct's 60 s fitting timeouts don't explain its misses.**
   - 53 of its 80 `final_v1` units hit at least one fitting timeout.
   - They are spread evenly: 17 detected units and 14 missed units had one.
   - The report's warning that timeouts may have handicapped ReAct still applies.

The confirmatory decisions after the rerun:

| | decision | key numbers |
|---|---|---|
| H1 (Holdout > ReAct, detection) | not supported | discordant tasks: Holdout only 5, ReAct only 8, p = 0.58 |
| H2 (critical false alarms ≤ 10%) | fail | 5 of 16 clean tasks |
| H3 (Holdout's optimism gap smaller) | not supported | p = 0.73, Holm 1.0 |
| H4 (audits beat the LLM judge) | not supported | Holdout only 10, llm_judge only 6, p = 0.45 |
| H5 (exploratory, names) | inconclusive | 7 issues, −0.071 [−0.214, 0.000] |

## 1. Detection: Holdout against its own audit layer

**The discordant tasks** (`final_v1`, detected in both seeds; McNemar 1 vs 9):

| audits_only only (9) | holdout only (1) |
|---|---|
| group_leakage on house_16h, qsar_biodeg, telco_churn | messy_formats@house_16h, because audits_only's two units there failed with node timeouts |
| id_with_signal on credit_card_default, puma32h, wilt | |
| imbalance_accuracy on magic_telescope, pol, steel_plates_fault | |

In `real_world_v1` the only discordant issue is Titanic's `boat` (original and anonymised).
causal_question, Diabetes `repeated_patients` and Bank `duration` are not discordant with
audits_only. audits_only never detects causal_question, and on the other two both systems
agree in every seed.

**Every unit where audits_only detected the issue and Holdout did not** (20 units), with
the classes:
- **(a)** the plan avoided the problem;
- **(b)** flagged below the minimum severity;
- **(c)** flagged but not collected by the scorer;
- **(d)** genuinely missed;
- **(e)** other.

| trap / issue | units | class | what happened |
|---|---|---|---|
| id_with_signal | 6 | (a) | Gemini's first plan drops `record_number` ("identifier column with unique sequential ids"). A3 then reports it at **warning** because the plan already drops it; the trap requires critical. audits_only keeps the column and gets A3 critical. |
| imbalance_accuracy | 6 | (a) | The goal asks for accuracy. Gemini plans `pr_auc` or `balanced_accuracy` (pol: also drops 22 features), so A7 has no metric mismatch to flag. audits_only receives the stated metric as an override and gets A7 critical. |
| Titanic `boat_post_outcome` | 4 | (a) | Gemini drops `boat` and `body` with the reason "post-outcome survival indicator causing leakage" (anonymised: "boat column is post-outcome…", read from the values). A1 and A2 then report **info**; the issue requires warning. |
| group_leakage | 3 | (d) | Gemini drops `account` but keeps a random or stratified split. A5 only considers columns the plan uses, so it raises nothing. |
| group_leakage | 1 | (b) | Same plan; A5 reports `account` at **warning** (house_16h s101). |
| any | 0 | (c) | Every relevant flag was collected; no flag at or above the minimum was missed. |

**Interpretation.**
- **(a) is a measurement consequence, not a scoring error.** The planner's prior knowledge
  avoids the problem, but Holdout records it only as a lower-severity flag or a drop
  reason. A user is never told "this column is a leak" at the severity the trap requires.
- **(d) and (b) are a genuine weakness.** Dropping an entity key does not remove group
  leakage: rows of the same account still sit on both sides of the split. Dropping the key
  only hides it from A5. audits_only keeps the column, sees it, and starts the corrective
  follow-up run with a group split.

## 2. Remediation

| suite | holdout | audits_only | llm_judge | react |
|---|---|---|---|---|
| `final_v1` (50 scorable runs) | 28 (56%) | 32 (64%) | 33 (66%) | 35 (70%) |
| `real_world_v1` (28) | 15 | 5 | 22 | 17 |

**Holdout's 22 unremediated `final_v1` units, by cause:**

| trap | units | class | cause | remediated by |
|---|---|---|---|---|
| cross_split_duplicates | 6 | (e) | A4 flags them **critical**, but no code applies its deduplication fix (`Deduplicate` is a suggested fix that nothing acts on). Holdout reports duplicates and never removes rows. | react only (audits_only also 0 of 6) |
| noisy_proxy | 6 | (b) | A1 **warning** only (neutral name), so the column is not dropped automatically. This miss was stated before the run. | llm_judge (4), react (6) |
| time_order | 6 | (d) | A6's row-order gate is blind to a smooth trend. This miss was stated before the run. | react (1, after the rerun) |
| group_leakage | 4 | (d) | Entity key dropped, random split kept (section 1); no follow-up run. | audits_only, llm_judge, react |

- **The gap to audits_only is exactly the 4 group_leakage units.** On the other three
  traps audits_only fails too.
- **No scoring bug:** in every case the final pipeline really lacks the fix (no
  deduplication, the proxy kept, no group or time split).

**Holdout's 13 unremediated real-world units:**
- **Diabetes (4):** Gemini drops `patient_nbr` (the anonymised `col_2`), so A5 is blind.
  In the original variant the split groups by `discharge_disposition_id` instead of the
  patient.
- **Electricity (4) and Covertype (4):** A6 reports only a **warning** (its sort-confound
  path), so no time split is applied.
- **Bank Marketing, anonymised s202 (1):** A2 **warning** on `col_11`, so the column is
  kept.

## 3. Critical false alarms on clean controls

H2 counts a clean task as a non-pass if any seed or variant raised a critical issue or
failed. Holdout: 5 of 16; audits_only: 4 of 12 planted tasks.

| clean task | seeds | check, columns | holdout | audits_only | evidence | real problem or false positive |
|---|---|---|---|---|---|---|
| causal_question:clean@telco_churn | 101 | A5 `number_vmail_messages` | yes | yes | 48 values, 67 rows per value, constant-feature share 0.05; GroupKFold drops the CV AUC from 0.927 to 0.836 (21% of the lift), noise 2% | **False positive.** A count of voicemail messages, mostly 0, is not an entity. Grouping by it shifts the distribution between folds, which A5 reads as memorisation. |
| group_leakage:clean@telco_churn | 101 | A5 `number_vmail_messages` | yes | no | as above, 23% of the lift | **False positive**, as above. |
| cross_split_duplicates:clean@cpu_act | 202 | A3 `freeswap` | yes | yes | signal "near_unique" (unique ratio 0.951), row-order ρ = 0.006, alone 56% of the way to perfect | **False positive.** A continuous system measurement is naturally almost all unique. The fix drops a genuinely predictive feature. |
| noisy_proxy:clean@steel_plates_fault | 101, 202 | A3 `V3`, `V4`, `V8` | yes | yes | "near_unique" (unique ratio 0.99–1.0), row-order ρ ≈ 0.04 | **False positive.** Continuous plate measurements (V3/V4 are probably Y_Minimum/Y_Maximum). |
| causal_question:clean@telco_churn (follow-up run) | 101 | A3 `phone_number` | no | yes | (follow-up round) | A real identifier. Whether it carries signal was not checked. |
| group_leakage:clean@house_16h | 101, 202 | (failed unit) | no | yes | node timeout, 613–631 s of compute against the 600 s limit | A genuine timeout, counted against audits_only as pre-registered. |
| abalone, original and anonymised (real) | 101 | A6 row order | yes | yes | target lag-1 autocorrelation 0.48 (z = 25); training on earlier rows drops the CV RMSE score from 2.26 to 2.45 (21% of the lift) | **Debatable.** The file's row order really is not exchangeable (batches), but it is not time. For an iid task it is not a leak, so the ground truth counts it as a false alarm. |

**Two defects in the flag text.**
- A3's message says "the data's ordering encodes the outcome" even when the evidence shows
  no row-order correlation (ρ ≈ 0); the flag was raised by the near-unique rule alone.
- A5's group candidates include count-valued numeric columns.

These are audit weaknesses for a later version. The thresholds stay frozen for this
benchmark.

## 4. H3: optimism-gap scaling

**The pre-registered H3** compares the per-task mean of Holdout's and ReAct's deploy
optimism gaps in each system's own metric. The result: +1,413.39 [−246.95, +4,501.65],
sign-flip p = 0.73, not supported.

**Is the mean dominated by raw regression units? Yes.**
- **Regression:** 12 tasks, mean difference **+4,240.16** (RMSE in the target's units).
- **Classification:** 24 tasks, mean difference **+0.0025**.
- **Share:** regression contributes **99.999%** of the absolute Holdout−ReAct difference.
- **One task dominates:** noisy_proxy@house_16h, gap +53,735.14 for Holdout against +57.40
  for ReAct, in house prices.
  - Holdout kept the proxy (A1 warning, section 2), so its validation RMSE is far better
    than its RMSE on the deploy holdout, where the proxy is permuted.

**Exploratory scale-free versions** (they do not replace H3):

| version | Holdout smaller | ReAct smaller | p |
|---|---|---|---|
| (i) sign test, signed gap (less optimistic) | 25 | 11 | 0.029 (two-sided binomial) |
| (i′) sign test, absolute gap (closer to the holdout either way) | 20 | 16 | 0.62 |
| (ii) gap ÷ the model's distance from the dummy on the deploy holdout, same metric (24 tasks where both systems' own metric is the benchmark metric) | median difference −0.006, mean +0.22 | | sign-flip p = 0.44 |

- **What the signed test measures.** Holdout is less often *optimistic* than ReAct.
- **What the other two show.** Neither the size of the miss nor a scale-free gap differs.
- **Caveat:** (ii) excludes the 12 tasks where a system validated in a different metric
  from the benchmark's (`pr_auc`, `balanced_accuracy`, `accuracy`, `r2`); there the dummy's
  score in that metric was not recorded.

**Per-task deploy gaps**
- Each value is seeds averaged, positive means optimistic, and RMSE gaps are
  sign-adjusted.
- "smaller signed gap" is the system whose gap is smaller.
- "normalised" is the gap divided by the model-to-dummy distance; it is blank when not
  defined.

| task | benchmark metric | holdout metric | react metric | holdout gap | react gap | smaller signed gap | holdout normalised | react normalised |
|---|---|---|---|---|---|---|---|---|
| causal_question@puma32h | rmse | rmse | rmse | +0.0000 | +0.0001 | holdout | +0.001 | +0.004 |
| cross_split_duplicates:clean@cpu_act | rmse | rmse | rmse | -0.3175 | -0.1924 | holdout | -0.020 | -0.012 |
| cross_split_duplicates@cpu_act | rmse | rmse | rmse | -0.1935 | -0.2077 | react | -0.011 | -0.012 |
| direct_leak@cpu_act | rmse | rmse | rmse | -0.3175 | -0.0873 | holdout | -0.020 | -0.005 |
| group_leakage:clean@house_16h | rmse | rmse | rmse | -299.4407 | +216.7438 | holdout | -0.013 | +0.011 |
| group_leakage@house_16h | rmse | rmse | rmse | -562.6232 | -1589.0610 | react | -0.072 | -0.172 |
| id_with_signal@puma32h | rmse | rmse | rmse | -0.0001 | -0.0003 | react | -0.005 | -0.013 |
| messy_formats@cpu_act | rmse | rmse | rmse | -0.3169 | -0.1161 | holdout | -0.020 | -0.008 |
| messy_formats@house_16h | rmse | rmse | rmse | -101.0896 | +3204.4221 | holdout | -0.006 | +0.153 |
| noisy_proxy@house_16h | rmse | rmse | rmse | +53735.1385 | +57.4018 | react | +4.264 | +0.002 |
| time_order:clean@puma32h | rmse | rmse | rmse | +0.0000 | +0.0000 | react | -0.001 | -0.002 |
| time_order@puma32h | rmse | rmse | rmse | +0.0057 | +0.0058 | holdout | +0.319 | +0.326 |
| causal_question:clean@telco_churn | roc_auc | roc_auc | roc_auc | -0.0370 | -0.0160 | holdout | -0.089 | -0.037 |
| causal_question@wilt | roc_auc | pr_auc | roc_auc | +0.0002 | +0.0088 | holdout | - | +0.018 |
| cross_split_duplicates@magic_telescope | roc_auc | roc_auc | accuracy,roc_auc | +0.0072 | -0.0029 | react | +0.017 | - |
| cross_split_duplicates@steel_plates_fault | roc_auc | roc_auc | accuracy,roc_auc | +0.0000 | +0.0097 | holdout | +0.000 | - |
| direct_leak:clean@qsar_biodeg | roc_auc | roc_auc | accuracy | +0.0242 | +0.0447 | holdout | +0.061 | - |
| direct_leak@pol | roc_auc | roc_auc | roc_auc | -0.0002 | +0.0005 | holdout | +0.000 | +0.001 |
| direct_leak@qsar_biodeg | roc_auc | roc_auc | roc_auc | +0.0242 | +0.0448 | holdout | +0.061 | +0.113 |
| direct_leak@telco_churn | roc_auc | roc_auc | roc_auc | -0.0460 | -0.0104 | holdout | -0.109 | -0.024 |
| group_leakage@qsar_biodeg | roc_auc | roc_auc | roc_auc | +0.1122 | +0.2239 | holdout | +4.657 | +2.509 |
| id_with_signal:clean@credit_card_default | roc_auc | roc_auc | roc_auc | +0.0038 | +0.0055 | holdout | +0.014 | +0.020 |
| id_with_signal@credit_card_default | roc_auc | roc_auc | roc_auc | +0.0013 | +0.0040 | holdout | +0.005 | +0.014 |
| id_with_signal@wilt | roc_auc | pr_auc,roc_auc | roc_auc | +0.0082 | +0.0052 | react | - | +0.011 |
| imbalance_accuracy:clean@magic_telescope | roc_auc | accuracy | accuracy | +0.0004 | +0.0064 | holdout | - | - |
| imbalance_accuracy:clean@pol | roc_auc | accuracy | accuracy | -0.0020 | -0.0002 | holdout | - | - |
| imbalance_accuracy@magic_telescope | roc_auc | pr_auc | accuracy | -0.0216 | +0.0007 | holdout | - | - |
| imbalance_accuracy@pol | roc_auc | balanced_accuracy | accuracy | -0.0231 | -0.0016 | holdout | - | - |
| imbalance_accuracy@steel_plates_fault | roc_auc | pr_auc | accuracy | -0.0314 | +0.0004 | holdout | - | - |
| messy_formats@credit_card_default | roc_auc | roc_auc | roc_auc | +0.0038 | +0.0041 | holdout | +0.014 | +0.015 |
| noisy_proxy:clean@steel_plates_fault | roc_auc | roc_auc | roc_auc | +0.0000 | -0.0092 | react | +0.000 | -0.021 |
| noisy_proxy@magic_telescope | roc_auc | roc_auc | accuracy,roc_auc | +0.3129 | +0.0030 | react | +1.762 | - |
| noisy_proxy@steel_plates_fault | roc_auc | roc_auc | roc_auc | +0.0435 | -0.0085 | react | +0.105 | -0.020 |
| time_order:clean@credit_card_default | roc_auc | roc_auc | roc_auc | +0.0009 | +0.0060 | holdout | +0.003 | +0.021 |
| time_order:clean@wilt | roc_auc | pr_auc | roc_auc | +0.0151 | +0.0069 | react | - | +0.014 |
| time_order@wilt | roc_auc | roc_auc | roc_auc | +0.1125 | +0.1242 | holdout | +1.845 | +2.747 |

## 5. ReAct: failures, stops and timeouts

### 5.1 Failed units (before and after Amendment 4)

Before the rerun ReAct had 12 failed units:
- **11 `EOFError: Ran out of input`** (an empty `final_model.pkl`; the bug fixed by
  Amendment 4):
  - `final_v1`:
    - imbalance_accuracy@magic_telescope s101;
    - messy_formats@cpu_act s101 and s202;
    - messy_formats@house_16h s101;
    - time_order@credit_card_default s202.
  - `real_world_v1`:
    - covertype original s101 and s202;
    - diabetes anonymised s101 and s202;
    - electricity anonymised s202;
    - titanic anonymised s101.
- **1 HTTP 502** (id_with_signal@wilt s101), not retried until Amendment 4.

After the rerun, **none failed**. 9 completed and 3 stopped:
- diabetes anonymised s101 and s202: per-run cap;
- covertype original s202: no final answer within 15 steps.

### 5.2 Stopped units (final)

| suite | unit | reason |
|---|---|---|
| `final_v1` | direct_leak@pol s202 | no final answer within 15 steps |
| `final_v1` | group_leakage:clean@telco_churn s202 | no final answer within 15 steps |
| `final_v1` | group_leakage@telco_churn s101 | no final answer within 15 steps |
| `real_world_v1` | abalone original s202 | per-run cap (Rs 15) at step 14 |
| `real_world_v1` | bank_marketing original s202 | no final answer within 15 steps |
| `real_world_v1` | covertype original s202 | no final answer within 15 steps (rerun) |
| `real_world_v1` | diabetes original s101 | per-run cap at step 15 |
| `real_world_v1` | diabetes anonymised s101 | per-run cap at step 14 (rerun) |
| `real_world_v1` | diabetes anonymised s202 | per-run cap at step 15 (rerun) |

### 5.3 A residual harness defect: stale model files in reused sandbox folders

- **What happened.** The rerun reused each unit's sandbox folder, and `prepare_workdir`
  does not clear an old model file. In 6 rerun units ReAct did not submit a valid model,
  and the loader found the **first run's empty file**: dated 12:29–14:51 on 2026-11-19,
  against rerun steps at 18:57–19:21; 5 of the 6 were inspected.
- **What was recorded.** "no valid model submitted: the model file is unreadable (0 bytes)"
  instead of "no model was submitted":
  - messy_formats@house_16h s101;
  - messy_formats@cpu_act s202;
  - titanic anonymised s101;
  - electricity anonymised s202;
  - diabetes anonymised s101 and s202.
- **Detection:** unaffected. ReAct's issues are kept either way.
- **Remediation:** an unreadable file withholds the pipeline, while a missing file keeps it
  (Amendment 4).
  - messy_formats is not scored for remediation.
  - The diabetes units stopped without a final answer.
- **So at most 2 outcomes could change:** ReAct's remediation of titanic anonymised s101
  and electricity anonymised s202. They may be understated, and they cannot be re-scored
  because their final pipeline was not saved.
- **No hypothesis uses remediation.** Not fixed here, since this analysis changes nothing.

### 5.4 Fitting-step timeouts and misses

| suite | units with a 60 s fitting timeout | trap misses | misses with a fitting timeout | detections with a fitting timeout |
|---|---|---|---|---|
| `final_v1` | 53 of 80 | 22 | 14 | 17 |
| `real_world_v1` | 10 of 36 | 8 | 1 | 4 |

Timeouts are about as common among detected units as among misses. There is no evidence
that they cause ReAct's misses. They may still have cost accuracy, as `results.md` warns.

## 6. Bugs and defects found, by kind

| finding | kind | status | effect on confirmatory results |
|---|---|---|---|
| ReAct's empty model file failed whole units | harness bug | fixed and rerun (Amendment 4) | H1 changed from 7 vs 7 to 5 vs 8; still not supported |
| HTTP 502 not retried | harness bug | fixed and rerun (Amendment 4) | 1 unit |
| Stale model file in reused sandbox folders | harness defect | open | at most 2 ReAct remediation outcomes; no hypothesis |
| Deduplication fix never applied | system design (Holdout) | open | remediation of cross_split_duplicates |
| Entity key dropped by the planner hides group leakage | system design (Holdout) | open | detection and remediation of group_leakage and Diabetes |
| The planner's avoidance is not reported as a finding | system design (Holdout) | open | detection of id_with_signal, imbalance_accuracy, Titanic |
| A3 near-unique rule on continuous columns | audit heuristic | open (thresholds frozen) | H2 false alarms |
| A5 candidates include count columns | audit heuristic | open | H2 false alarms |
| A6 flags non-temporal row order | audit heuristic (debatable) | open | H2 (abalone) |
| A3 message claims row order encodes the outcome | flag text | open | none |
| H3 in raw target units | analysis design | as pre-registered | H3 dominated by one regression task |
| Scoring code | n/a | no bug found | n/a |
