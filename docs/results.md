# Benchmark results

## Pre-registered hypotheses

| | hypothesis | n (tasks) | estimate [95% CI] | p | p (Holm: H1, H3, H4) | decision |
|---|---|---|---|---|---|---|
| H1 | Holdout's detection recall on planted traps exceeds ReAct's | 28 | -0.0893 [-0.3214, +0.1429] | 0.5811 | 1.0000 | **not supported** |
| H2 | Holdout's critical false-alarm rate on clean tasks is at most 10% | 16 | +0.3125 [+0.0000, +0.5483] | - | - | **fail** |
| H3 | Holdout's optimism gap (deploy holdout) is smaller than ReAct's | 36 | +1413.3888 [-246.9499, +4501.6458] | 0.7278 | 1.0000 | **not supported** |
| H4 | deterministic audits (holdout) beat the LLM judge (llm_judge) on detection | 28 | +0.1607 [-0.0893, +0.4107] | 0.4545 | 1.0000 | **not supported** |
| H5 | (exploratory) Holdout's real-world detection does not depend on column names | 7 | -0.0714 [-0.2143, +0.0000] | - | - | **inconclusive or dependent** |

- H1: holdout only 5, react only 8, both 8, neither 7 (detected in both seeds)
- H2: 5 of 16 clean tasks did not pass (critical issue or failed unit; 0 with a failed unit); descriptive only: one-sided 95% Clopper-Pearson upper bound 54.8%, exact binomial p (H0: rate >= 10%) 0.9967
- H3: mean per-task difference holdout - react (negative favours Holdout); secondary, iid holdout: -109.5527 [-307.5266, +2.0597], sign-flip p 0.1277, n 36
- H4: holdout only 10, llm_judge only 6, both 3, neither 9 (detected in both seeds)
- H5: original - anonymised detection rate; margin +/-0.20; outside the Holm family

## final_v1

> **Warning: ReAct sandbox steps timed out** in 53 of 80 ReAct runs (86 model-fitting steps at the 60 s limit, 0 other steps at 20 s). Timeouts may have handicapped ReAct: its results here may understate what it could do with more time per step.

Runs: completed 393, failed 4, stopped 3

### Unit failures

| system | failed units | failure rate |
|---|---|---|
| holdout | 0 of 80 | 0.0% |
| audits_only | 4 of 80 | **5.0%** |
| llm_judge | 0 of 80 | 0.0% |
| react | 0 of 80 | 0.0% |
| flaml | 0 of 80 | 0.0% |

A failed unit counts as not detected and not remediated, and as a non-pass (a false alarm) on a clean task.

### Standard-tier fallbacks (Flex unavailable after 10 attempts)

| system | fallback calls | of paid calls | units with a fallback |
|---|---|---|---|
| holdout | 0 | 0.0% | 0 of 80 |
| llm_judge | 0 | 0.0% | 0 of 80 |
| react | 48 | 4.5% | 14 of 80 |

Per unit: react direct_leak@cpu_act s202: 2; react direct_leak@pol s101: 1; react direct_leak@pol s202: 2; react direct_leak@qsar_biodeg s101: 10; react direct_leak@qsar_biodeg s202: 7; react direct_leak@telco_churn s101: 7; react direct_leak@telco_churn s202: 7; react id_with_signal@credit_card_default s101: 1; react noisy_proxy@house_16h s101: 2; react noisy_proxy@house_16h s202: 2; react noisy_proxy@magic_telescope s101: 1; react noisy_proxy@magic_telescope s202: 3; react noisy_proxy@steel_plates_fault s101: 2; react noisy_proxy@steel_plates_fault s202: 1

Fallback calls are the same request on gemini-3.8-flash, priced at the standard rate.

### Detection by trap type (share of task-seed runs)

| trap | holdout | audits_only | llm_judge | react | flaml |
|---|---|---|---|---|---|
| causal_question | 16.7% | 0.0% | 100.0% | 100.0% | 0.0% |
| cross_split_duplicates | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| direct_leak | 100.0% | 100.0% | 37.5% | 62.5% | 0.0% |
| group_leakage | 33.3% | 100.0% | 100.0% | 83.3% | 0.0% |
| id_with_signal | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |
| imbalance_accuracy | 0.0% | 100.0% | 16.7% | 0.0% | 0.0% |
| messy_formats | 100.0% | 66.7% | 33.3% | 100.0% | 0.0% |
| noisy_proxy | 100.0% | 100.0% | 33.3% | 100.0% | 0.0% |
| time_order (semi-synthetic) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Detection recall (task = unit, seeds averaged; 95% bootstrap CI)

| system | recall | both seeds | either seed | per seed |
|---|---|---|---|---|
| holdout | 51.8% [33.9%, 69.6%] | 46.4% | 57.1% | 101: 57.1%, 202: 46.4% |
| audits_only | 75.0% [57.1%, 89.3%] | 75.0% | 75.0% | 101: 75.0%, 202: 75.0% |
| llm_judge | 35.7% [19.6%, 53.6%] | 32.1% | 39.3% | 101: 32.1%, 202: 39.3% |
| react | 60.7% [42.9%, 78.6%] | 57.1% | 64.3% | 101: 60.7%, 202: 60.7% |
| flaml | 0.0% [0.0%, 0.0%] | 0.0% | 0.0% | 101: 0.0%, 202: 0.0% |

### Remediation (the final pipeline fixed it; scorable traps only)

| system | remediated | scorable runs |
|---|---|---|
| holdout | 56.0% | 50 |
| audits_only | 64.0% | 50 |
| llm_judge | 66.0% | 50 |
| react | 70.0% | 50 |
| flaml | 12.0% | 50 |

### False alarms on clean controls (task = unit: any seed or variant alarming or failing counts)

| system | critical | one-sided 95% upper bound | warning or above | any issue (info too) |
|---|---|---|---|---|
| holdout | 4/12 = 33.3% | 60.9% | 33.3% | 100.0% |
| audits_only | 4/12 = 33.3% | 60.9% | 41.7% | 100.0% |
| llm_judge | 3/12 = 25.0% | 52.7% | 25.0% | 25.0% |
| react | 5/12 = 41.7% | 68.5% | 58.3% | 100.0% |
| flaml | 0/12 = 0.0% | 22.1% | 0.0% | 0.0% |

### Pairwise detection: exact McNemar per task (detected in both seeds; either seed)

| A vs B | A only | B only | p (both) | p (either) |
|---|---|---|---|---|
| holdout vs audits_only | 1 | 9 | 0.0215 | 0.1797 |
| holdout vs llm_judge | 10 | 6 | 0.4545 | 0.2668 |
| holdout vs react | 5 | 8 | 0.5811 | 0.7539 |
| holdout vs flaml | 13 | 0 | 0.0002 | 0.0000 |
| audits_only vs llm_judge | 16 | 4 | 0.0118 | 0.0309 |
| audits_only vs react | 9 | 4 | 0.2668 | 0.5488 |
| audits_only vs flaml | 21 | 0 | 0.0000 | 0.0000 |
| llm_judge vs react | 1 | 8 | 0.0391 | 0.0391 |
| llm_judge vs flaml | 9 | 0 | 0.0039 | 0.0010 |
| react vs flaml | 16 | 0 | 0.0000 | 0.0000 |

### Scores and optimism on both holdouts (completed runs; task = unit; mean [95% CI])

| system | optimism gap deploy | optimism gap iid | holdout_deploy vs dummy | holdout_iid vs dummy | holdout_deploy - FLAML |
|---|---|---|---|---|---|
| holdout | +1319.2844 [-50.5939, +4025.0602] | -87.4383 [-242.2002, -2.5784] | 0.677 | 0.749 | +10.2189 |
| audits_only | +1359.3524 [-122.4674, +4200.5486] | -1383.2153 [-4145.0549, -0.0324] | 0.690 | 0.803 | -46.7879 |
| llm_judge | -53.9657 [-161.6132, +5.7115] | -113.8181 [-341.1310, +5.6996] | 0.702 | 0.735 | -842.9783 |
| react | +52.4822 [-118.7993, +274.6451] | +12.4040 [-243.4141, +273.0520] | 0.711 | 0.733 | -912.2197 |
| flaml | +1362.6970 [-24.7890, +4116.5903] | -85.0428 [-254.2565, +7.2546] | 0.605 | 0.794 | - |

Optimism gap: the system's own validation claim minus its holdout score in the same metric (positive = optimistic). holdout_deploy is deployment-faithful (leak columns unavailable, unseen entities, the future); holdout_iid keeps leaks intact. Real cases scored on detection and remediation only are excluded.

### Cost and answer quality (completed runs)

| system | cost/run (list) | cost/run (cache-adjusted) | compute time/run | API wait/run | calls | retries | first-attempt valid | repairs/run | substitutions/run |
|---|---|---|---|---|---|---|---|---|---|
| holdout | Rs 1.22 | Rs 1.21 | 56 s | 26 s | 4.1 | 0 | 94.9% | 0.14 | 0.03 |
| audits_only | Rs 0.00 | Rs 0.00 | 28 s | 0 s | 0.0 | 0 | - | 0.00 | 0.00 |
| llm_judge | Rs 1.45 | Rs 1.21 | 31 s | 39 s | 5.2 | 1 | 99.4% | 0.03 | 0.00 |
| react | Rs 8.84 | Rs 8.11 | 203 s | 2067 s | 13.2 | 800 | 99.7% | 0.04 | 0.00 |
| flaml | Rs 0.00 | Rs 0.00 | 63 s | 0 s | 0.0 | 0 | - | 0.00 | 0.00 |

### Timeouts

| system | units | node:audit | node:compile_and_fit | react_fit_step |
|---|---|---|---|---|
| holdout | 0 of 80 | 0 | 0 | 0 |
| audits_only | 4 of 80 | 2 | 2 | 0 |
| llm_judge | 0 of 80 | 0 | 0 | 0 |
| react | 53 of 80 | 0 | 0 | 86 |
| flaml | 0 of 80 | 0 | 0 | 0 |

Cells: total timeouts; units: runs with at least one timeout.

### A11 flags on generated reports of tasks whose goal is not causal

An upper bound on A11's false-positive rate: whether a flagged sentence really is a causal claim needs a human label (sentences listed in the appendix).

| system | reports | flagged | rate |
|---|---|---|---|
| holdout | 72 | 0 | 0.0% |
| audits_only | 68 | 0 | 0.0% |
| llm_judge | 72 | 4 | 5.6% |
| react | 69 | 10 | 14.5% |

## real_world_v1

> **Warning: ReAct sandbox steps timed out** in 11 of 36 ReAct runs (10 model-fitting steps at the 60 s limit, 2 other steps at 20 s). Timeouts may have handicapped ReAct: its results here may understate what it could do with more time per step.

Runs: completed 174, stopped 6

### Unit failures

| system | failed units | failure rate |
|---|---|---|
| holdout | 0 of 36 | 0.0% |
| audits_only | 0 of 36 | 0.0% |
| llm_judge | 0 of 36 | 0.0% |
| react | 0 of 36 | 0.0% |
| flaml | 0 of 36 | 0.0% |

A failed unit counts as not detected and not remediated, and as a non-pass (a false alarm) on a clean task.

### Standard-tier fallbacks (Flex unavailable after 10 attempts)

| system | fallback calls | of paid calls | units with a fallback |
|---|---|---|---|
| holdout | 0 | 0.0% | 0 of 36 |
| llm_judge | 0 | 0.0% | 0 of 36 |
| react | 0 | 0.0% | 0 of 36 |

Fallback calls are the same request on gemini-3.8-flash, priced at the standard rate.

### Real-world issues: original vs anonymised (detection rate over seeds)

| case / issue | holdout orig / anon | audits_only orig / anon | llm_judge orig / anon | react orig / anon | flaml orig / anon |
|---|---|---|---|---|---|
| bank_marketing / duration_post_outcome | 0.0% / 50.0% | 0.0% / 0.0% | 0.0% / 0.0% | 50.0% / 100.0% | 0.0% / 0.0% |
| bank_marketing / ordered_by_date (secondary) | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| covertype_ordered / order_dependence | 100.0% / 100.0% | 100.0% / 100.0% | 50.0% / 100.0% | 50.0% / 100.0% | 0.0% / 0.0% |
| diabetes_readmission / encounter_id_identifier | 100.0% / 100.0% | 100.0% / 100.0% | 0.0% / 0.0% | 50.0% / 0.0% | 0.0% / 0.0% |
| diabetes_readmission / repeated_patients | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 50.0% / 0.0% | 0.0% / 0.0% |
| electricity / temporal_dependence | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 0.0% / 0.0% |
| titanic / boat_post_outcome | 0.0% / 0.0% | 100.0% / 100.0% | 0.0% / 0.0% | 100.0% / 100.0% | 0.0% / 0.0% |
| titanic / body_post_outcome | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 0.0% / 0.0% |

#### H5 (exploratory): paired original - anonymised detection, per system

| system | mean difference [95% CI] |
|---|---|
| holdout | -0.071 [-0.214, +0.000] |
| audits_only | +0.000 [+0.000, +0.000] |
| llm_judge | -0.071 [-0.214, +0.000] |
| react | +0.000 [-0.286, +0.286] |
| flaml | +0.000 [+0.000, +0.000] |

### False alarms on clean controls (task = unit: any seed or variant alarming or failing counts)

| system | critical | one-sided 95% upper bound | warning or above | any issue (info too) |
|---|---|---|---|---|
| holdout | 1/4 = 25.0% | 75.1% | 100.0% | 100.0% |
| audits_only | 1/4 = 25.0% | 75.1% | 100.0% | 100.0% |
| llm_judge | 0/4 = 0.0% | 52.7% | 0.0% | 25.0% |
| react | 0/4 = 0.0% | 52.7% | 50.0% | 100.0% |
| flaml | 0/4 = 0.0% | 52.7% | 0.0% | 0.0% |

### Scores and optimism on both holdouts (completed runs; task = unit; mean [95% CI])

| system | optimism gap deploy | optimism gap iid | holdout_deploy vs dummy | holdout_iid vs dummy | holdout_deploy - FLAML |
|---|---|---|---|---|---|
| holdout | +0.0221 [-0.0055, +0.0530] | +0.0022 [-0.0097, +0.0168] | 0.639 | 0.725 | +0.0062 |
| audits_only | +0.0286 [-0.0001, +0.0608] | +0.0105 [-0.0036, +0.0260] | 0.631 | 0.709 | +0.0096 |
| llm_judge | +0.0071 [-0.0149, +0.0324] | -0.0063 [-0.0339, +0.0159] | 0.642 | 0.715 | +0.0071 |
| react | -0.0081 [-0.0333, +0.0199] | -0.0245 [-0.0685, +0.0142] | 0.728 | 0.804 | +0.0057 |
| flaml | +0.0442 [+0.0109, +0.0805] | +0.0171 [+0.0019, +0.0370] | 0.640 | 0.731 | - |

Optimism gap: the system's own validation claim minus its holdout score in the same metric (positive = optimistic). holdout_deploy is deployment-faithful (leak columns unavailable, unseen entities, the future); holdout_iid keeps leaks intact. Real cases scored on detection and remediation only are excluded.

### Cost and answer quality (completed runs)

| system | cost/run (list) | cost/run (cache-adjusted) | compute time/run | API wait/run | calls | retries | first-attempt valid | repairs/run | substitutions/run |
|---|---|---|---|---|---|---|---|---|---|
| holdout | Rs 1.29 | Rs 1.29 | 80 s | 33 s | 4.1 | 0 | 93.3% | 0.19 | 0.00 |
| audits_only | Rs 0.00 | Rs 0.00 | 71 s | 0 s | 0.0 | 0 | - | 0.00 | 0.00 |
| llm_judge | Rs 1.30 | Rs 1.08 | 63 s | 31 s | 4.9 | 0 | 96.9% | 0.11 | 0.00 |
| react | Rs 9.47 | Rs 8.32 | 134 s | 104 s | 12.2 | 0 | 98.9% | 0.13 | 0.00 |
| flaml | Rs 0.00 | Rs 0.00 | 75 s | 0 s | 0.0 | 0 | - | 0.00 | 0.00 |

### Timeouts

| system | units | react_fit_step | react_step |
|---|---|---|---|
| holdout | 0 of 36 | 0 | 0 |
| audits_only | 0 of 36 | 0 | 0 |
| llm_judge | 0 of 36 | 0 | 0 |
| react | 11 of 36 | 10 | 2 |
| flaml | 0 of 36 | 0 | 0 |

Cells: total timeouts; units: runs with at least one timeout.

### A11 flags on generated reports of tasks whose goal is not causal

An upper bound on A11's false-positive rate: whether a flagged sentence really is a causal claim needs a human label (sentences listed in the appendix).

| system | reports | flagged | rate |
|---|---|---|---|
| holdout | 36 | 0 | 0.0% |
| audits_only | 36 | 0 | 0.0% |
| llm_judge | 36 | 2 | 5.6% |
| react | 30 | 3 | 10.0% |

