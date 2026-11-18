# Pilot report

Suite `dev_v1`, results in `results\pilot`.

## Units

| system | task | seed | status | Rs | seconds | calls | retries | first-attempt valid | error |
|---|---|---|---|---|---|---|---|---|---|
| holdout | direct_leak@adult | 11 | completed | 1.092 | 121 | 5 | 0 | 3/3 |  |
| audits_only | direct_leak@adult | 11 | completed | 0.000 | 54 | 0 | 0 | - |  |
| flaml | direct_leak@adult | 11 | completed | 0.000 | 63 | 0 | 0 | - |  |
| holdout | id_with_signal@credit_approval | 11 | completed | 1.243 | 40 | 5 | 0 | 2/3 |  |
| audits_only | id_with_signal@credit_approval | 11 | completed | 0.000 | 23 | 0 | 0 | - |  |
| flaml | id_with_signal@credit_approval | 11 | completed | 0.000 | 60 | 0 | 0 | - |  |
| holdout | noisy_proxy@kin8nm | 11 | completed | 0.917 | 21 | 4 | 0 | 3/3 |  |
| audits_only | noisy_proxy@kin8nm | 11 | completed | 0.000 | 9 | 0 | 0 | - |  |
| flaml | noisy_proxy@kin8nm | 11 | completed | 0.000 | 62 | 0 | 0 | - |  |
| llm_judge | direct_leak@adult | 11 | completed | 1.757 | 92 | 6 | 0 | 4/4 |  |
| react | direct_leak@adult | 11 | completed | 8.206 | 209 | 13 | 0 | 13/13 |  |
| llm_judge | id_with_signal@credit_approval | 11 | completed | 0.817 | 21 | 4 | 0 | 3/3 |  |
| react | id_with_signal@credit_approval | 11 | completed | 6.797 | 119 | 11 | 0 | 11/11 |  |
| llm_judge | noisy_proxy@kin8nm | 11 | completed | 0.962 | 25 | 4 | 0 | 3/3 |  |
| react | noisy_proxy@kin8nm | 11 | completed | 9.297 | 291 | 15 | 0 | 15/15 | iid holdout scoring failed: scoring react's model on iid failed: unknown metric 'neg_mean_squared_error'; deploy holdout |

## What consumed the money

| system | node | Rs |
|---|---|---|
| holdout | explain_and_fix | 1.074 |
| holdout | plan | 1.044 |
| holdout | report | 0.793 |
| holdout | audit | 0.340 |
| llm_judge | audit | 1.227 |
| llm_judge | plan | 1.047 |
| llm_judge | report | 0.908 |
| llm_judge | explain_and_fix | 0.354 |
| react | react | 24.301 |

ReAct on `direct_leak@adult`: 13 steps; Rs per LLM call: 0.182, 0.326, 0.339, 0.393, 0.463, 0.584, 0.600, 0.739, 0.834, 0.814, 0.951, 1.008, 0.975; prompt tokens 78624, observation tokens re-sent with the history (estimate) 14437 truncated vs 14507 untruncated.

ReAct on `id_with_signal@credit_approval`: 11 steps; Rs per LLM call: 0.162, 0.271, 0.238, 0.462, 0.611, 0.585, 0.834, 0.818, 0.940, 0.961, 0.914; prompt tokens 64181, observation tokens re-sent with the history (estimate) 13208 truncated vs 24423 untruncated.

ReAct on `noisy_proxy@kin8nm`: 15 steps; Rs per LLM call: 0.151, 0.137, 0.308, 0.428, 0.427, 0.514, 0.622, 0.617, 0.759, 0.732, 0.844, 0.910, 0.934, 0.949, 0.964; prompt tokens 96799, observation tokens re-sent with the history (estimate) 13891 truncated vs 13891 untruncated.

Ledger spend since the pilot started: Rs 32.163; attributed to results: Rs 31.089; unattributed (e.g. a unit that failed or paused after spending, or other processes): Rs 1.075.

## Measured per run

| system | runs | Rs/run (mean) | Rs/run (max) | seconds/run (mean) |
|---|---|---|---|---|
| audits_only | 3 | 0.000 | 0.000 | 29 |
| flaml | 3 | 0.000 | 0.000 | 62 |
| holdout | 3 | 1.084 | 1.243 | 61 |
| llm_judge | 3 | 1.179 | 1.757 | 46 |
| react | 3 | 8.100 | 9.297 | 206 |

## Projected full benchmark (116 units per system: planted 80, real world 36; margin 20%)

| system | Rs (with ReAct truncation) | Rs (without truncation) | hours |
|---|---|---|---|
| audits_only | 0.00 | 0.00 | 0.9 |
| flaml | 0.00 | 0.00 | 2.0 |
| holdout | 150.88 | 150.88 | 2.0 |
| llm_judge | 164.08 | 164.08 | 1.5 |
| react | 1127.55 | 1164.51 | 6.6 |
| **total** | **1442.52** | **1479.47** | |

- Project spend so far: Rs 37.40; remaining under the Rs 750 cap (benchmark spend may use the reserve): Rs 712.60.
- Benchmark reserve: Rs 600.
- Prepaid balance left (estimate): Rs 709.60.
- Prices as of 2026-11-17 (0.75 / 3.75 USD per 1M input / output tokens today); Gemini 3.8 Flash paid-tier prices double from 2027-01-01 (input 0.75 -> 1.50, output 3.75 -> 7.50 USD per 1M tokens).
- The pilot ran on three dev tasks (adult 48k rows, credit-approval 690, kin8nm 8k); the final bases range from 1k to 30k rows, so per-run cost may differ.
