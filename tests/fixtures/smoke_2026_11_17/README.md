# Fixtures from the first paid smoke runs (2026-11-17, gemini-3.8-flash)

Runs `20261117T204403Z-e483f5` (churn example) and `20261117T204437Z-aacba8` (direct_leak
trap). The raw rejected answers were not stored at the time (only valid answers were
cached, and the smoke ran without `--cache`), so the rejected answers here are
**reconstructed** from what the run kept. Since then every rejected answer is kept in the
manifest (`agent.rejected_attempts`, first 4000 characters).

- `plan_*_repaired.json`: the plan the model returned on its repair attempt, taken verbatim
  from `agent.rounds[0].plan`.
- `plan_*_rejected.json`: the same plan with `model.name` set to the exact invalid tag from
  the recorded validation error (`HistGradientBoostingClassifier` for churn,
  `hist_gradient_boosting` for the trap). That error was the only one recorded for the
  first attempt; the other fields of the first answer are not known exactly.
- `report_direct_leak_first_draft.json`: the trap run's first report draft. It is the final
  report with the code-added caveat removed and the numbers put back as placeholders. The
  sentence A11 flagged ("On the independent holdout set, the model achieved a score of ...,
  resulting in an optimism gap of ...", quoted verbatim in the run's A11 evidence) is
  restored at the end of the summary; its position in the draft is a reconstruction.
