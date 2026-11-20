"""Wording shared by the web app, README.md and docs/results_v11.md.

A test checks that README.md and docs/results_v11.md contain each sentence here verbatim,
so the app and the documents carry the same labels and caveats.
"""

from typing import Final

from holdout import __version__

REPO_URL: Final = "https://github.com/iamrohitkumaryadav9/holdout"
TAG_URL: Final = f"{REPO_URL}/tree/v{__version__}"

PREREGISTERED_LABEL: Final = "Pre-registered results (v1.0.0)"
POSTHOC_LABEL: Final = "Post-hoc exploratory: audits_only v1.1 vs v1.0 (not pre-registered)"
OPTIMISTIC: Final = "optimistic: the fixes were derived from errors on this suite"
PRIMARY: Final = "primary comparison"

#: The pre-registered results (docs/results.md), leading wherever results are shown.
PREREGISTERED_SUMMARY: Final = (
    "- Detection recall on final_v1 (28 planted-trap tasks, 2 seeds; task = unit; 95% "
    "bootstrap CI): Holdout (the agent) 51.8% [33.9%, 69.6%]; audits_only (the "
    "deterministic audit layer alone) 75.0% [57.1%, 89.3%]; ReAct 60.7% [42.9%, 78.6%]; "
    "LLM judge 35.7% [19.6%, 53.6%]; FLAML 0.0%.",
    "- audits_only vs ReAct: 75.0% vs 60.7%, exact McNemar p = 0.27, not significant. "
    "Holdout vs ReAct: 51.8% vs 60.7%, p = 0.58, not significant. Holdout vs its own audit "
    "layer: 51.8% vs 75.0%, p = 0.02 (the LLM planner avoided problems the audits would "
    "have reported; see docs/error_analysis.md).",
    "- Pre-registered hypotheses: H1 (Holdout beats ReAct on detection), H3 (smaller "
    "optimism gap) and H4 (audits beat the LLM judge) not supported; H2 (critical false "
    "alarms on clean tasks at most 10%) failed: 5 of 16 clean tasks (31.3%).",
    "- Real data (real_world_v1, 7 issues in 9 public datasets): audits_only's detection "
    "rate is 57.1% (issues x variants, seeds averaged), but only 3.6% at critical severity; "
    "all 4 clean controls get warnings.",
    "- Cost per run on Gemini: Holdout Rs 1.22 (final_v1) and Rs 1.29 (real_world_v1); "
    "ReAct Rs 8.84 (final_v1) and Rs 9.47 (real_world_v1).",
)

#: The post-hoc v1.1 comparison (docs/results_v11.md), always second and labelled so.
POSTHOC_SUMMARY: Final = (
    "- dev_v1 (primary comparison; both versions run fresh): recall 77.8% -> 88.9%, a gain "
    "of one task, causal_question@credit_approval on both seeds; no clean false-alarm "
    "change (0/3 critical, 1/3 warning in both).",
    "- final_v1 (optimistic: the fixes were derived from errors on this suite): recall "
    "75.0% -> 85.7% like for like; clean controls with a critical false alarm 4/12 -> 1/12, "
    "and the remaining one is a timeout, not a flag.",
    "- real_world_v1 (optimistic: the fixes were derived from errors on this suite): "
    "detection unchanged (57.1%, 3.6% at critical); remediation 17.9% -> 3.6%.",
)

#: The trade-offs of v1.1, stated plainly wherever v1.1 numbers appear.
TRADEOFFS: Final = (
    "Titanic `body` remediation lost on real_world_v1 (remediation 17.9% -> 3.6%): v1.0 "
    "dropped this post-outcome leak for the wrong reason (A3 misread it as an identifier). "
    "It is left unpatched deliberately, to avoid tuning on the test suite.",
    "The messy_formats@house_16h gain on final_v1 is a timeout artifact (v1.0's audit ran "
    "out of time; v1.1's finishes), not a better format check, and it is not counted as an "
    "improvement: like for like, final_v1 recall is 75.0% -> 85.7%.",
    "All 4 real-world clean controls still get warnings (mushroom, credit_g, kr_vs_kp, "
    "abalone), as in v1.0.",
    "Fix 5 (deduplication as a plan action) does not change the duplicate traps: Holdout's "
    "own sealed holdout still overlaps the development rows, so the holdout is invalidated "
    "and the run ends.",
    "The Gemini planner path of v1.1 is untested live: no API calls were made, and it is "
    "covered by fake-LLM tests only.",
)

CRITICAL_EXPLAINED: Final = (
    "Critical: Holdout blocks the holdout evaluation. A critical finding means the score "
    "cannot be trusted as it stands (a leak, a contaminated split or a misleading metric); "
    "fix it or start a new run with the corrected split."
)
CRITICAL_FINDING_EXPLAINED: Final = (
    "Critical: found on the raw data before planning; the score cannot be trusted unless "
    "it is fixed. Holdout fixes what code can fix (the status column says whether it was "
    "resolved and how) and keeps the holdout sealed while a critical problem remains open."
)
WARNING_EXPLAINED: Final = (
    "Warnings need your judgement, not automatic rejection. They are common on real data: "
    "in the benchmark, all 4 real-world clean controls got at least one warning. A warning "
    "says where to look (a very strong single column, a possible post-outcome name, a weak "
    "time signal); decide with what you know about how the data was collected."
)
INFO_EXPLAINED: Final = "Info: recorded for the audit trail; nothing to act on."
