"""Plan layer: the typed contract a proposer fills, and the code that decides.

A proposer (the offline default today, an LLM later) only ever produces a ``ModelPlan``.
``validate_plan`` judges it against the data profile, ``compile_plan`` turns an accepted
plan into a leakage-safe scikit-learn pipeline, and ``evaluate_cv`` scores it with folds
that mirror the holdout split.
"""
