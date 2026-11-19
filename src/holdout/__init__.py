"""Holdout: a self-auditing data-science agent for tabular CSVs.

The governing rule is *seal before see*: the holdout split is written to disk and locked
before anything computes a feature-target statistic, and it can be scored exactly once per
run through one logged call. Proposals (today a default plan, later an LLM) are data; the
deterministic code in this package decides what is allowed.
"""

__version__ = "1.1.0"
