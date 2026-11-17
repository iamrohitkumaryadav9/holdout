"""Data layer: load, profile (target-blind), split and seal.

The order of operations is the product: nothing here computes a feature-target statistic,
and the holdout is written to disk and sealed before any modelling code receives data.
"""
