"""Trap generators: planted data problems with known ground truth, and their clean twins."""

from bench.traps.catalog import TRAPS, build_case, clean_twin
from bench.traps.types import TrapCase, TrapTruth

__all__ = ["TRAPS", "TrapCase", "TrapTruth", "build_case", "clean_twin"]
