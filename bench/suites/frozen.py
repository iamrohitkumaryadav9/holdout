"""Pinned hashes of the frozen suites. Tests compare the committed files against these, so
nothing can regenerate a suite silently: changing one requires a new version (``*_v2``)
and a documented amendment to the pre-registration."""

from typing import Final

#: Hashes use LF line endings (see bench.suites.real_world.suite_sha256). real_world_v1 was
#: amended once before any system ran on it (2026-11-18: per-case deploy_holdout,
#: holdout_scored and deploy_note), as requested; the change is in its git history.
REAL_WORLD_V1_SHA256: Final = "df2f25beee8375ecae432a1a4ed4a67d850f8931fcbf73877c46c59544b8658e"
REAL_WORLD_V1_ANONYMISED_SHA256: Final = (
    "f00d3c97e2bd68c2249620588ae55428427e4549f00c1c1282e038c20130a020"
)
FINAL_V1_SHA256: Final = "d4de5207759396fc4888e4552eb8792168c240638603e875d0b92f5cf3049b8b"
DEV_V1_SHA256: Final = "cb17bb2702d08b7e3924dc812ed6baeae533f8c9542861852c0e15a1e072f9cf"
