"""Pinned hashes of the frozen suites. Tests compare the committed files against these, so
nothing can regenerate a suite silently: changing one requires a new version (``*_v2``)
and a documented amendment to the pre-registration."""

from typing import Final

#: Hashes use LF line endings (see bench.suites.real_world.suite_sha256).
REAL_WORLD_V1_SHA256: Final = "bcfb5d6119e3a4bbec1201260ec6fb5c2c6264bfb29e533784e34ac5cb9bb7dd"
REAL_WORLD_V1_ANONYMISED_SHA256: Final = (
    "3380f21ac57a8a3faf11df1e380c2e52027018c125fe4922a70128bd9de3651d"
)
FINAL_V1_SHA256: Final = "d4de5207759396fc4888e4552eb8792168c240638603e875d0b92f5cf3049b8b"
DEV_V1_SHA256: Final = "cb17bb2702d08b7e3924dc812ed6baeae533f8c9542861852c0e15a1e072f9cf"
