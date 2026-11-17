"""The real-world suite (``bench/suites/real_world_v1.toml``) and its anonymised variants.

The TOML is the frozen ground truth (committed before any benchmark system ran). This
module parses and validates it against the dataset registry and the audit enums, loads a
case's frame (original or anonymised), and builds the anonymisation mapping: columns renamed
``col_1..col_n`` in file order, the target renamed ``target``. The mappings are written to
``real_world_v1_anonymised.json`` so the renaming is recorded, not recomputed silently.
"""

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from bench.data.openml import load_frame
from bench.data.registry import DatasetSpec, load_registry, require_pool
from holdout.audit.types import Category, Severity
from holdout.core.schemas import EntityScope, Task

SUITES_DIR = Path(__file__).resolve().parent
REAL_WORLD_PATH = SUITES_DIR / "real_world_v1.toml"
ANONYMISED_PATH = SUITES_DIR / "real_world_v1_anonymised.json"
ANON_TARGET = "target"

Variant = Literal["original", "anonymised"]
_REMEDIATION_PREFIXES = ("drop:", "split:group:", "split:time:")


class SuiteError(ValueError):
    """A suite file is inconsistent with the registry, the enums or its own rules."""


@dataclass(frozen=True)
class RealIssue:
    id: str
    categories: tuple[Category, ...]
    min_severity: Severity
    columns: tuple[str, ...]
    remediation: str
    scored: bool
    secondary: bool
    citation_url: str
    citation: str
    summary: str


@dataclass(frozen=True)
class RealCase:
    id: str
    dataset: str
    openml_id: int
    openml_version: int
    target: str
    task: Task
    kind: Literal["issues", "clean"]
    goal: str
    anonymised_goal: str
    entity_scope: EntityScope | None
    issues: tuple[RealIssue, ...] = field(default=())
    citation_url: str = ""
    citation: str = ""
    summary: str = ""

    def goal_for(self, variant: Variant) -> str:
        return self.goal if variant == "original" else self.anonymised_goal

    @property
    def scored_issues(self) -> tuple[RealIssue, ...]:
        return tuple(i for i in self.issues if i.scored)


def suite_sha256(path: Path) -> str:
    """sha256 of the file with line endings normalised to LF (a Windows checkout with CRLF
    and a Linux one hash the same)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_real_world(
    path: Path = REAL_WORLD_PATH, registry: dict[str, DatasetSpec] | None = None
) -> list[RealCase]:
    """Parse and validate the real-world suite.

    Raises:
        SuiteError: on an unknown dataset, a dataset outside the real-world pool, an id or
            version that differs from the registry, an unknown category or severity, a
            malformed remediation, a scored issue without a citation, or a clean control
            without a stated reason.
    """
    registry = registry if registry is not None else load_registry()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = [_case(entry) for entry in raw.get("case", [])]
    problems: list[str] = []
    ids = [c.id for c in cases]
    problems += [f"case id {i!r} repeats" for i in sorted(set(ids)) if ids.count(i) > 1]
    for case in cases:
        spec = registry.get(case.dataset)
        if spec is None:
            problems.append(f"{case.id}: unknown dataset {case.dataset!r}")
            continue
        try:
            require_pool(spec, "real_world")
        except ValueError as exc:
            problems.append(f"{case.id}: {exc}")
        if (spec.openml_id, spec.version) != (case.openml_id, case.openml_version):
            problems.append(f"{case.id}: OpenML id/version differ from the registry")
        if spec.target != case.target or spec.task is not case.task:
            problems.append(f"{case.id}: target or task differ from the registry")
        if case.kind == "clean":
            if case.issues:
                problems.append(f"{case.id}: a clean control has no issues")
            if not (case.citation_url and case.summary):
                problems.append(f"{case.id}: a clean control needs a cited reason")
        elif not case.scored_issues:
            problems.append(f"{case.id}: an issues case needs at least one scored issue")
        for issue in case.issues:
            where = f"{case.id}/{issue.id}"
            if issue.scored:
                if not issue.categories:
                    problems.append(f"{where}: a scored issue needs a category")
                if not issue.remediation.startswith(_REMEDIATION_PREFIXES):
                    problems.append(f"{where}: bad remediation {issue.remediation!r}")
            if not (issue.citation_url.startswith("https://") and issue.summary):
                problems.append(f"{where}: every issue needs a citation URL and summary")
    if problems:
        raise SuiteError("invalid real-world suite:\n  " + "\n  ".join(problems))
    return cases


def _case(entry: dict[str, Any]) -> RealCase:
    try:
        scope = entry.get("entity_scope")
        return RealCase(
            id=str(entry["id"]),
            dataset=str(entry["dataset"]),
            openml_id=int(entry["openml_id"]),
            openml_version=int(entry["openml_version"]),
            target=str(entry["target"]),
            task=Task(entry["task"]),
            kind=entry["kind"],
            goal=str(entry["goal"]),
            anonymised_goal=str(entry["anonymised_goal"]),
            entity_scope=EntityScope(scope) if scope else None,
            issues=tuple(_issue(i) for i in entry.get("issue", [])),
            citation_url=str(entry.get("citation_url", "")),
            citation=str(entry.get("citation", "")),
            summary=str(entry.get("summary", "")),
        )
    except (KeyError, ValueError) as exc:
        raise SuiteError(f"bad case {entry.get('id')!r}: {exc}") from exc


def _issue(entry: dict[str, Any]) -> RealIssue:
    return RealIssue(
        id=str(entry["id"]),
        categories=tuple(Category(c) for c in entry["categories"]),
        min_severity=Severity(entry["min_severity"]),
        columns=tuple(entry["columns"]),
        remediation=str(entry["remediation"]),
        scored=bool(entry["scored"]),
        secondary=bool(entry.get("secondary", False)),
        citation_url=str(entry["citation_url"]),
        citation=str(entry["citation"]),
        summary=str(entry["summary"]),
    )


# ----------------------------------------------------------------------------- anonymising


@dataclass(frozen=True)
class Anonymisation:
    """Original name -> anonymous name, for every column including the target."""

    columns: dict[str, str]

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(frame.columns) ^ set(self.columns)
        if missing:
            raise SuiteError(f"the frame's columns differ from the mapping: {sorted(missing)}")
        return frame.rename(columns=self.columns)

    def rename(self, name: str) -> str:
        return self.columns[name]


def anonymisation_for(columns: list[str], target: str) -> Anonymisation:
    """``col_1..col_n`` for the features in file order; the target becomes ``target``."""
    mapping: dict[str, str] = {}
    index = 0
    for column in columns:
        if column == target:
            mapping[column] = ANON_TARGET
        else:
            index += 1
            mapping[column] = f"col_{index}"
    return Anonymisation(mapping)


def write_anonymisations(cases: list[RealCase], path: Path = ANONYMISED_PATH) -> dict[str, Any]:
    """Build the mapping of every case from its (verified) data file and write it."""
    registry = load_registry()
    payload: dict[str, Any] = {"schema_version": 1, "suite_sha256": suite_sha256(REAL_WORLD_PATH)}
    payload["cases"] = {}
    for case in cases:
        frame, target, _ = load_frame(registry[case.dataset])
        payload["cases"][case.id] = anonymisation_for(list(frame.columns), target).columns
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return payload


def load_anonymisations(path: Path = ANONYMISED_PATH) -> dict[str, Anonymisation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {case: Anonymisation(dict(cols)) for case, cols in data["cases"].items()}


def case_frame(
    case: RealCase,
    variant: Variant,
    *,
    registry: dict[str, DatasetSpec] | None = None,
    anonymisations: dict[str, Anonymisation] | None = None,
    allow_download: bool = False,
) -> tuple[pd.DataFrame, str, str, Anonymisation | None]:
    """(frame, target, goal, mapping) for one case and variant. The mapping is None for the
    original variant; for the anonymised one it is the recorded mapping, which must match
    the data's columns exactly."""
    registry = registry if registry is not None else load_registry()
    frame, target, _ = load_frame(registry[case.dataset], allow_download=allow_download)
    if variant == "original":
        return frame, target, case.goal, None
    mapping = (anonymisations or load_anonymisations())[case.id]
    if mapping != anonymisation_for(list(frame.columns), target):
        raise SuiteError(f"{case.id}: the recorded anonymisation no longer matches the data")
    return mapping.apply(frame), ANON_TARGET, case.anonymised_goal, mapping
