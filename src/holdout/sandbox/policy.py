"""Static allowlist check for EDA snippets (the first of several layers; see the package docs)."""

import ast
from dataclasses import dataclass
from typing import Final

#: Modules a snippet may import (``import x`` / ``from x import y``), by full dotted name.
ALLOWED_IMPORTS: Final = frozenset(
    {"pandas", "numpy", "scipy", "scipy.stats", "matplotlib", "matplotlib.pyplot"}
)

#: Built-in names a snippet may not reference at all.
BLOCKED_NAMES: Final = frozenset(
    {
        "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
        "memoryview", "__builtins__", "__loader__", "__spec__",
    }
)  # fmt: skip

#: Attribute names that read or write files, evaluate strings, or reach the system.
BLOCKED_ATTRIBUTES: Final = frozenset(
    {
        "eval", "query", "system", "popen", "load", "save", "savez", "savez_compressed",
        "fromfile", "tofile", "loadtxt", "savetxt", "genfromtxt", "memmap", "ctypeslib",
        "f2py", "show",
    }
)  # fmt: skip

#: Attribute prefixes for pandas/numpy I/O (``read_csv``, ``to_pickle``, ...) ...
BLOCKED_ATTRIBUTE_PREFIXES: Final = ("read_", "to_")
#: ... except these in-memory conversions.
ALLOWED_TO_ATTRIBUTES: Final = frozenset(
    {"to_string", "to_numpy", "to_dict", "to_list", "to_frame", "to_series", "to_period"}
)


@dataclass(frozen=True)
class Violation:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


def check_snippet(source: str) -> list[Violation]:
    """Every construct outside the allowlist; empty when the snippet may run."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Violation(exc.lineno or 0, f"syntax error: {exc.msg}")]
    violations: list[Violation] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    violations.append(Violation(line, f"import of {alias.name!r} is not allowed"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module not in ALLOWED_IMPORTS:
                violations.append(Violation(line, f"import from {module!r} is not allowed"))
            for alias in node.names:
                full = f"{module}.{alias.name}"
                if alias.name == "*" or (
                    full not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS
                ):
                    violations.append(Violation(line, f"import of {full!r} is not allowed"))
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                violations.append(Violation(line, f"use of {node.id!r} is not allowed"))
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                violations.append(Violation(line, f"private or dunder attribute {attr!r}"))
            elif attr in BLOCKED_ATTRIBUTES:
                violations.append(Violation(line, f"attribute {attr!r} is not allowed"))
            elif attr.startswith(BLOCKED_ATTRIBUTE_PREFIXES) and attr not in ALLOWED_TO_ATTRIBUTES:
                violations.append(Violation(line, f"file I/O attribute {attr!r} is not allowed"))
        elif isinstance(node, ast.Global | ast.Nonlocal):
            violations.append(Violation(line, "global and nonlocal are not allowed"))
    return sorted(violations, key=lambda v: v.line)
