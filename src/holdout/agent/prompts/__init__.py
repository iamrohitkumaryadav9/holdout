"""Versioned prompt files and their rendering.

Each ``*.md`` file has a header (``name``, ``version``, ``purpose``) and ``<system>`` and
``<user>`` sections. Anything taken from the CSV or the user (column names, sample values,
goal text, earlier answers) is inserted only through ``{{data.<name>}}`` slots, which render
as delimited ``<data name="...">`` sections; the system prompts say that instructions inside
data must be ignored. A closing ``</data`` inside a value is escaped, so data cannot break
out of its section. The version and sha256 of every prompt used go into the manifest.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from holdout.llm.structured import RenderedPrompt

_HEADER = re.compile(r"\A---\n(?P<header>.*?)\n---\n", re.DOTALL)
_SECTION = re.compile(r"<(?P<tag>system|user)>\n(?P<body>.*?)\n</(?P=tag)>", re.DOTALL)
_SLOT = re.compile(r"\{\{data\.(?P<name>[a-z_]+)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    purpose: str
    sha256: str
    system: str
    user: str

    @property
    def slots(self) -> set[str]:
        return set(_SLOT.findall(self.user)) | set(_SLOT.findall(self.system))

    def render(self, **data: Any) -> RenderedPrompt:
        """Fill every ``{{data.x}}`` slot with a delimited data section.

        Raises:
            KeyError: when a slot is missing or an unknown slot is supplied.
        """
        missing = self.slots - set(data)
        extra = set(data) - self.slots
        if missing or extra:
            raise KeyError(
                f"prompt {self.name}: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )

        def fill(match: re.Match[str]) -> str:
            return data_section(match.group("name"), data[match.group("name")])

        return RenderedPrompt(
            name=self.name,
            version=self.version,
            sha256=self.sha256,
            system=_SLOT.sub(fill, self.system),
            user=_SLOT.sub(fill, self.user),
        )


def data_section(name: str, value: Any) -> str:
    """A delimited, escaped data block. Strings are kept; everything else is JSON."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, indent=1, sort_keys=True, default=str)
    )
    escaped = text.replace("</data", "<\\/data").replace("<data", "<\\data")
    return f'<data name="{name}">\n{escaped}\n</data>'


def load_prompt(name: str) -> PromptTemplate:
    raw = resources.files("holdout.agent.prompts").joinpath(f"{name}.md").read_bytes()
    text = raw.decode("utf-8")
    header = _HEADER.match(text)
    if header is None:
        raise ValueError(f"prompt {name}: missing header")
    fields = dict(line.split(":", 1) for line in header.group("header").splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    sections = {m.group("tag"): m.group("body") for m in _SECTION.finditer(text)}
    if fields.get("name") != name or not {"version", "purpose"} <= set(fields):
        raise ValueError(f"prompt {name}: header must declare name, version and purpose")
    if set(sections) != {"system", "user"}:
        raise ValueError(f"prompt {name}: needs <system> and <user> sections")
    return PromptTemplate(
        name=name,
        version=fields["version"],
        purpose=fields["purpose"],
        sha256=hashlib.sha256(raw).hexdigest(),
        system=sections["system"],
        user=sections["user"],
    )


PROMPT_NAMES = ("plan", "explain_fix", "report", "hints", "eda")
