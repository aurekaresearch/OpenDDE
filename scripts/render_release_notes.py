#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Render one GitHub Release body from the matching CHANGELOG section."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STABLE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
CHANGELOG_HEADING_PATTERN = re.compile(
    r"^## \[(?P<version>Unreleased|[0-9]+\.[0-9]+\.[0-9]+)\]"
    r"(?: - (?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}))?$",
    re.MULTILINE,
)
REFERENCE_LINK_PATTERN = re.compile(r"^\[[^]]+\]:\s+\S+", re.MULTILINE)
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD)\b", re.IGNORECASE)
FENCE_OPEN_PATTERN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")


@dataclass(frozen=True)
class ChangelogSection:
    """One parsed CHANGELOG section."""

    version: str
    date: str | None
    body: str


def _non_fenced_lines(markdown: str) -> list[tuple[int, str]]:
    """Return (index, line) pairs for lines outside fenced code blocks."""
    lines: list[tuple[int, str]] = []
    in_fence = False
    for index, line in enumerate(markdown.splitlines()):
        if FENCE_OPEN_PATTERN.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((index, line))
    return lines


def parse_changelog(changelog: str) -> list[ChangelogSection]:
    """Parse release sections, ignoring headings inside fenced code blocks."""
    lines = changelog.splitlines()
    non_fenced = _non_fenced_lines(changelog)
    headings = [
        (index, match)
        for index, line in non_fenced
        if (match := CHANGELOG_HEADING_PATTERN.match(line)) is not None
    ]
    reference_indices = [
        index for index, line in non_fenced if REFERENCE_LINK_PATTERN.match(line)
    ]
    sections: list[ChangelogSection] = []
    for order, (index, match) in enumerate(headings):
        start = index + 1
        if order + 1 < len(headings):
            end = headings[order + 1][0]
        else:
            end = next((i for i in reference_indices if i >= start), len(lines))
        sections.append(
            ChangelogSection(
                version=match.group("version"),
                date=match.group("date"),
                body="\n".join(lines[start:end]).strip(),
            )
        )
    return sections


def has_placeholder(markdown: str) -> bool:
    """Return whether Markdown outside fenced code has a TODO/TBD placeholder."""
    return any(
        PLACEHOLDER_PATTERN.search(line) for _, line in _non_fenced_lines(markdown)
    )


def extract_release_section(
    changelog: str,
    version: str,
    sections: list[ChangelogSection] | None = None,
) -> str:
    """Extract and validate the body of one stable-version CHANGELOG section."""
    if STABLE_VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"version must be stable X.Y.Z SemVer: {version}")

    parsed_sections = sections if sections is not None else parse_changelog(changelog)
    matching = [section for section in parsed_sections if section.version == version]
    if len(matching) != 1:
        raise ValueError(
            f"CHANGELOG must contain exactly one release section for {version}"
        )

    section = matching[0].body
    if not section or section == "No changes yet.":
        raise ValueError(f"CHANGELOG release section for {version} is empty")
    if has_placeholder(section):
        raise ValueError(f"CHANGELOG release section for {version} has a placeholder")
    return section


def render_release_notes(
    changelog: str,
    version: str,
    sections: list[ChangelogSection] | None = None,
) -> str:
    """Render one release body directly from the matching CHANGELOG section."""
    return f"{extract_release_section(changelog, version, sections)}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Stable X.Y.Z version")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=ROOT / "CHANGELOG.md",
        help="CHANGELOG to read (default: repository CHANGELOG.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="File to write; omit to print the rendered notes",
    )
    args = parser.parse_args()

    try:
        notes = render_release_notes(
            args.changelog.read_text(encoding="utf-8"), args.version
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.output is None:
        sys.stdout.write(notes)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(notes, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
