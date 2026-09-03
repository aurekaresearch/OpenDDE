#!/usr/bin/env python3
"""Validate that OpenDDE release metadata is internally consistent.

This check does not fetch remote model assets. Maintainers must verify published
byte sizes and SHA-256 digests independently before tagging a release.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

from render_release_notes import (
    extract_release_section,
    has_placeholder,
    parse_changelog,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opendde.config.model_manifest import (  # noqa: E402
    ManifestError,
    checkpoint_url,
    load_model_manifest,
)
from opendde.config.model_registry import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    model_configs,
)

VERSION_PATTERN = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
STABLE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REPOSITORY_URL = "https://github.com/aurekaresearch/OpenDDE"


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _package_version(errors: list[str]) -> str:
    match = VERSION_PATTERN.search(_read_text("opendde/version.py"))
    _require(match is not None, "opendde/version.py has no __version__", errors)
    if match is None:
        return ""
    version = match.group(1)
    _require(
        STABLE_VERSION_PATTERN.fullmatch(version) is not None,
        f"package version is not a stable X.Y.Z release: {version}",
        errors,
    )
    return version


def _validate_manifest(version: str, errors: list[str]) -> dict[str, Any]:
    try:
        manifest = load_model_manifest(ROOT / "opendde/config/model_manifest.json")
    except (ManifestError, OSError) as exc:
        errors.append(f"invalid model manifest: {exc}")
        return {}

    _require(
        manifest["package"]["version"] == version,
        "manifest package version does not match opendde/version.py",
        errors,
    )
    _require(
        manifest["default_model"] == DEFAULT_MODEL_NAME,
        "manifest default model does not match the runtime registry",
        errors,
    )
    _require(
        {model["name"] for model in manifest["models"]} == set(model_configs),
        "manifest model set does not match the runtime registry",
        errors,
    )
    return manifest


def _validate_changelog(version: str, errors: list[str]) -> str:
    changelog = _read_text("CHANGELOG.md")
    sections = parse_changelog(changelog)
    unreleased = [section for section in sections if section.version == "Unreleased"]
    released = [section for section in sections if section.version != "Unreleased"]
    current = [section for section in released if section.version == version]

    _require(
        len(unreleased) == 1,
        "CHANGELOG must contain exactly one [Unreleased] section",
        errors,
    )
    _require(bool(released), "CHANGELOG has no stable release sections", errors)
    if released:
        _require(
            released[0].version == version,
            "latest CHANGELOG release does not match opendde/version.py",
            errors,
        )
    _require(
        len(current) == 1,
        f"CHANGELOG must contain exactly one release section for {version}",
        errors,
    )

    section = ""
    if len(current) == 1:
        release_date = current[0].date
        _require(
            release_date is not None,
            f"CHANGELOG release {version} has no YYYY-MM-DD date",
            errors,
        )
        if release_date is not None:
            try:
                date.fromisoformat(release_date)
            except ValueError:
                errors.append(f"CHANGELOG release {version} has an invalid date")
        try:
            section = extract_release_section(changelog, version, sections)
        except ValueError as exc:
            errors.append(str(exc))

    expected_unreleased_link = (
        f"[Unreleased]: {REPOSITORY_URL}/compare/v{version}...HEAD"
    )
    _require(
        expected_unreleased_link in changelog,
        "CHANGELOG [Unreleased] comparison link differs from the package version",
        errors,
    )

    if len(current) == 1:
        expected_release_link = f"[{version}]: {REPOSITORY_URL}/releases/tag/v{version}"
        _require(
            expected_release_link in changelog,
            f"CHANGELOG release link for {version} is missing or incorrect",
            errors,
        )
    return section


def _validate_optional_release_notes(
    version: str, changelog_section: str, errors: list[str]
) -> None:
    release_notes_path = ROOT / "docs" / "releases" / f"{version}.md"
    if not release_notes_path.is_file():
        return

    release_notes = release_notes_path.read_text(encoding="utf-8")
    _require(
        release_notes.startswith(f"# OpenDDE {version}\n"),
        "long-form release notes title does not match the package version",
        errors,
    )
    _require(
        not has_placeholder(release_notes),
        "long-form release notes contain a TODO or TBD placeholder",
        errors,
    )
    _require(
        f"docs/releases/{version}.md" in changelog_section,
        "long-form release notes exist but are not linked from CHANGELOG",
        errors,
    )


def _validate_documented_models(
    manifest: dict[str, Any],
    readme: str,
    supported_models: str,
    errors: list[str],
) -> None:
    for model in manifest["models"]:
        model_name = model["name"]
        _require(
            f"`{model_name}`" in supported_models,
            f"supported-models does not document model {model_name}",
            errors,
        )
        _require(
            f"{model['parameter_count']:,}" in supported_models,
            f"supported-models lacks the parameter count for {model_name}",
            errors,
        )
        _require(
            model["data_cutoff"] in supported_models,
            f"supported-models lacks the data cutoff for {model_name}",
            errors,
        )

        for checkpoint in model["checkpoints"]:
            filename = checkpoint["filename"]
            markdown_link = f"[{filename}]({checkpoint_url(filename, manifest)})"
            _require(
                markdown_link in readme,
                f"README lacks the released checkpoint link for {filename}",
                errors,
            )
            _require(
                markdown_link in supported_models,
                f"supported-models lacks the released checkpoint link for {filename}",
                errors,
            )
            _require(
                checkpoint["sha256"] in supported_models,
                f"supported-models lacks the SHA-256 for {filename}",
                errors,
            )


def validate_release(tag: str | None = None) -> tuple[list[str], str]:
    """Return release consistency errors; an empty list means success."""
    errors: list[str] = []
    version = _package_version(errors)
    manifest = _validate_manifest(version, errors)

    if tag is not None:
        _require(tag == f"v{version}", f"tag {tag!r} must be v{version}", errors)

    pyproject = tomllib.loads(_read_text("pyproject.toml"))
    _require(pyproject["project"]["name"] == "opendde", "project name differs", errors)
    _require(
        pyproject["tool"]["setuptools"]["dynamic"]["version"]
        == {"attr": "opendde.version.__version__"},
        "pyproject version is not sourced from opendde.version.__version__",
        errors,
    )
    _require(
        "model_manifest.json"
        in pyproject["tool"]["setuptools"]["package-data"]["opendde.config"],
        "model_manifest.json is not included as package data",
        errors,
    )

    changelog_section = _validate_changelog(version, errors)
    _validate_optional_release_notes(version, changelog_section, errors)

    docker_image = "aurekaresearch/opendde:v1"
    docker_guide = _read_text("docs/docker_installation.md")
    readme = _read_text("README.md")
    supported_models = _read_text("docs/supported_models.md")
    _require(
        docker_image in readme,
        "README lacks the v1 image tag",
        errors,
    )
    _require(
        docker_image in docker_guide,
        "Docker guide lacks the v1 image tag",
        errors,
    )
    if manifest:
        _validate_documented_models(manifest, readme, supported_models, errors)

    return errors, version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Git tag to compare with the package version")
    args = parser.parse_args()

    errors, version = validate_release(args.tag)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    suffix = f" and tag v{version}" if args.tag else ""
    print(f"OpenDDE {version} release metadata is internally consistent{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
