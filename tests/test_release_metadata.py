# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import importlib.util
import subprocess
import sys
from pathlib import Path

from opendde import __version__
from opendde.config.model_manifest import load_model_manifest

ROOT = Path(__file__).resolve().parents[1]


def load_repository_script(module_name: str):
    scripts_dir = ROOT / "scripts"
    spec = importlib.util.spec_from_file_location(
        module_name, scripts_dir / f"{module_name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(scripts_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_source_distribution_excludes_repository_tests():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include CHANGELOG.md" in manifest
    assert "recursive-include docs/releases *.md" not in manifest
    assert "prune docs/releases" in manifest
    assert "prune tests" in manifest
    assert "exclude MANIFEST.in" in manifest


def test_current_changelog_renders_the_github_release_body():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_release_notes.py"),
            "--version",
            __version__,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "### Fixed" in result.stdout
    assert "**Full Changelog**:" not in result.stdout
    assert "compare/v1.0.1" not in result.stdout
    assert "## [Unreleased]" not in result.stdout


def test_release_parser_ignores_fenced_code():
    release_notes = load_repository_script("render_release_notes")
    changelog = """# Changelog

## [Unreleased]

## [1.2.3] - 2026-01-02

### Fixed

- A real change.

```markdown
## [1.2.3]
[fake]: https://example.invalid
TODO: this is a literal example.
```

~~~text
## [9.9.9]
~~~

[Unreleased]: https://example.test/compare/v1.2.3...HEAD
[1.2.3]: https://example.test/releases/tag/v1.2.3
"""

    rendered = release_notes.render_release_notes(changelog, "1.2.3")

    assert "A real change" in rendered
    assert "TODO: this is a literal example" in rendered
    assert "[1.2.3]: https://example.test" not in rendered
    assert len(release_notes.parse_changelog(changelog)) == 2


def test_long_form_release_document_is_optional(monkeypatch, tmp_path):
    release_check = load_repository_script("check_release")
    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    errors = []

    release_check._validate_optional_release_notes(
        "9.9.9", "### Fixed\n\n- A complete change.", errors
    )

    assert errors == []


def test_existing_long_form_release_document_is_validated(monkeypatch, tmp_path):
    release_check = load_repository_script("check_release")
    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    notes_path = tmp_path / "docs" / "releases" / "9.9.9.md"
    notes_path.parent.mkdir(parents=True)
    notes_path.write_text("# Wrong title\n\nTODO: finish this.\n", encoding="utf-8")
    errors = []

    release_check._validate_optional_release_notes(
        "9.9.9", "### Fixed\n\n- A complete change.", errors
    )

    assert "long-form release notes title does not match the package version" in errors
    assert "long-form release notes contain a TODO or TBD placeholder" in errors
    assert "long-form release notes exist but are not linked from CHANGELOG" in errors


def test_release_validator_accepts_current_version():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_release.py"),
            "--tag",
            f"v{__version__}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        f"OpenDDE {__version__} release metadata is internally consistent"
        in result.stdout
    )


def test_release_validator_checks_documented_model_metadata():
    release_check = load_repository_script("check_release")
    manifest = load_model_manifest()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    supported_models = (ROOT / "docs/supported_models.md").read_text(encoding="utf-8")
    tampered = supported_models.replace("655,791,538", "1").replace(
        "2021-09-30", "2099-01-01"
    )
    errors = []

    release_check._validate_documented_models(manifest, readme, tampered, errors)

    assert any("lacks the parameter count" in error for error in errors)
    assert any("lacks the data cutoff" in error for error in errors)


def test_docker_build_script_is_a_manual_local_helper():
    script = (ROOT / "scripts/build_docker_image.sh").read_text(encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "scripts/build_docker_image.sh",
            "--dry-run",
            "--tag",
            "local/opendde:test",
            "--platform",
            "linux/arm64",
            "--pull",
            "--no-cache",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "docker build" in result.stdout
    assert "local/opendde:test" in result.stdout
    assert "--platform linux/arm64" in result.stdout
    assert "--pull" in result.stdout
    assert "--no-cache" in result.stdout
    assert "git -C" not in script
    assert "docker push" not in script
    assert "VCS_REF" not in script


def test_docker_build_script_rejects_publish_and_passthrough_options():
    for option in ("--push", "--allow-dirty", "--"):
        result = subprocess.run(
            ["bash", "scripts/build_docker_image.sh", "--dry-run", option],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "unknown option" in result.stderr
        assert "docker build" not in result.stdout


def test_docker_context_excludes_local_credentials():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for sensitive_path in (
        ".env",
        ".netrc",
        ".pypirc",
        ".aws",
        ".ssh",
        ".streamlit",
        ".abstra",
        "*.pem",
        "*.key",
    ):
        assert sensitive_path in dockerignore


def test_normal_ci_does_not_build_or_publish_distributions():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m build" not in workflow
    assert "uv build" not in workflow
    assert "twine" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "upload-artifact" not in workflow
    assert workflow.count(".venv/bin/ruff check .") == 1
    assert "  lint:\n" in workflow
    assert "  tests:\n" in workflow


def test_ci_does_not_build_or_publish_docker_images():
    workflows = "\n".join(
        (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
        for filename in ("ci.yml", "release.yml")
    )

    for forbidden in (
        "docker build",
        "docker push",
        "docker/build-push-action",
        "scripts/build_docker_image.sh",
    ):
        assert forbidden not in workflows


def test_release_workflow_is_tag_only_and_builds_once():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert 'tags:\n      - "v*.*.*"' in workflow
    assert "workflow_dispatch:" not in workflow
    assert "python -m build" not in workflow
    assert workflow.count("uv build") == 1
    assert workflow.count("gh-action-pypi-publish") == 1
    assert workflow.count("actions/upload-artifact@v7") == 1
    assert workflow.count("actions/download-artifact@v8") == 2
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert 'scripts/check_release.py --tag "$GITHUB_REF_NAME"' in workflow
    assert "scripts/render_release_notes.py" in workflow
    assert 'cp "docs/releases/${version}.md"' not in workflow
    assert "Verify built distributions in clean CPU environments" in workflow
    assert 'tar -tzf "${sdists[0]}"' in workflow
    assert "Source archive does not contain CHANGELOG.md" in workflow
    assert "Source archive unexpectedly contains long-form release notes" in workflow
    assert '--torch-backend cpu "${artifact}"' in workflow
    assert "model_manifest.json" in workflow
