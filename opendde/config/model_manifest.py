# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Load, resolve, and verify release-pinned OpenDDE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when the model manifest cannot be read."""


def load_model_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the manifest, defaulting to the copy installed with the package."""
    try:
        if path is None:
            resource = files("opendde.config").joinpath("model_manifest.json")
            contents = resource.read_text(encoding="utf-8")
        else:
            contents = Path(path).read_text(encoding="utf-8")
        return json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in model manifest: {exc}") from exc


def get_model_manifest(
    model_name: str, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return one model entry or raise ``KeyError`` for an unknown model."""
    if manifest is None:
        manifest = load_model_manifest()
    for model in manifest["models"]:
        if model["name"] == model_name:
            return model
    raise KeyError(f"Unknown released model: {model_name}")


def get_default_model_manifest(
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the manifest's explicit default model record."""
    if manifest is None:
        manifest = load_model_manifest()
    return get_model_manifest(manifest["default_model"], manifest)


def get_checkpoint_manifest(
    model_manifest: dict[str, Any], checkpoint_filename: str
) -> dict[str, Any]:
    """Return one checkpoint entry or raise ``KeyError`` if it is absent."""
    for checkpoint in model_manifest["checkpoints"]:
        if checkpoint["filename"] == checkpoint_filename:
            return checkpoint
    model_name = model_manifest.get("name", "<unknown>")
    raise KeyError(
        f"Unknown checkpoint {checkpoint_filename!r} for model {model_name!r}"
    )


def find_checkpoint_manifest(
    checkpoint_filename: str, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a checkpoint record by filename across all released models."""
    if manifest is None:
        manifest = load_model_manifest()
    for model in manifest["models"]:
        for checkpoint in model["checkpoints"]:
            if checkpoint["filename"] == checkpoint_filename:
                return checkpoint
    raise KeyError(f"Unknown released checkpoint: {checkpoint_filename}")


def source_root(manifest: dict[str, Any] | None = None) -> str:
    """Return the immutable ``repository/resolve/revision`` asset root."""
    if manifest is None:
        manifest = load_model_manifest()
    source = manifest["source"]
    return f"{source['repository']}/resolve/{source['revision']}"


def checkpoint_url(
    checkpoint_filename: str,
    manifest: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
) -> str:
    """Resolve one released checkpoint URL from the manifest source identity."""
    if manifest is None:
        manifest = load_model_manifest()
    if model_name is None:
        find_checkpoint_manifest(checkpoint_filename, manifest)
    else:
        get_checkpoint_manifest(
            get_model_manifest(model_name, manifest), checkpoint_filename
        )
    return f"{source_root(manifest)}/{checkpoint_filename}"


def _checkpoint_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


def verify_checkpoint_file(
    path: str | Path,
    checkpoint_filename: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one released checkpoint against its manifest size and SHA-256."""
    if manifest is None:
        manifest = load_model_manifest()
    checkpoint_path = Path(path)
    filename = checkpoint_filename or checkpoint_path.name
    expected = find_checkpoint_manifest(filename, manifest)

    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected["size_bytes"]:
        raise ValueError(
            f"Checkpoint size mismatch for {filename}: expected "
            f"{expected['size_bytes']} bytes, got {actual_size}"
        )
    actual_sha256 = _checkpoint_sha256(checkpoint_path)
    if actual_sha256 != expected["sha256"]:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {filename}: expected "
            f"{expected['sha256']}, got {actual_sha256}"
        )
    return expected


def _add_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to the copy installed with OpenDDE",
    )


def main(argv: list[str] | None = None) -> int:
    """Run lightweight manifest resolution and checkpoint verification commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_model_parser = subparsers.add_parser(
        "resolve-model",
        help="print model name, default checkpoint, URL, and source root as TSV",
    )
    resolve_model_parser.add_argument(
        "--model", help="Released model name; defaults to manifest.default_model"
    )
    _add_manifest_argument(resolve_model_parser)

    checkpoint_url_parser = subparsers.add_parser(
        "checkpoint-url", help="print the immutable URL for a released checkpoint"
    )
    checkpoint_url_parser.add_argument("checkpoint", help="Manifest checkpoint name")
    checkpoint_url_parser.add_argument(
        "--model", help="Require the checkpoint to belong to this released model"
    )
    _add_manifest_argument(checkpoint_url_parser)

    verify_parser = subparsers.add_parser(
        "verify-checkpoint",
        help="verify a released checkpoint's byte size and SHA-256",
    )
    verify_parser.add_argument("path", type=Path, help="Local checkpoint path")
    verify_parser.add_argument(
        "--checkpoint",
        help="Manifest filename when the local path uses a temporary name",
    )
    _add_manifest_argument(verify_parser)
    args = parser.parse_args(argv)

    try:
        manifest = load_model_manifest(args.manifest)
        if args.command == "resolve-model":
            model_name = args.model or manifest["default_model"]
            model = get_model_manifest(model_name, manifest)
            checkpoint = model["default_checkpoint"]
            print(
                "\t".join(
                    (
                        model_name,
                        checkpoint,
                        checkpoint_url(checkpoint, manifest),
                        source_root(manifest),
                    )
                )
            )
        elif args.command == "checkpoint-url":
            print(checkpoint_url(args.checkpoint, manifest, model_name=args.model))
        else:
            expected = verify_checkpoint_file(
                args.path,
                checkpoint_filename=args.checkpoint,
                manifest=manifest,
            )
            filename = args.checkpoint or args.path.name
            print(
                f"Verified {filename}: {expected['size_bytes']} bytes, "
                f"SHA-256 {expected['sha256']}"
            )
    except (KeyError, ManifestError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
