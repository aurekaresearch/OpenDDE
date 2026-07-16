# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Download utilities for OpenDDE."""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from os.path import exists as opexists
from typing import Any, Callable
from urllib.parse import urlsplit

import torch
import requests

from opendde.config.dependency_url import (
    CHECKPOINT_FILES,
    MANAGED_ASSETS,
    URL,
    ManagedAsset,
)

logger = logging.getLogger(__name__)
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT = (30, 300)


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """
    Callback for tracking download progress.

    Args:
        block_num: Current block number.
        block_size: Size of each block in bytes.
        total_size: Total file size in bytes.
    """
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"\rDownloaded {downloaded} bytes", end="", flush=True)
        return

    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    status = f"\r[{bar}] {percent:.1f}%"
    print(status, end="", flush=True)

    if downloaded >= total_size:
        print()


def _decompress_zst(zst_path: str, output_path: str, source_url: str) -> None:
    """Decompress a .zst archive using python-zstandard or the zstd binary."""
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None

    if zstd is not None:
        try:
            with (
                open(zst_path, "rb") as compressed,
                open(output_path, "wb") as output,
            ):
                zstd.ZstdDecompressor().copy_stream(compressed, output)
            return
        except Exception as e:
            if opexists(output_path):
                os.remove(output_path)
            raise RuntimeError(
                f"Failed to decompress .zst archive downloaded from {source_url} "
                f"to {output_path}: {e}"
            ) from e

    zstd_binary = shutil.which("zstd")
    if zstd_binary is None:
        raise RuntimeError(
            f"Downloaded {source_url} is a .zst archive. Install the `zstd` "
            "command or the optional Python `zstandard` package, or download "
            f"and decompress it manually to {output_path}."
        )

    try:
        subprocess.run(
            [zstd_binary, "-d", "-f", "-o", output_path, zst_path], check=True
        )
    except Exception as e:
        if opexists(output_path):
            os.remove(output_path)
        raise RuntimeError(
            f"Failed to decompress .zst archive downloaded from {source_url} "
            f"to {output_path}: {e}"
        ) from e


def _should_decompress_zst(tos_url: str, checkpoint_path: str) -> bool:
    return urlsplit(tos_url).path.endswith(".zst") and not checkpoint_path.endswith(
        ".zst"
    )


def _retrieve_url(source_url: str, destination: str) -> None:
    """Retrieve a URL with bounded retries for interrupted transfers."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with requests.get(
                source_url,
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT,
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", -1))
                downloaded = 0
                with open(destination, "wb") as destination_file:
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        destination_file.write(chunk)
                        downloaded += len(chunk)
                        progress_callback(downloaded, 1, total_size)
                if total_size >= 0 and downloaded != total_size:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"retrieval incomplete: got {downloaded} out of "
                        f"{total_size} bytes"
                    )
                if total_size < 0:
                    print()
            return
        except requests.RequestException as error:
            try:
                os.remove(destination)
            except FileNotFoundError:
                pass
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            logger.warning(
                "Download from %s was interrupted (%s); retrying (%d/%d).",
                source_url,
                error,
                attempt + 1,
                _DOWNLOAD_ATTEMPTS,
            )


@contextmanager
def _temporary_download_path(checkpoint_path: str, *, suffix: str) -> Iterator[str]:
    """Create and clean up a staging file beside the final destination."""
    destination_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    destination_name = os.path.basename(checkpoint_path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination_name}.", suffix=suffix, dir=destination_dir
    )
    os.close(fd)
    try:
        yield temporary_path
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def download_from_url(
    tos_url: str,
    checkpoint_path: str,
    check_weight: bool = True,
    *,
    validator: Callable[[str], None] | None = None,
) -> None:
    """
    Download a file from URL and optionally verify if it's a valid checkpoint.

    Args:
        tos_url: URL to download from.
        checkpoint_path: Local path to save the downloaded file.
        check_weight: Whether to load the staged file as a PyTorch checkpoint.
        validator: Optional validation callback invoked before atomic replacement.

    Raises:
        RuntimeError: If download or verification fails.
    """
    with _temporary_download_path(checkpoint_path, suffix=".part") as staged_path:
        if _should_decompress_zst(tos_url, checkpoint_path):
            with _temporary_download_path(
                checkpoint_path, suffix=".download.zst"
            ) as compressed_path:
                _retrieve_url(tos_url, compressed_path)
                _decompress_zst(compressed_path, staged_path, tos_url)
        else:
            _retrieve_url(tos_url, staged_path)

        try:
            if check_weight:
                ckpt = torch.load(staged_path, map_location="cpu", weights_only=False)
                del ckpt
            if validator is not None:
                validator(staged_path)
        except Exception as e:
            if check_weight and validator is None:
                raise RuntimeError(
                    f"Download model checkpoint failed: {e}. Please download "
                    f"manually with: wget {tos_url} -O {checkpoint_path}"
                ) from e
            raise RuntimeError(
                f"Downloaded asset from {tos_url} failed validation: {e}"
            ) from e

        os.replace(staged_path, checkpoint_path)


def resolve_checkpoint_path(configs: Any) -> str:
    """
    Resolve the checkpoint path from configuration.

    Args:
        configs: Configuration object.

    Returns:
        Full path to the checkpoint file.
    """
    checkpoint_path = configs.get("load_checkpoint_path", "")
    if checkpoint_path:
        return checkpoint_path
    checkpoint_file = CHECKPOINT_FILES.get(
        configs.model_name, f"{configs.model_name}.pt"
    )
    return os.path.join(configs.load_checkpoint_dir, checkpoint_file)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_managed_asset(path: str, asset: ManagedAsset) -> None:
    actual_size = os.path.getsize(path)
    if actual_size != asset.size:
        raise ValueError(f"expected {asset.size} bytes, got {actual_size} bytes")
    actual_sha256 = _sha256(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(f"expected SHA256 {asset.sha256}, got {actual_sha256}")


def _ensure_managed_asset(asset_name: str, destination: str) -> None:
    """Ensure a release-managed asset has the expected identity."""
    asset = MANAGED_ASSETS[asset_name]
    if opexists(destination) and os.path.getsize(destination) == asset.size:
        return

    if opexists(destination):
        logger.warning(
            "Managed asset at %s has an unexpected size; downloading a verified "
            "replacement.",
            destination,
        )
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    source_url = URL[asset_name]
    if os.path.basename(urlsplit(source_url).path) != os.path.basename(destination):
        raise ValueError(
            f"Managed asset file name mismatch: {source_url} and {destination}."
        )
    logger.info(
        "Downloading managed asset from\n %s...\n to %s", source_url, destination
    )
    download_from_url(
        source_url,
        destination,
        check_weight=False,
        validator=lambda path: _validate_managed_asset(path, asset),
    )


def download_inference_cache(configs: Any) -> None:
    """
    Download necessary data and model checkpoints for inference.

    Args:
        configs: Configuration object containing paths and model names.
    """
    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
    ):
        _ensure_managed_asset(cache_name, configs["data"][cache_name])

    if configs.use_template:
        for cache_name in (
            "obsolete_pdbs_path",
            "release_dates_path",
        ):
            _ensure_managed_asset(
                cache_name,
                configs["data"]["template"][cache_name],
            )

    checkpoint_path = resolve_checkpoint_path(configs)
    explicit_checkpoint = bool(configs.get("load_checkpoint_path", ""))

    if explicit_checkpoint:
        if not opexists(checkpoint_path):
            raise FileNotFoundError(
                f"Given checkpoint path not exist [{checkpoint_path}]"
            )
        return

    _ensure_managed_asset(configs.model_name, checkpoint_path)
