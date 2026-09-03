# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gzip
import json
import pickle
import time
from typing import TYPE_CHECKING, Any, Union

# Only import lmdb for type checking (no runtime cost)
if TYPE_CHECKING:
    import lmdb

import hashlib
import os
from pathlib import Path

import torch

from opendde.utils.logger import get_logger
from opendde.utils.torch_utils import map_values_to_list, to_device

logger = get_logger(__name__)

_JSON_FILE_CACHE: dict[str, Any] = {}


def compat_pickle_load(file_obj: Any) -> Any:
    return pickle.load(file_obj)


class LMDBDict:
    """
    A lightweight dict-like wrapper around an LMDB database using SHA1-hashed keys.
    Values are read lazily from disk on access.

    Only record path and configuration at init time; do not open the
    Environment yet. Each process will open its own read-only env on the
    first access (safe under fork-based multiprocessing).

    Args:
        lmdb_path: Path to the LMDB database.
        lock (bool, optional): Whether to use locking. Defaults to False.
    """

    def __init__(self, lmdb_path, lock=False):
        self.path = str(lmdb_path)
        self.lock = lock
        # Per-process cache: pid -> lmdb.Environment
        self._env_by_pid: dict[int, Any] = {}  # Avoid runtime dependency on lmdb.Env

    def _get_env(self) -> "lmdb.Environment":
        """
        Maintain one read-only env per process to avoid sharing an Environment
        across forked processes.
        """
        pid = os.getpid()
        env = self._env_by_pid.get(pid)
        if env is None:
            # Lazy import: only import lmdb when actually needed
            try:
                import lmdb
            except ImportError as e:
                raise ImportError(
                    "LMDB is required to use LMDBDict. Install it with: "
                    "python -m pip install lmdb or uv pip install lmdb"
                ) from e

            env = lmdb.open(
                self.path,
                subdir=False,
                readonly=True,
                lock=self.lock,
                readahead=False,
                meminit=False,
            )
            self._env_by_pid[pid] = env
        return env

    def _hash_key(self, key):
        """Compute SHA1 hash for a key, consistent with the writer script."""
        return hashlib.sha1(str(key).encode("utf-8")).hexdigest().encode("utf-8")

    def __getitem__(self, key):
        hashed_key = self._hash_key(key)
        env = self._get_env()
        with env.begin() as txn:
            value_bytes = txn.get(hashed_key)
            if value_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB")
            # Assume the stored value is a UTF-8 string; fall back to raw bytes
            # to support arbitrary binary payloads.
            try:
                return value_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return value_bytes

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key):
        hashed_key = self._hash_key(key)
        env = self._get_env()
        with env.begin() as txn:
            return txn.get(hashed_key) is not None

    def __len__(self):
        env = self._get_env()
        with env.begin() as txn:
            return txn.stat()["entries"]

    def keys(self):
        # Warning: do not convert keys() of a huge LMDB into a list().
        env = self._get_env()
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, _ in cursor:
                # Note: yields hashed keys, original keys cannot be recovered here.
                yield k

    def close(self):
        for env in self._env_by_pid.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_by_pid.clear()


def load_json_cached(path: Union[str, Path]) -> Any:
    """
    Load a JSON/Parquet/LMDB file with a simple in-process cache.

    - For JSON: Loads entire file into RAM (returns dict).
    - For Parquet: Loads into DataFrame (returns DataFrame).
    - For LMDB: Opens connection handle (returns LMDBDict wrapper).
    """
    path_str = str(Path(path))

    # 1. Cache JSON results per process; LMDB always returns a fresh wrapper.
    if path_str.endswith(".json") and path_str in _JSON_FILE_CACHE:
        logger.info("[OpenDDE IO] load_json_cached cache hit: path=%s", path_str)
        return _JSON_FILE_CACHE[path_str]

    t0 = time.time()

    # 2. Load according to suffix
    if path_str.endswith(".lmdb"):
        # LMDB branch: construct a new LMDBDict instance (not cached in _JSON_FILE_CACHE).
        data = LMDBDict(path_str)

    elif path_str.endswith(".json"):
        # JSON branch: load full JSON into memory and cache it.
        with open(path_str, "r") as f:
            try:
                import orjson

                data = orjson.loads(f.read())
            except ImportError:
                data = json.load(f)
        _JSON_FILE_CACHE[path_str] = data
    elif path_str == ".":
        # Empty JSON file
        data = {}
    else:
        raise ValueError(f"Unsupported file format: {path_str}")

    logger.info(
        "[OpenDDE IO] load_json_cached finished in %.3fs: path=%s",
        time.time() - t0,
        path_str,
    )
    return data


def load_gzip_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load a gzip pickle file.

    Args:
        pkl (Union[str, Path]): A gzip pickle file path.

    Returns:
        Any: The loaded data.
    """
    with gzip.open(pkl, "rb") as f:
        data = compat_pickle_load(f)
    return data


class FloatEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", 2)
        super(FloatEncoder, self).__init__(*args, **kwargs)

    def encode(self, o):
        def float_converter(o):
            if isinstance(o, float):
                return format(o, f".{self.precision}f")
            if isinstance(o, list):
                return [float_converter(i) for i in o]
            if isinstance(o, dict):
                return {k: float_converter(v) for k, v in o.items()}
            return o

        return super(FloatEncoder, self).encode(float_converter(o))


def save_json(data, output_fpath, indent=4):
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)


def save_tensor(data, output_fpath):
    torch.save(to_device(data, device=None), output_fpath)
