# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gzip
import json
import pickle

import torch

from opendde.utils.file_io import LMDBDict, load_gzip_pickle, save_json


def test_load_gzip_pickle_roundtrips_lmdbdict(tmp_path):
    """load_gzip_pickle restores a gzip-pickled object as-is.

    The legacy package-name remapping shim was removed in the inference-only
    build, so this only exercises a plain round-trip.
    """
    pkl_path = tmp_path / "data.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(LMDBDict("some.lmdb"), f)

    loaded = load_gzip_pickle(pkl_path)

    assert isinstance(loaded, LMDBDict)
    assert loaded.path == "some.lmdb"


def test_save_json_does_not_mutate_nested_caller_data(tmp_path):
    tensor = torch.tensor([1.25])
    data = {"nested": {"score": tensor}, "label": "sample"}
    output = tmp_path / "result.json"

    save_json(data, output)

    assert data["nested"]["score"] is tensor
    assert json.loads(output.read_text()) == {
        "nested": {"score": [1.25]},
        "label": "sample",
    }
