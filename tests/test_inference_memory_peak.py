import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from opendde.model.modules.embedders import RelativePositionEncoding
from opendde.model.opendde import OpenDDE


def test_lazy_relp_matches_eager_relp_and_projection_bitwise():
    encoder = RelativePositionEncoding(r_max=2, s_max=1, c_z=3)
    features = {
        "asym_id": torch.tensor([0, 0, 1, 1]),
        "residue_index": torch.tensor([0, 1, 0, 1]),
        "entity_id": torch.tensor([0, 0, 1, 1]),
        "token_index": torch.tensor([0, 1, 0, 1]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
    }

    eager = encoder.generate_relp(dict(features), lazy=False)["relp"]
    lazy = encoder.generate_relp(dict(features), lazy=True)["relp"]

    assert torch.equal(lazy.materialize(), eager)
    assert torch.equal(encoder(lazy), encoder(eager))


def test_large_pairformer_attention_chunk_is_bounded():
    bound = OpenDDE._bound_pairformer_chunk_size

    assert bound(200, None) is None
    assert bound(390, None) is None
    assert bound(1024, None) == 256
    assert bound(1536, 512) == 128
    assert bound(1952, 256) == 64
    assert bound(2340, 128) == 64
    assert bound(3000, 32) == 32


def test_opendde_class_does_not_silently_override_duplicate_methods():
    source_path = Path(__file__).parents[1] / "opendde/model/opendde.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenDDE"
    )
    methods = [
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert len(methods) == len(set(methods))


class _ChunkPolicyStub:
    """Minimal stand-in exposing the config fields chunk resolution reads."""

    configs = SimpleNamespace(
        infer_setting=SimpleNamespace(
            chunk_size=256,
            chunk_size_thresholds={"1024": -1, "1536": 512, "2048": 256, "2560": 128},
        )
    )

    _get_dynamic_chunk_size = OpenDDE._get_dynamic_chunk_size
    _bound_pairformer_chunk_size = staticmethod(OpenDDE._bound_pairformer_chunk_size)
    _resolve_pairformer_chunk_size = OpenDDE._resolve_pairformer_chunk_size


def test_explicit_fixed_chunk_size_is_not_overridden_by_the_bound():
    model = _ChunkPolicyStub()

    assert (
        model._resolve_pairformer_chunk_size(2048, 256, dynamic_chunk_size=False) == 256
    )
    assert (
        model._resolve_pairformer_chunk_size(3000, None, dynamic_chunk_size=False)
        is None
    )


def test_dynamic_chunk_size_is_bounded_by_the_score_budget():
    model = _ChunkPolicyStub()

    # Threshold table says unchunked at N=1024; the budget bounds it to 256.
    assert model._resolve_pairformer_chunk_size(1024, 4, dynamic_chunk_size=True) == 256
    # Threshold table says 512 at N=1536; the budget bounds it to 128.
    assert model._resolve_pairformer_chunk_size(1536, 4, dynamic_chunk_size=True) == 128
    # Small inputs stay unchunked.
    assert model._resolve_pairformer_chunk_size(200, 4, dynamic_chunk_size=True) is None
