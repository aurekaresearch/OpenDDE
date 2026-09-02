import ast
from pathlib import Path

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
