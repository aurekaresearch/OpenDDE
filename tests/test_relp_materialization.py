# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import torch

from opendde.model.modules.embedders import RelativePositionEncoding


def test_lazy_relp_matches_eager_relp_and_projection_bitwise():
    """``materialize`` fills one pre-allocated buffer section by section, so it
    has to stay bit-for-bit identical to the eager concatenated encoding."""
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


def test_lazy_relp_tiles_match_eager_at_inference_scale():
    """Fold-CP materializes row/column tiles instead of the whole encoding, so
    the sliced paths need the same guarantee at realistic r_max/s_max."""
    torch.manual_seed(0)
    n_token = 24
    features = {
        name: torch.randint(0, 4, (n_token,))
        for name in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")
    }
    encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=8)

    eager = encoder.generate_relp(dict(features), lazy=False)["relp"]
    lazy = encoder.generate_relp(dict(features), lazy=True)["relp"]

    assert torch.equal(lazy.materialize(), eager)
    assert torch.equal(lazy[..., 4:12, 6:20], eager[4:12, 6:20])
    assert torch.equal(lazy[..., 4:12, 6:20, 3:40], eager[4:12, 6:20, 3:40])
