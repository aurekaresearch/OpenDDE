# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from unittest import mock

import torch

from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.model.modules.structural_tokens import StructuralTokenExpander


def test_structural_pair_features_distinguish_twin_and_backbone_direction():
    expander = StructuralTokenExpander(c_s=4, c_z=3, c_s_inputs=5)
    role = torch.tensor(
        [
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["protein_sc"],
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_base"],
        ]
    )
    parent = torch.tensor([0, 0, 1, 2, 2])
    input_feature_dict = {
        "subtoken_role_id": role,
        "parent_residue_idx": parent,
        "asym_id": torch.tensor([0, 0, 1]),
        "residue_index": torch.tensor([10, 11, 1]),
        "prev_parent_residue_idx": torch.tensor([-1, -1, 0, -1, -1]),
        "next_parent_residue_idx": torch.tensor([1, 1, -1, -1, -1]),
        "structural_polymer_type": torch.tensor([1, 1, 1, 2, 2]),
    }

    pair_features = expander.build_structural_pair_features(
        input_feature_dict=input_feature_dict,
        role=role,
        parent=parent,
    )

    assert pair_features["same_parent_residue"][0, 1]
    assert pair_features["same_residue_twin"][0, 1]
    assert pair_features["same_residue_twin"][1, 0]
    assert pair_features["same_residue_twin"][3, 4]
    assert pair_features["same_residue_twin"][4, 3]
    assert not pair_features["same_residue_twin"][0, 0]

    assert pair_features["next_bb_chain"][0, 2]
    assert pair_features["prev_bb_chain"][2, 0]
    assert not pair_features["next_bb_chain"][2, 0]
    assert not pair_features["prev_bb_chain"][0, 2]

    assert pair_features["role_pair_type"][0, 2].item() == 0
    assert pair_features["role_pair_type"][0, 1].item() == 1
    assert pair_features["role_pair_type"][1, 0].item() == 2
    assert pair_features["role_pair_type"][3, 4].item() == 4
    assert pair_features["role_pair_type"][4, 3].item() == 5
    assert pair_features["role_pair_type"][4, 4].item() == 6


def test_chunked_forward_matches_unchunked_forward():
    torch.manual_seed(0)
    base_expander = StructuralTokenExpander(
        c_s=4,
        c_z=3,
        c_s_inputs=5,
        init_mode="scratch",
        pair_projection_mode="factorized",
        pair_chunk_size=None,
    )
    chunked_expander = StructuralTokenExpander(
        c_s=4,
        c_z=3,
        c_s_inputs=5,
        init_mode="scratch",
        pair_projection_mode="factorized",
        pair_chunk_size=2,
    )
    chunked_expander.load_state_dict(base_expander.state_dict())

    role = torch.tensor(
        [
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["protein_sc"],
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_base"],
        ]
    )
    parent = torch.tensor([0, 0, 1, 2, 2])
    input_feature_dict = {
        "subtoken_role_id": role,
        "parent_residue_idx": parent,
        "asym_id": torch.tensor([0, 0, 1]),
        "residue_index": torch.tensor([10, 11, 1]),
        "prev_parent_residue_idx": torch.tensor([-1, -1, 0, -1, -1]),
        "next_parent_residue_idx": torch.tensor([1, 1, -1, -1, -1]),
        "structural_polymer_type": torch.tensor([1, 1, 1, 2, 2]),
    }
    s_inputs_res = torch.randn(3, 5)
    s_res = torch.randn(3, 4)
    z_res = torch.randn(3, 3, 3)

    unchunked = base_expander(
        input_feature_dict=input_feature_dict,
        s_inputs_res=s_inputs_res,
        s_res=s_res,
        z_res=z_res,
    )
    chunked = chunked_expander(
        input_feature_dict=input_feature_dict,
        s_inputs_res=s_inputs_res,
        s_res=s_res,
        z_res=z_res,
    )

    assert torch.allclose(chunked[0], unchunked[0])
    assert torch.allclose(chunked[1], unchunked[1])
    assert torch.allclose(chunked[2], unchunked[2])
    assert torch.allclose(
        chunked[3]["structural_pair_attn_bias"],
        unchunked[3]["structural_pair_attn_bias"],
    )


def test_chunked_inference_writes_final_outputs_without_cat():
    expander = StructuralTokenExpander(
        c_s=4,
        c_z=3,
        c_s_inputs=5,
        init_mode="scratch",
        pair_projection_mode="factorized",
        pair_chunk_size=2,
    )
    role = torch.tensor(
        [
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["protein_sc"],
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
        ]
    )
    parent = torch.tensor([0, 0, 1])
    input_feature_dict = {
        "subtoken_role_id": role,
        "parent_residue_idx": parent,
        "asym_id": torch.tensor([0, 0]),
        "residue_index": torch.tensor([10, 11]),
        "prev_parent_residue_idx": torch.tensor([-1, -1, 0]),
        "next_parent_residue_idx": torch.tensor([1, 1, -1]),
        "structural_polymer_type": torch.tensor([1, 1, 1]),
    }

    with (
        torch.no_grad(),
        mock.patch(
            "opendde.model.modules.structural_tokens.torch.cat",
            side_effect=AssertionError(
                "inference must not retain and concatenate chunks"
            ),
        ),
    ):
        _, _, z_struct, pair_features = expander(
            input_feature_dict=input_feature_dict,
            s_inputs_res=torch.randn(2, 5),
            s_res=torch.randn(2, 4),
            z_res=torch.randn(2, 2, 3),
        )

    assert z_struct.shape == (3, 3, 3)
    assert pair_features["structural_pair_attn_bias"].shape == (3, 3)
