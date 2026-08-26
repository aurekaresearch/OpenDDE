# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from types import SimpleNamespace

import pytest
import torch

from opendde.model.modules.confidence import ConfidenceHead
from opendde.model.modules.diffusion import DiffusionConditioning, DiffusionModule
from opendde.model.modules.pairformer import TemplateEmbedder
from opendde.model.opendde import OpenDDE, update_input_feature_dict
from opendde.model.seed_batch import (
    select_seed_batch_features,
    stack_seed_batch_features,
)


def _seed_features(ref_offset: float = 0.0) -> dict[str, torch.Tensor]:
    n_token = 3
    n_atom = 5
    n_msa = 4
    n_template = 2
    return {
        "restype": torch.zeros(n_token, 32),
        "profile": torch.zeros(n_token, 32),
        "deletion_mean": torch.zeros(n_token),
        "msa": torch.arange(n_msa * n_token).reshape(n_msa, n_token),
        "msa_mask": torch.ones(n_msa, n_token),
        "has_deletion": torch.zeros(n_msa, n_token),
        "deletion_value": torch.zeros(n_msa, n_token),
        "ref_pos": torch.full((n_atom, 3), ref_offset),
        "ref_charge": torch.zeros(n_atom),
        "ref_mask": torch.ones(n_atom),
        "ref_element": torch.zeros(n_atom, 128),
        "ref_atom_name_chars": torch.zeros(n_atom, 4, 64),
        "ref_space_uid": torch.arange(n_atom),
        "asym_id": torch.zeros(n_token, dtype=torch.long),
        "template_aatype": torch.zeros(n_template, n_token, dtype=torch.long),
        "template_distogram": torch.zeros(n_template, n_token, n_token, 39),
        "template_pseudo_beta_mask": torch.ones(n_template, n_token, n_token),
        "template_backbone_frame_mask": torch.ones(n_template, n_token, n_token),
        "template_unit_vector": torch.zeros(n_template, n_token, n_token, 3),
    }


def test_seed_batch_stacks_seed_features_and_shares_templates():
    features = [_seed_features(1.0), _seed_features(2.0)]

    batched = stack_seed_batch_features(features, [101, 202])

    assert batched["ref_pos"].shape == (2, 5, 3)
    assert batched["msa"].shape == (2, 4, 3)
    assert batched["inference_seed"].tolist() == [101, 202]
    assert batched["template_distogram"] is features[0]["template_distogram"]
    assert batched["template_aatype"] is features[0]["template_aatype"]
    assert batched["asym_id"] is features[0]["asym_id"]


def test_seed_batch_stacks_templates_only_when_they_differ():
    features = [_seed_features(), _seed_features()]
    features[1]["template_distogram"][0, 0, 0, 0] = 1

    batched = stack_seed_batch_features(features, [101, 202])

    assert batched["template_distogram"].shape == (2, 2, 3, 3, 39)
    assert batched["template_aatype"].shape == (2, 3)


def test_seed_batch_rejects_different_topology():
    features = [_seed_features(), _seed_features()]
    features[1]["asym_id"][0] = 1

    with pytest.raises(ValueError, match="asym_id"):
        stack_seed_batch_features(features, [101, 202])


def test_singleton_seed_batch_preserves_scalar_feature_shapes():
    features = _seed_features()

    selected = stack_seed_batch_features([features], [101])

    assert selected["ref_pos"].shape == features["ref_pos"].shape
    assert selected["inference_seed"].shape == torch.Size([])


def test_runner_singleton_seed_batch_keeps_scalar_model_path():
    from runner.inference import InferenceRunner

    captured = []
    runner = object.__new__(InferenceRunner)

    def predict(data, **_kwargs):
        captured.append(data["input_feature_dict"])
        return {"coordinate": torch.zeros(1, 1, 3)}

    runner.predict = predict
    predictions = runner.predict_seed_batch(
        [{"sample_name": "sample", "input_feature_dict": _seed_features()}],
        [101],
        msa_generators=[torch.Generator().manual_seed(101)],
    )

    assert len(predictions) == 1
    assert captured[0]["ref_pos"].shape == (5, 3)
    assert captured[0]["inference_seed"].shape == torch.Size([])


def test_shared_template_features_broadcast_over_seed_batch():
    batch_size = 2
    n_token = 3
    module = TemplateEmbedder(n_blocks=0, c=4, c_z=3)
    features = _seed_features()
    z = torch.zeros(batch_size, n_token, n_token, 3)
    pair_mask = torch.ones(batch_size, n_token, n_token)
    multichain_mask = torch.ones(n_token, n_token)

    output = module.single_template_forward(
        template_id=0,
        input_feature_dict=features,
        z=z,
        pair_mask=pair_mask,
        multichain_mask=multichain_mask,
    )

    assert output.shape == (batch_size, n_token, n_token, 4)


def test_shared_relative_positions_broadcast_over_seed_batch():
    batch_size = 2
    n_token = 3
    c_z = 4
    module = DiffusionConditioning(
        c_z=c_z,
        c_s=4,
        c_s_inputs=5,
        c_noise_embedding=6,
    )
    relp = torch.zeros(n_token, n_token, module.relpe.linear_no_bias.in_features)
    z_trunk = torch.zeros(batch_size, n_token, n_token, c_z)

    pair_z = module.prepare_cache(relp_feature=relp, z_trunk=z_trunk)

    assert pair_z.shape == (batch_size, n_token, n_token, c_z)


def test_seed_batched_diffusion_and_confidence_match_scalar_model_stages():
    torch.manual_seed(9)
    batch_size = 2
    n_sample = 2
    n_atom = 7
    n_token = 3
    c_s = 16
    c_z = 4
    c_s_inputs = 5
    atom_to_token_idx = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    seed_feature_names = {
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_atom_name_chars",
        "ref_element",
        "ref_space_uid",
    }
    diffusion_module = DiffusionModule(
        c_atom=8,
        c_atompair=4,
        c_token=8,
        c_s=c_s,
        c_z=c_z,
        c_s_inputs=c_s_inputs,
        atom_encoder={"n_blocks": 1, "n_heads": 1},
        transformer={"n_blocks": 1, "n_heads": 1},
        atom_decoder={"n_blocks": 1, "n_heads": 1},
    ).eval()
    confidence_head = ConfidenceHead(
        n_blocks=1,
        c_s=c_s,
        c_z=c_z,
        c_s_inputs=c_s_inputs,
        b_pae=6,
        b_pde=6,
        b_plddt=5,
        max_atoms_per_token=4,
    ).eval()
    input_features = {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": torch.randn(batch_size, n_atom, 3),
        "ref_charge": torch.randint(-1, 2, (batch_size, n_atom)),
        "ref_mask": torch.ones(batch_size, n_atom),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.randint(0, 64, (batch_size, n_atom, 4)), 64
        ).float(),
        "ref_element": torch.nn.functional.one_hot(
            torch.randint(0, 10, (batch_size, n_atom)), 128
        ).float(),
        "ref_space_uid": atom_to_token_idx.expand(batch_size, -1).clone(),
        "relp": torch.randn(
            n_token,
            n_token,
            diffusion_module.diffusion_conditioning.relpe.linear_no_bias.in_features,
        ),
    }
    confidence_features = {
        "distogram_rep_atom_mask": torch.tensor(
            [1, 0, 1, 0, 0, 1, 0], dtype=torch.bool
        ),
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": torch.tensor([0, 1, 0, 1, 2, 0, 1]),
    }
    s_inputs = torch.randn(batch_size, n_token, c_s_inputs)
    s_trunk = torch.randn(batch_size, n_token, c_s)
    z_trunk = torch.randn(batch_size, n_token, n_token, c_z)
    x_noisy = torch.randn(batch_size, n_sample, n_atom, 3)
    noise_level = torch.rand(batch_size, n_sample) + 1

    with torch.no_grad():
        batched_coordinate = diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=noise_level,
            input_feature_dict=update_input_feature_dict(dict(input_features)),
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=None,
            p_lm=None,
            c_l=None,
            inplace_safe=False,
        )
        batched_confidence = confidence_head(
            input_feature_dict=confidence_features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_mask=None,
            x_pred_coords=batched_coordinate,
            triangle_attention="torch",
            triangle_multiplicative="torch",
            inplace_safe=False,
        )
        scalar_results = []
        for seed_idx in range(batch_size):
            scalar_features = {
                name: value[seed_idx] if name in seed_feature_names else value
                for name, value in input_features.items()
            }
            scalar_coordinate = diffusion_module(
                x_noisy=x_noisy[seed_idx],
                t_hat_noise_level=noise_level[seed_idx],
                input_feature_dict=update_input_feature_dict(scalar_features),
                s_inputs=s_inputs[seed_idx],
                s_trunk=s_trunk[seed_idx],
                z_trunk=z_trunk[seed_idx],
                pair_z=None,
                p_lm=None,
                c_l=None,
                inplace_safe=False,
            )
            scalar_confidence = confidence_head(
                input_feature_dict=confidence_features,
                s_inputs=s_inputs[seed_idx],
                s_trunk=s_trunk[seed_idx],
                z_trunk=z_trunk[seed_idx],
                pair_mask=None,
                x_pred_coords=scalar_coordinate,
                triangle_attention="torch",
                triangle_multiplicative="torch",
                inplace_safe=False,
            )
            scalar_results.append((scalar_coordinate, scalar_confidence))

    assert batched_coordinate.shape == (batch_size, n_sample, n_atom, 3)
    torch.testing.assert_close(
        batched_coordinate,
        torch.stack([result[0] for result in scalar_results]),
        rtol=2e-5,
        atol=2e-5,
    )
    for output_idx, batched_output in enumerate(batched_confidence):
        torch.testing.assert_close(
            batched_output,
            torch.stack([result[1][output_idx] for result in scalar_results]),
            rtol=2e-5,
            atol=2e-5,
        )


def test_seed_batch_postprocessing_splits_seed_before_sample():
    batch_size = 2
    n_sample = 3
    n_atom = 5
    n_token = 3
    calls = []

    def postprocess(**kwargs):
        seed = int(kwargs["input_feature_dict"]["inference_seed"])
        kwargs["pred_dict"]["summary_confidence"] = [seed]
        calls.append((seed, kwargs["pair_z"].shape))

    model = SimpleNamespace(run_post_confidence_outputs_stage=postprocess)
    input_features = {
        "inference_seed": torch.tensor([101, 202]),
        "ref_pos": torch.arange(batch_size * n_atom * 3).reshape(batch_size, n_atom, 3),
        "is_ligand": torch.zeros(n_atom),
        "template_aatype": torch.zeros(batch_size, n_token, dtype=torch.long),
    }
    pred_dict = {
        "coordinate": torch.zeros(batch_size, n_sample, n_atom, 3),
        "contact_probs": torch.zeros(batch_size, n_token, n_token),
        "plddt": torch.zeros(batch_size, n_sample, n_atom, 2),
        "pae": torch.zeros(batch_size, n_sample, n_token, n_token, 2),
        "pde": torch.zeros(batch_size, n_sample, n_token, n_token, 2),
        "resolved": torch.zeros(batch_size, n_sample, n_atom, 2),
    }
    pair_z = torch.zeros(batch_size, n_token, n_token, 4)

    predictions = OpenDDE.postprocess_seed_batch(
        model,
        pred_dict=pred_dict,
        input_feature_dict=input_features,
        pair_input_feature_dict=input_features,
        pair_z=pair_z,
        N_cycle=1,
    )

    assert [prediction["summary_confidence"] for prediction in predictions] == [
        [101],
        [202],
    ]
    assert all(
        prediction["coordinate"].shape == (n_sample, n_atom, 3)
        for prediction in predictions
    )
    assert calls == [(101, (n_token, n_token, 4)), (202, (n_token, n_token, 4))]
    selected = select_seed_batch_features(input_features, 0, batch_size)
    assert selected["template_aatype"] is input_features["template_aatype"]
