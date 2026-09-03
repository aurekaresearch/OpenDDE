# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import time
import unittest

import torch
import torch.nn.functional as F

from opendde.model.msa_sampling import subsample_msa_feature_dict_valid_first
from opendde.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    centre_random_augmentation,
    expand_at_dim,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self._start_time = time.time()
        return super().setUp()

    def test_reshape_at_dim(self):
        x = torch.rand([1, 3 * 4, 5, 2 * 5 * 7, 9])
        x_reshape = reshape_at_dim(x, dim=1, target_shape=(3, 4))
        x_rs = x.reshape([1, 3, 4, 5, 2 * 5 * 7, 9])
        self.assertTrue(torch.allclose(x_reshape, x_rs))

        x_reshape = reshape_at_dim(x, dim=-2, target_shape=(5, 2, 7))
        x_rs = x.reshape([1, 3 * 4, 5, 5, 2, 7, 9])
        self.assertTrue(torch.allclose(x_reshape, x_rs))

    def test_move_final_dim_to_dim(self):
        x = torch.rand([3, 2, 4, 5, 3, 7])
        x_perm = x.permute(0, 1, 2, 3, 5, 4)
        self.assertTrue(torch.allclose(move_final_dim_to_dim(x, dim=-2), x_perm))
        x_perm = x.permute(0, 1, 2, 3, 4, 5)
        self.assertTrue(torch.allclose(move_final_dim_to_dim(x, dim=-1), x_perm))
        x_perm = x.permute(5, 0, 1, 2, 3, 4)
        self.assertTrue(torch.allclose(move_final_dim_to_dim(x, dim=0), x_perm))
        x_perm = x.permute(0, 1, 5, 2, 3, 4)
        self.assertTrue(torch.allclose(move_final_dim_to_dim(x, dim=2), x_perm))

    def test_pad_at_dim(self):
        x = torch.rand([3, 2, 4, 5, 3, 7])
        x_pad = F.pad(x, (0, 0, 1, 2))
        self.assertTrue(torch.allclose(pad_at_dim(x, dim=-2, pad_length=(1, 2)), x_pad))

        x_pad = F.pad(x, (0, 0, 0, 0, 3, 5))
        self.assertTrue(torch.allclose(pad_at_dim(x, dim=-3, pad_length=(3, 5)), x_pad))

    def test_aggregate_atom_to_token(self):
        # value check
        N_atom = 10
        n_token = 3
        x_atom = torch.ones([10, N_atom, 3])
        atom_to_token_idx = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]).long()
        out = aggregate_atom_to_token(
            x_atom=x_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="sum",
        )
        self.assertTrue(torch.equal(torch.unique(out), torch.tensor([2, 4])))
        out = aggregate_atom_to_token(
            x_atom=x_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )
        self.assertTrue(torch.equal(torch.unique(out), torch.tensor([1])))
        # batch shape check
        # it support batch mode
        x_atom = torch.ones([N_atom, 3])
        x_atom = expand_at_dim(x_atom, dim=0, n=2)
        atom_to_token_idx = expand_at_dim(atom_to_token_idx, dim=0, n=2)
        out = aggregate_atom_to_token(
            x_atom=x_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="sum",
        )
        self.assertTrue(torch.equal(torch.unique(out), torch.tensor([2, 4])))

    def test_broadcast_token_to_atom(self):
        N_token = 3
        x_token = torch.zeros([10, N_token, 3])
        for i in range(N_token):
            x_token[:, i, :] = i
        atom_to_token_idx = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]).long()
        out = broadcast_token_to_atom(
            x_token=x_token, atom_to_token_idx=atom_to_token_idx
        )
        # value check
        self.assertTrue(torch.all(out[:, :4, :].eq(0)))
        self.assertTrue(torch.all(out[:, 4:8, :].eq(1)))
        self.assertTrue(torch.all(out[:, 8:, :].eq(2)))
        # batch mode check
        x_token = expand_at_dim(x_token, 0, 2)
        atom_to_token_idx = expand_at_dim(atom_to_token_idx, 0, 2)
        self.assertTrue(torch.all(out[..., :4, :].eq(0)))
        self.assertTrue(torch.all(out[..., 4:8, :].eq(1)))
        self.assertTrue(torch.all(out[..., 8:, :].eq(2)))
        # also it does not support an extra N sample dim after batch dim

    def test_centre_random_augmentation(self):
        bs_dims = (4, 3, 2)
        N_atom = 7
        N_sample = 8

        x = torch.rand(size=(*bs_dims, N_atom, 3))
        out = centre_random_augmentation(x_input_coords=x, N_sample=N_sample)
        # shape check
        self.assertEqual(out.shape, torch.Size((*bs_dims, N_sample, N_atom, 3)))

    def test_subsample_msa_valid_first_prefers_non_gap_rows(self):
        torch.manual_seed(0)
        gap_token = 31
        feat_dict = {
            "msa": torch.tensor(
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                    [gap_token, gap_token, gap_token],
                    [gap_token, gap_token, gap_token],
                ]
            ),
            "has_deletion": torch.zeros(5, 3),
            "deletion_value": torch.zeros(5, 3),
        }

        sampled = subsample_msa_feature_dict_valid_first(
            feat_dict=feat_dict,
            dim_dict={k: -2 for k in feat_dict},
            num_msa=3,
            gap_token=gap_token,
        )

        self.assertEqual(sampled["msa"].shape[0], 3)
        self.assertTrue(torch.all(sampled["msa"].ne(gap_token).any(dim=-1)))

    def test_subsample_msa_valid_first_backfills_with_gap_rows(self):
        torch.manual_seed(0)
        gap_token = 31
        feat_dict = {
            "msa": torch.tensor(
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [gap_token, gap_token, gap_token],
                    [gap_token, gap_token, gap_token],
                ]
            ),
            "has_deletion": torch.zeros(4, 3),
            "deletion_value": torch.zeros(4, 3),
        }

        sampled = subsample_msa_feature_dict_valid_first(
            feat_dict=feat_dict,
            dim_dict={k: -2 for k in feat_dict},
            num_msa=3,
            gap_token=gap_token,
        )

        self.assertEqual(sampled["msa"].shape[0], 3)
        self.assertTrue(torch.all(sampled["msa"][:2].ne(gap_token).any(dim=-1)))
        self.assertTrue(torch.all(sampled["msa"][2:].eq(gap_token).all(dim=-1)))

    def test_subsample_msa_valid_first_ignores_all_one_mask(self):
        torch.manual_seed(0)
        gap_token = 31
        feat_dict = {
            "msa": torch.tensor(
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [gap_token, gap_token, gap_token],
                ]
            ),
            "has_deletion": torch.zeros(3, 3),
            "deletion_value": torch.zeros(3, 3),
        }
        msa_mask = torch.ones(3, 3, dtype=torch.bool)

        sampled = subsample_msa_feature_dict_valid_first(
            feat_dict=feat_dict,
            dim_dict={k: -2 for k in feat_dict},
            num_msa=2,
            msa_mask=msa_mask,
            gap_token=gap_token,
        )

        self.assertEqual(sampled["msa"].shape[0], 2)
        self.assertTrue(torch.all(sampled["msa"].ne(gap_token).any(dim=-1)))

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        print(f"Test {self.id()} took {elapsed_time:.6f}s")


if __name__ == "__main__":
    unittest.main()
