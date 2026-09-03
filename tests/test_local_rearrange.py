# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import unittest

import torch

from opendde.model.modules.primitives import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)


def create_qkv(batch_size_dims, n_q, n_kv, d):
    q = torch.rand(size=(*batch_size_dims, n_q, d))
    k = torch.rand(size=(*batch_size_dims, n_kv, d))
    v = torch.rand(size=(*batch_size_dims, n_kv, d))
    return q, k, v


class TestUtils(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_equivalence(self):

        batch_size_dims = (3, 5)
        n = 128 * 2 + 18
        d = 9
        n_queries = 32
        n_keys = 128
        inf = 10e10

        torch.random.manual_seed(42)
        q, k, v = create_qkv(batch_size_dims, n, n, d)
        (
            q_trunked,
            k_trunked,
            _,
            attn_bias_trunked,
            q_pad_length,
        ) = rearrange_to_dense_trunk(
            q,
            k,
            v,
            n_queries,
            n_keys,
            inf=inf,
        )

        q_b, k_b, padding_info = rearrange_qk_to_dense_trunk(
            q, k, dim_q=-2, dim_k=-2, n_queries=n_queries, n_keys=n_keys
        )
        self.assertTrue(
            torch.allclose(
                padding_info["mask_trunked"] > 0, attn_bias_trunked[0, 0] > -1
            )
        )
        self.assertTrue(torch.allclose(q_b, q_trunked))
        self.assertTrue(torch.allclose(k_b, k_trunked))


if __name__ == "__main__":
    unittest.main()
