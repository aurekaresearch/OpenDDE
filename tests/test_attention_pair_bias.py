# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import math
import os
import time
import unittest
from unittest import mock

import torch

os.environ["LAYERNORM_TYPE"] = "torch"
from opendde.model.modules.transformer import (
    AttentionPairBias,
    _foldcp_diffusion_query_range,
    _prepare_foldcp_diffusion_bias_cache_source,
    foldcp_diffusion_bias_cache_is_safe,
)


class TestAttentionPairBias(unittest.TestCase):
    def setUp(self) -> None:
        self._start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().setUp()

    def get_model(
        self,
        has_s=True,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
    ):

        model = AttentionPairBias(
            has_s=has_s, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z
        ).to(self.device)

        return model

    def test_foldcp_diffusion_query_range(self) -> None:
        for n_token, cp_size in ((1, 2), (9, 4), (13, 8), (17, 3)):
            with self.subTest(n_token=n_token, cp_size=cp_size):
                ranges = [
                    _foldcp_diffusion_query_range(
                        n_token=n_token,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                    )
                    for cp_rank in range(cp_size)
                ]
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], n_token)
                self.assertTrue(
                    all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
                )
                lengths = [end - start for start, end in ranges]
                self.assertLessEqual(max(lengths) - min(lengths), 1)

        invalid_args = (
            {"n_token": 0, "cp_size": 2, "cp_rank": 0},
            {"n_token": 1, "cp_size": 0, "cp_rank": 0},
            {"n_token": 1, "cp_size": 2, "cp_rank": -1},
            {"n_token": 1, "cp_size": 2, "cp_rank": 2},
        )
        for args in invalid_args:
            with self.subTest(args=args), self.assertRaises(ValueError):
                _foldcp_diffusion_query_range(**args)

    def test_foldcp_diffusion_bias_cache_budget_boundary(self) -> None:
        cache_args = {
            "n_blocks": 2,
            "n_heads": 3,
            "bias_rows": 5,
            "bias_cols": 7,
            "element_size": 4,
        }
        resident_bytes = 2 * 3 * 5 * 7 * 4
        variable = "OPENDDE_FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES"
        with mock.patch.dict(os.environ, {variable: str(resident_bytes)}):
            self.assertTrue(foldcp_diffusion_bias_cache_is_safe(**cache_args))
        with mock.patch.dict(os.environ, {variable: str(resident_bytes - 1)}):
            self.assertFalse(foldcp_diffusion_bias_cache_is_safe(**cache_args))

    def test_foldcp_diffusion_bias_cache_gathers_only_query_owned_rows(self) -> None:
        z_local = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2)
        extra_attn_bias = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        local_source, gathered_source, packed, extra_local = (
            _prepare_foldcp_diffusion_bias_cache_source(
                z_local,
                extra_attn_bias,
                row_start=0,
                row_end=5,
                col_start=4,
                col_end=5,
                valid_rows=5,
                valid_cols=1,
                tile_cols=2,
                mesh_cols=3,
            )
        )

        self.assertTrue(packed)
        self.assertEqual(local_source.shape, (6, 2, 3))
        self.assertEqual(gathered_source.shape, (6, 2, 3))
        for destination in range(3):
            query_start, query_end = _foldcp_diffusion_query_range(
                n_token=5,
                cp_size=3,
                cp_rank=destination,
            )
            source_col = destination * 2
            valid_query_rows = query_end - query_start
            self.assertTrue(
                torch.equal(
                    local_source[source_col, :valid_query_rows, :2],
                    z_local[query_start:query_end, 0],
                )
            )
            self.assertTrue(
                torch.equal(
                    local_source[source_col, :valid_query_rows, 2],
                    extra_attn_bias[query_start:query_end, 0],
                )
            )
            self.assertEqual(torch.count_nonzero(local_source[source_col + 1]), 0)
            self.assertEqual(
                torch.count_nonzero(local_source[source_col, valid_query_rows:]),
                0,
            )
        self.assertTrue(torch.equal(extra_local, extra_attn_bias[:, :1]))

    def test_foldcp_diffusion_bias_cache_all_to_all_reconstructs_source(self) -> None:
        for n_token in (7, 8, 9, 10, 17):
            for cp_size in (2, 3, 4, 5, 7):
                if cp_size > n_token:
                    continue
                with self.subTest(n_token=n_token, cp_size=cp_size):
                    tile_cols = math.ceil(n_token / cp_size)
                    global_z = torch.arange(
                        n_token * n_token * 2,
                        dtype=torch.float32,
                    ).reshape(n_token, n_token, 2)
                    global_extra = torch.arange(
                        n_token * n_token,
                        dtype=torch.float32,
                    ).reshape(n_token, n_token)
                    send_buffers = []
                    for source in range(cp_size):
                        col_start = source * tile_cols
                        col_end = min(col_start + tile_cols, n_token)
                        valid_cols = max(0, col_end - col_start)
                        z_local = torch.zeros(n_token, tile_cols, 2)
                        extra_local = torch.zeros(n_token, tile_cols)
                        z_local[:, :valid_cols] = global_z[:, col_start:col_end]
                        extra_local[:, :valid_cols] = global_extra[:, col_start:col_end]
                        send_buffer, _, packed, _ = (
                            _prepare_foldcp_diffusion_bias_cache_source(
                                z_local,
                                extra_local,
                                row_start=0,
                                row_end=n_token,
                                col_start=col_start,
                                col_end=col_end,
                                valid_rows=n_token,
                                valid_cols=valid_cols,
                                tile_cols=tile_cols,
                                mesh_cols=cp_size,
                            )
                        )
                        self.assertTrue(packed)
                        send_buffers.append(send_buffer)

                    for destination in range(cp_size):
                        query_start, query_end = _foldcp_diffusion_query_range(
                            n_token=n_token,
                            cp_size=cp_size,
                            cp_rank=destination,
                        )
                        received = torch.cat(
                            [
                                source[
                                    destination * tile_cols : (destination + 1)
                                    * tile_cols
                                ]
                                for source in send_buffers
                            ],
                            dim=0,
                        )
                        projection_source = received[
                            :n_token, : query_end - query_start
                        ]
                        self.assertTrue(
                            torch.equal(
                                projection_source[..., :-1],
                                global_z[query_start:query_end].transpose(0, 1),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                projection_source[..., -1],
                                global_extra[query_start:query_end].transpose(0, 1),
                            )
                        )

    def test_project_attention_bias_fusion_accepts_zero_width(self) -> None:
        model = self.get_model(has_s=False, n_heads=2, c_a=4, c_z=3)
        z = torch.empty((13, 0, 3), device=self.device)
        extra_attn_bias = torch.empty((13, 0), device=self.device)

        expected = model._project_attention_bias(
            z,
            extra_attn_bias=extra_attn_bias,
            enable_efficient_fusion=False,
        )
        actual = model._project_attention_bias(
            z,
            extra_attn_bias=extra_attn_bias,
            enable_efficient_fusion=True,
        )

        self.assertEqual(actual.shape, (2, 13, 0))
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.equal(actual, expected))

    def test_shape(self) -> None:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """
        n_heads = 3
        c_a = 3 * 55
        c_s = 23
        c_z = 17

        N_token = 135
        bs_dims = (2, 3)

        inputs = {
            "a": torch.rand(size=(*bs_dims, N_token, c_a)).to(self.device),
            "s": torch.rand(size=(*bs_dims, N_token, c_s)).to(self.device),
            "z": torch.rand(size=(*bs_dims, N_token, N_token, c_z)).to(self.device),
        }

        model = self.get_model(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)

        out = model(**inputs)
        target_shape = (*bs_dims, N_token, c_a)
        self.assertEqual(out.shape, out.reshape(target_shape).shape)

    def test_local_attention_shape(self) -> None:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            a (torch.Tensor): atom embedding
                [..., N_atom, c_a]
            s (torch.Tensor): atom embedding
                [..., N_atom, c_s]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """
        n_heads = 3
        c_a = 3 * 27
        c_s = 23
        c_z = 17

        N_token = 128 * 2 + 45

        bs_dims = (2, 3)

        N_q = 32
        N_k = 128
        N_blocks = math.ceil(N_token / N_q)

        inputs = {
            "a": torch.rand(size=(*bs_dims, N_token, c_a)).to(self.device),
            "s": torch.rand(size=(*bs_dims, N_token, c_s)).to(self.device),
            "z": torch.rand(size=(*bs_dims, N_blocks, N_q, N_k, c_z)).to(self.device),
            "n_queries": 32,
            "n_keys": 128,
        }

        model = self.get_model(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)

        out = model(**inputs)
        target_shape = (*bs_dims, N_token, c_a)
        self.assertEqual(out.shape, out.reshape(target_shape).shape)

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        print(f"Test {self.id()} took {elapsed_time:.6f}s")


if __name__ == "__main__":
    unittest.main()
