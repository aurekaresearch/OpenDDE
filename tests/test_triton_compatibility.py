# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2025 Shad Nygren, Virtual Hipster Corporation
# Contributed to the OpenDDE project under the Apache License 2.0

"""Tests for Triton dependency and PyTorch triangle-attention fallback paths."""

import unittest

import torch


class TestTritonCompatibility(unittest.TestCase):
    """Document Triton availability while keeping triangle attention on cueq/torch."""

    def test_triton_version(self):
        """Verify Triton major version is compatible when installed."""
        try:
            import triton

            version = triton.__version__
            self.assertEqual(
                version.split(".")[0],
                "3",
                f"Triton major version should be 3, got {version}",
            )
        except ImportError:
            self.skipTest("Triton not installed")

    def test_torch_triangle_attention_path(self):
        """PyTorch-native triangle attention path should work on CPU."""
        from opendde.model.triangular.layers import Attention

        attention = Attention(c_q=8, c_k=8, c_v=8, c_hidden=4, no_heads=2)
        q_x = torch.randn(2, 4, 8)
        out = attention(q_x=q_x, kv_x=q_x, triangle_attention="torch")
        self.assertEqual(out.shape, q_x.shape)

    def test_unsupported_triangle_attention_option_rejected(self):
        """Only supported triangle-attention implementations should be accepted."""
        from opendde.model.triangular.layers import Attention

        attention = Attention(c_q=8, c_k=8, c_v=8, c_hidden=4, no_heads=2)
        q_x = torch.randn(2, 4, 8)
        with self.assertRaises(ValueError):
            attention(q_x=q_x, kv_x=q_x, triangle_attention="unsupported")


if __name__ == "__main__":
    unittest.main()
