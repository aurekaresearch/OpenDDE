# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import math
import os
import unittest

import torch

os.environ["LAYERNORM_TYPE"] = "torch"
from opendde.model.modules.transformer import AtomAttentionDecoder


class TestAtomAttentionDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_token = 24
        self.c_atom = 16
        self.c_atompair = 8
        self.n_token = 5
        self.n_atom = 10
        self.n_queries = 4
        self.n_keys = 8
        self.n_sample = 2
        self.n_blocks = math.ceil(self.n_atom / self.n_queries)
        self.atom_to_token_idx = torch.tensor(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=self.device
        )
        self.ref_mask = torch.ones(self.n_atom, device=self.device)

    def _make_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "a": torch.randn(
                self.n_sample, self.n_token, self.c_token, device=self.device
            ),
            "q_skip": torch.randn(
                self.n_sample, self.n_atom, self.c_atom, device=self.device
            ),
            "c_skip": torch.randn(
                self.n_sample, self.n_atom, self.c_atom, device=self.device
            ),
            "p_skip": torch.randn(
                self.n_sample,
                self.n_blocks,
                self.n_queries,
                self.n_keys,
                self.c_atompair,
                device=self.device,
            ),
        }

    def test_decoder_shape(self) -> None:
        decoder = AtomAttentionDecoder(
            n_blocks=2,
            n_heads=4,
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        ).to(self.device)

        out = decoder(
            atom_to_token_idx=self.atom_to_token_idx,
            **self._make_inputs(),
        )

        self.assertEqual(out.shape, (self.n_sample, self.n_atom, 3))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_decoder_accepts_shared_pair_skip(self) -> None:
        decoder = AtomAttentionDecoder(
            n_blocks=2,
            n_heads=4,
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        ).to(self.device)
        inputs = self._make_inputs()
        shared_p_skip = inputs["p_skip"][:1]
        expanded_p_skip = shared_p_skip.expand(self.n_sample, *shared_p_skip.shape[1:])

        out = decoder(
            atom_to_token_idx=self.atom_to_token_idx,
            **{**inputs, "p_skip": shared_p_skip},
        )
        out_expanded = decoder(
            atom_to_token_idx=self.atom_to_token_idx,
            **{**inputs, "p_skip": expanded_p_skip},
        )

        self.assertEqual(out.shape, (self.n_sample, self.n_atom, 3))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.allclose(out, out_expanded, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
