# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import math
import unittest

import torch

from opendde.config.inference import build_inference_config
from opendde.config.model_registry import DEFAULT_MODEL_NAME, model_configs
from opendde.model.sample_confidence import compute_contact_prob
from opendde.model.utils import distogram_bin_tops, distogram_breaks


class TestDistogramBins(unittest.TestCase):
    def test_only_opendde_v1_profile_is_registered(self):
        self.assertEqual(tuple(model_configs), (DEFAULT_MODEL_NAME,))

    def test_opendde_v1_config_uses_u96_distogram_bins(self):
        cfg = build_inference_config(model_name=DEFAULT_MODEL_NAME)

        self.assertEqual(cfg.no_bins, 96)
        self.assertEqual(cfg.confidence.distogram.min_bin, 2.25)
        self.assertEqual(cfg.confidence.distogram.max_bin, 25.75)
        self.assertEqual(cfg.confidence.distogram.no_bins, 96)

    def test_u96_breaks_match_quarter_angstrom_grid(self):
        breaks = distogram_breaks(min_bin=2.25, max_bin=25.75, no_bins=96)

        self.assertEqual(breaks.shape, torch.Size([95]))
        self.assertAlmostEqual(breaks[0].item(), 2.25)
        self.assertAlmostEqual(breaks[-1].item(), 25.75)
        self.assertAlmostEqual((breaks[1] - breaks[0]).item(), 0.25)
        self.assertAlmostEqual(breaks[23].item(), 8.0)

    def test_u96_bin_tops_include_inf_top_bin(self):
        bin_tops = distogram_bin_tops(min_bin=2.25, max_bin=25.75, no_bins=96)

        self.assertEqual(bin_tops.shape, torch.Size([96]))
        self.assertAlmostEqual(bin_tops[23].item(), 8.0)
        self.assertTrue(math.isinf(bin_tops[-1].item()))

    def test_contact_prob_uses_bin_tops_not_centers(self):
        logits = torch.zeros(1, 1, 96)
        contact_prob = compute_contact_prob(
            distogram_logits=logits,
            min_bin=2.25,
            max_bin=25.75,
            no_bins=96,
            thres=8.0,
        )

        self.assertAlmostEqual(contact_prob.item(), 24 / 96)


if __name__ == "__main__":
    unittest.main()
