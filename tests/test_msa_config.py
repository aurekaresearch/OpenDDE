# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import unittest

from opendde.config.config import parse_configs
from opendde.config.data import data_configs
from opendde.config.inference_defaults import inference_configs
from opendde.config.model_base import configs as configs_base
from opendde.model.modules.pairformer import MSAModule


class TestMSAConfig(unittest.TestCase):
    def test_msa_config_parsing(self):
        cfg = parse_configs(
            {**configs_base, **{"data": data_configs}, **inference_configs},
            fill_required_with_null=True,
        )

        self.assertEqual(cfg.data.msa.msa_pool_size, 16384)
        self.assertEqual(cfg.data.msa.msa_depth, 1280)
        self.assertEqual(cfg.data.msa.max_paired_per_species, 600)
        self.assertTrue(cfg.use_msa)

        cfg = parse_configs(
            {**configs_base, **{"data": data_configs}, **inference_configs},
            arg_str="--model.N_cycle 2 --data.msa.msa_depth 256",
            fill_required_with_null=True,
        )
        self.assertEqual(cfg.model.N_cycle, 2)
        self.assertEqual(cfg.data.msa.msa_depth, 256)

    def test_msa_module_requires_explicit_msa_depth(self):
        with self.assertRaisesRegex(ValueError, "msa_depth"):
            MSAModule(msa_configs={})

        module = MSAModule(
            n_blocks=0,
            c_m=4,
            c_z=4,
            c_s_inputs=4,
            msa_configs={"msa_depth": 8},
        )
        self.assertEqual(module.msa_depth, 8)


if __name__ == "__main__":
    unittest.main()
