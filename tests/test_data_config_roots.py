# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import importlib
import os
import unittest
from unittest import mock

import opendde.config.data as configs_data_module


class TestDataConfigRoots(unittest.TestCase):
    def test_data_configs_only_include_inference_roots(self):
        self.assertEqual(
            set(configs_data_module.data_configs),
            {
                "msa",
                "template",
                "ccd_components_file",
                "ccd_components_rdkit_mol_file",
            },
        )

    def test_opendde_root_drives_inference_paths(self):
        with mock.patch.dict(
            os.environ,
            {"OPENDDE_ROOT_DIR": "/tmp/opendde_root"},
            clear=False,
        ):
            module = importlib.reload(configs_data_module)

            self.assertEqual(
                module.data_configs["ccd_components_file"],
                "/tmp/opendde_root/common/components.cif",
            )
            self.assertEqual(
                module.data_configs["template"]["prot_template_mmcif_dir"],
                "/tmp/opendde_root/search_database/mmcif",
            )
            self.assertEqual(
                module.data_configs["template"]["release_dates_path"],
                "/tmp/opendde_root/common/release_date_cache.json",
            )
            self.assertEqual(
                module.data_configs["template"]["obsolete_pdbs_path"],
                "/tmp/opendde_root/common/obsolete_to_successor.json",
            )

        importlib.reload(configs_data_module)

    def test_inference_msa_config_is_scalar(self):
        msa_config = configs_data_module.data_configs["msa"]
        self.assertEqual(msa_config["msa_pool_size"], 16384)
        self.assertEqual(msa_config["msa_depth"], 1280)
        self.assertEqual(msa_config["max_paired_per_species"], 600)
        self.assertEqual(msa_config["max_input_sequences"], -1)


if __name__ == "__main__":
    unittest.main()
