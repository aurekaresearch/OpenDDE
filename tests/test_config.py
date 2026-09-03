# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import sys
import unittest
from unittest import mock

from opendde.config.config import parse_configs, parse_sys_args
from opendde.config.data import data_configs
from opendde.config.inference_defaults import inference_configs
from opendde.config.model_base import configs as configs_base


class TestConfigParsing(unittest.TestCase):
    def test_parse_sys_args_preserves_help_flag(self):
        with mock.patch.object(sys, "argv", ["runner/inference.py", "--help"]):
            self.assertEqual(parse_sys_args(), "--help")

    def test_parse_sys_args_rejects_missing_value(self):
        with mock.patch.object(sys, "argv", ["runner/inference.py", "--model_name"]):
            with self.assertRaisesRegex(ValueError, "Missing value"):
                parse_sys_args()

    def test_parse_sys_args_accepts_equals_form(self):
        with mock.patch.object(
            sys,
            "argv",
            ["runner/inference.py", "--model_name=opendde_v1"],
        ):
            self.assertEqual(parse_sys_args(), "--model_name=opendde_v1")

    def test_parse_configs_supports_quoted_values(self):
        cfg = parse_configs(
            {**configs_base, **{"data": data_configs}, **inference_configs},
            arg_str="--dump_dir 'output with spaces'",
            fill_required_with_null=True,
        )
        self.assertEqual(cfg.dump_dir, "output with spaces")


if __name__ == "__main__":
    unittest.main()
