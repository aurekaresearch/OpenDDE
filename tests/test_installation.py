# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2025 Shad Nygren, Virtual Hipster Corporation
# Contributed to the OpenDDE project under the Apache License 2.0

"""
Test suite for installation and dependency compatibility issues.
"""

import importlib
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement


class TestInstallation(unittest.TestCase):
    """Test that package dependencies are compatible"""

    def test_required_packages_importable(self):
        """Verify every declared core runtime can actually be imported."""
        required_modules = [
            "torch",
            "click",
            "scipy",
            "ml_collections",
            "tqdm",
            "pandas",
            "rdkit",
            "Bio",
            "biotite",
            "sklearn",
            "pydantic",
            "optree",
            "numpy",
            "networkx",
            "packaging",
            "requests",
        ]

        for module_name in required_modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_opendde_imports(self):
        """Test that opendde package can be imported when installed"""
        try:
            import opendde  # noqa: F401

            self.assertTrue(True)
        except ImportError:
            self.skipTest(
                "OpenDDE not installed - this documents the need for installation"
            )

    def test_cuda_availability(self):
        """Document CUDA availability for GPU acceleration"""
        try:
            import torch

            if torch.cuda.is_available():
                self.assertTrue(True, "CUDA is available")
            else:
                # Not a failure, just documentation
                print("Note: CUDA not available, will use CPU")
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_cpu_and_gpu_install_extras_are_declared(self):
        """Verify the compatibility CPU extra and useful GPU extra."""
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        optional_dependencies = pyproject["project"]["optional-dependencies"]
        runtime_dependencies = pyproject["project"]["dependencies"]
        dev_dependencies = pyproject["dependency-groups"]["dev"]

        self.assertIn("cpu", optional_dependencies)
        self.assertEqual(optional_dependencies["cpu"], [])
        self.assertIn("gpu", optional_dependencies)

        gpu_requirements = [
            Requirement(dependency) for dependency in optional_dependencies["gpu"]
        ]
        gpu_requirements_by_name = {
            requirement.name: requirement for requirement in gpu_requirements
        }
        expected_gpu_packages = {
            "triton": "==3.3.1",
            "cuequivariance": "==0.10.0",
            "cuequivariance-torch": "==0.10.0",
            "cuequivariance-ops-cu12": "==0.10.0",
            "cuequivariance-ops-torch-cu12": "==0.10.0",
        }
        self.assertEqual(set(gpu_requirements_by_name), set(expected_gpu_packages))
        self.assertEqual(len(gpu_requirements), len(expected_gpu_packages))

        platform_matrix = {
            "Linux x86_64": (
                {"platform_system": "Linux", "platform_machine": "x86_64"},
                True,
            ),
            "Linux aarch64": (
                {"platform_system": "Linux", "platform_machine": "aarch64"},
                False,
            ),
            "Windows AMD64": (
                {"platform_system": "Windows", "platform_machine": "AMD64"},
                False,
            ),
            "macOS arm64": (
                {"platform_system": "Darwin", "platform_machine": "arm64"},
                False,
            ),
        }
        for requirement in gpu_requirements:
            self.assertEqual(
                str(requirement.specifier), expected_gpu_packages[requirement.name]
            )
            marker = requirement.marker
            assert marker is not None
            for platform_name, (environment, expected) in platform_matrix.items():
                with self.subTest(requirement=requirement.name, platform=platform_name):
                    self.assertEqual(marker.evaluate(environment=environment), expected)

        runtime_package_names = {
            Requirement(dependency).name.lower() for dependency in runtime_dependencies
        }
        self.assertTrue(
            runtime_package_names.isdisjoint(
                {
                    "cuequivariance",
                    "triton",
                    "scikit-learn-extra",
                    "torchvision",
                    "torchaudio",
                    "pyyaml",
                    "matplotlib",
                    "ipywidgets",
                    "py3dmol",
                    "modelcif",
                    "gemmi",
                    "pdbeccdutils",
                    "protobuf",
                    "typing-extensions",
                }
            )
        )
        self.assertIn("numpy==2.4.1", runtime_dependencies)
        self.assertIn("packaging>=23.2", runtime_dependencies)

        self.assertFalse(any("icecream" in dep for dep in runtime_dependencies))
        self.assertFalse(any("ipdb" in dep for dep in runtime_dependencies))
        self.assertFalse(any("icecream" in dep for dep in dev_dependencies))
        self.assertFalse(any("ipdb" in dep for dep in dev_dependencies))
        self.assertEqual(pyproject["project"]["requires-python"], ">=3.11,<3.14")
        self.assertEqual(
            pyproject["project"]["scripts"]["opendde"],
            "runner.cli:opendde_cli",
        )
        self.assertNotIn(
            "Operating System :: Microsoft :: Windows",
            pyproject["project"]["classifiers"],
        )

    def test_doctor_command_is_registered(self):
        """The public CLI should print the generated environment report."""
        from click.testing import CliRunner
        from opendde.utils import environment
        from runner import cli

        report = "OpenDDE environment report"
        with patch.object(
            environment, "format_doctor_report", return_value=report
        ) as format_report:
            result = CliRunner().invoke(cli.opendde_cli, ["doctor"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, f"{report}\n")
        format_report.assert_called_once_with()

        suggestion = CliRunner().invoke(cli.opendde_cli, ["doctro"])
        self.assertNotEqual(suggestion.exit_code, 0)
        self.assertIn("Did you mean one of these?", suggestion.output)
        self.assertIn("doctor", suggestion.output)

    def test_lightweight_cli_survives_broken_torch_and_rdkit_imports(self):
        """Doctor and top-level help must start before heavy runtime imports."""
        script = """
import builtins
from click.testing import CliRunner

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise OSError("simulated Torch ABI failure")
    if name == "rdkit" or name.startswith("rdkit."):
        raise OSError("simulated RDKit ABI failure")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from runner.cli import opendde_cli

runner = CliRunner()
help_result = runner.invoke(opendde_cli, ["--help"])
if help_result.exit_code != 0:
    raise SystemExit(help_result.exit_code)
doctor_result = runner.invoke(opendde_cli, ["doctor"])
print(doctor_result.output, end="")
raise SystemExit(doctor_result.exit_code)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OpenDDE environment", result.stdout)
        self.assertIn("simulated Torch ABI failure", result.stdout)

    def test_predict_command_exposes_device_selection(self):
        """Users can override automatic CUDA/CPU selection."""
        import click
        from runner import cli
        from runner.batch_inference import opendde_cli

        context = click.Context(cli.opendde_cli)
        self.assertIs(opendde_cli, cli.opendde_cli)
        self.assertEqual(
            set(cli.opendde_cli.list_commands(context)),
            {"doctor", "json", "msa", "mt", "pred", "prep"},
        )
        predict_command = cli.opendde_cli.get_command(context, "pred")
        assert predict_command is not None
        device_options = [
            parameter
            for parameter in predict_command.params
            if isinstance(parameter, click.Option) and parameter.name == "device"
        ]

        self.assertEqual(len(device_options), 1)
        device_option = device_options[0]
        device_type = device_option.type
        assert isinstance(device_type, click.Choice)
        self.assertEqual(set(device_type.choices), {"auto", "cpu", "cuda"})
        self.assertEqual(device_option.default, "auto")

        dtype_options = [
            parameter
            for parameter in predict_command.params
            if isinstance(parameter, click.Option) and parameter.name == "dtype"
        ]
        self.assertEqual(len(dtype_options), 1)
        dtype_type = dtype_options[0].type
        assert isinstance(dtype_type, click.Choice)
        self.assertEqual(set(dtype_type.choices), {"bf16", "fp32"})
        self.assertEqual(dtype_options[0].default, "fp32")

    def test_legacy_module_entry_point_uses_the_same_commands(self):
        """The documented ``python -m`` invocation must remain functional."""
        result = subprocess.run(
            [sys.executable, "-m", "runner.batch_inference", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pred", result.stdout)
        self.assertIn("doctor", result.stdout)


if __name__ == "__main__":
    unittest.main()  # Test signed commit
