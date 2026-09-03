# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import shutil
import subprocess
import time
from typing import Sequence

from opendde.data.tools.common import parse_kalign_a3m, tmpdir_manager, to_a3m
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_kalign_binary(binary_path: str | None) -> str:
    """Resolve an explicit Kalign binary before falling back to PATH."""
    found_path = shutil.which(binary_path) if binary_path else None
    if found_path is None and binary_path != "kalign":
        found_path = shutil.which("kalign")
    if found_path is None:
        raise RuntimeError(
            f"Kalign binary not found. Neither the provided path "
            f"({binary_path}) nor kalign in system PATH is executable.\n"
            f"To install kalign, you can use one of the following methods:\n"
            f"1. Using conda: conda install -c bioconda kalign\n"
            f"2. Using apt (Ubuntu/Debian): apt-get install kalign\n"
            f"3. Download from: https://github.com/TimoLassmann/kalign\n"
            f"After installation, make sure the binary is accessible either in PATH or at the specified location."
        )
    return found_path


class Kalign:
    """
    Python wrapper of the Kalign binary.
    Adapted from openfold.data.tools.kalign

    Args:
      binary_path: The path to the Kalign binary.

    Raises:
      RuntimeError: If Kalign binary not found within the path.
    """

    def __init__(self, *, binary_path: str | None):
        self.binary_path = resolve_kalign_binary(binary_path)

    def align(self, sequences: Sequence[str]) -> Sequence[str]:
        """Aligns the sequences and returns the alignment in A3M string.

        Args:
          sequences: A list of query sequence strings. The sequences have to be at
            least 6 residues long (Kalign requires this). Note that the order in
            which you give the sequences might alter the output slightly as
            different alignment tree might get constructed.

        Returns:
          A list of strings with the aligned sequences in a3m format.

        Raises:
          RuntimeError: If Kalign fails.
          ValueError: If any of the sequences is less than 6 residues long.
        """

        for s in sequences:
            if len(s) < 6:
                raise ValueError(
                    "Kalign requires all sequences to be at least 6 "
                    f"residues long. Got {s} ({len(s)} residues)."
                )

        with tmpdir_manager() as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, "input.fasta")
            output_a3m_path = os.path.join(query_tmp_dir, "output.a3m")

            logger.debug(
                "Kalign tmpdir=%s input_fasta=%s output_a3m=%s",
                query_tmp_dir,
                input_fasta_path,
                output_a3m_path,
            )

            write_st = time.perf_counter()
            with open(input_fasta_path, "w") as f:
                f.write(to_a3m(sequences))
            write_seconds = time.perf_counter() - write_st

            cmd = [
                self.binary_path,
                "-i",
                input_fasta_path,
                "-o",
                output_a3m_path,
                "-format",
                "fasta",
            ]

            run_st = time.perf_counter()
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            run_seconds = time.perf_counter() - run_st

            if process.returncode:
                logger.debug(
                    "Kalign timings write=%.4fs run=%.4fs read=%.4fs",
                    write_seconds,
                    run_seconds,
                    0.0,
                )
                raise RuntimeError(
                    f"Kalign failed with exit code {process.returncode}\n"
                    f"stdout:\n{process.stdout}\n\nstderr:\n{process.stderr}\n"
                )

            read_st = time.perf_counter()
            with open(output_a3m_path) as f:
                a3m = f.read()
            read_seconds = time.perf_counter() - read_st

            logger.debug(
                "Kalign timings write=%.4fs run=%.4fs read=%.4fs",
                write_seconds,
                run_seconds,
                read_seconds,
            )
            return parse_kalign_a3m(a3m)
