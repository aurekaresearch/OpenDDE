# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from biotite.structure import AtomArray

from opendde.data.inference.input_validation import (
    validate_inference_seed,
    validate_sample_name,
)
from opendde.data.utils import save_structure_cif
from opendde.utils.file_io import save_json


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing
    unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """

    def _rounded_copy(value):
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.float()
            return np.round(value.detach().cpu().numpy(), 2)
        if isinstance(value, np.ndarray):
            return np.round(value, 2)
        if isinstance(value, list):
            return list(np.round(np.array(value), 2))
        if isinstance(value, dict):
            return {key: _rounded_copy(item) for key, item in value.items()}
        return value

    # Build a detached serialization tree. Dumping output is an I/O operation
    # and must not delete keys or replace tensors in the caller's prediction.
    return {
        key: _rounded_copy(value)
        for key, value in full_confidence_dict.items()
        if key not in {"atom_coordinate", "atom_is_polymer"}
    }


def _umask_directory_mode() -> int:
    """Return the directory mode ``os.makedirs`` would have produced."""

    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o777 & ~current_umask


def _remove_output_path(path: str) -> None:
    """Remove one file/symlink/tree without following a destination symlink."""

    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


class DataDumper:
    """
    Class for dumping prediction data, including structure coordinates and confidence scores.

    Args:
        base_dir (str): Base directory for saving dumped data.
        need_atom_confidence (bool): Whether to save detailed atom-level confidence data.
        sorted_by_ranking_score (bool): Whether to sort output files by ranking score.
    """

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score

    def dump(
        self,
        group_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            group_name (str): Optional output grouping name.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(group_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
        )

    def _get_dump_dir(self, group_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data based on the group name,
        sample name, and seed.
        """
        sample_name = validate_sample_name(sample_name)
        if group_name:
            group_name = validate_sample_name(group_name)
        seed = validate_inference_seed(seed, location="output seed")
        dump_dir = os.path.join(self.base_dir, group_name, sample_name, f"seed_{seed}")
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
    ):
        """
        Dump raw predictions from the model.

        Args:
            pred_dict (dict): Prediction results.
            dump_dir (str): Directory where to save the predictions.
            pdb_id (str): PDB ID or sample name.
            atom_array (AtomArray): Reference atom array for structure formatting.
            entity_poly_type (dict[str, str]): Dictionary mapping entity IDs to their polymer types.
            seed (int): Random seed used for the prediction.
        """
        os.makedirs(dump_dir, exist_ok=True)
        prediction_save_dir = os.path.join(dump_dir, "predictions")
        staging_dir = tempfile.mkdtemp(prefix=".predictions-staging-", dir=dump_dir)
        # ``mkdtemp`` creates a private (0700) directory; the published
        # ``predictions/`` directory must keep the caller's umask-derived mode.
        os.chmod(staging_dir, _umask_directory_mode())

        try:
            # Dump structure
            b_factor = None
            if "full_data" in pred_dict:
                all_atom_plddt = []
                # len(pred_dict["full_data"]) == N_sample
                for each_sample_dict in pred_dict["full_data"]:
                    if "atom_plddt" in each_sample_dict:
                        # atom_plddt.shape == [N_atom]
                        atom_plddt = each_sample_dict["atom_plddt"]
                        if atom_plddt.dtype == torch.bfloat16:
                            atom_plddt = atom_plddt.to(torch.float32)
                        all_atom_plddt.append(atom_plddt.cpu().numpy() * 100.0)

                if len(all_atom_plddt) == len(pred_dict["full_data"]):
                    b_factor = all_atom_plddt
            sorted_indices = self._get_ranker_indices(data=pred_dict)
            self._save_structure(
                pred_coordinates=pred_dict["coordinate"],
                prediction_save_dir=staging_dir,
                sample_name=pdb_id,
                atom_array=atom_array,
                entity_poly_type=entity_poly_type,
                seed=seed,
                sorted_indices=sorted_indices,
                b_factor=b_factor,
            )
            # Dump confidence
            self._save_confidence(
                data=pred_dict,
                prediction_save_dir=staging_dir,
                sample_name=pdb_id,
                seed=seed,
                sorted_indices=sorted_indices,
            )
            self._replace_prediction_directory(staging_dir, prediction_save_dir)
            staging_dir = ""
        finally:
            if staging_dir and os.path.lexists(staging_dir):
                _remove_output_path(staging_dir)

    @staticmethod
    def _replace_prediction_directory(staging_dir: str, destination: str) -> None:
        """Publish one complete output set without mixing it with a prior run."""

        backup = ""
        if os.path.lexists(destination):
            backup = os.path.join(
                os.path.dirname(destination),
                f".predictions-backup-{uuid.uuid4().hex}",
            )
            os.replace(destination, backup)
        try:
            os.replace(staging_dir, destination)
        except BaseException:
            if backup and os.path.lexists(backup):
                os.replace(backup, destination)
            raise
        if backup and os.path.lexists(backup):
            _remove_output_path(backup)

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
        sorted_indices: Optional[List[int]],
        b_factor: Optional[List[np.ndarray]] = None,
    ):
        """
        Save predicted structures to CIF files.

        Args:
            pred_coordinates (torch.Tensor): Predicted coordinates [N_sample, N_atom, 3].
            prediction_save_dir (str): Directory where to save the structures.
            sample_name (str): Sample name.
            atom_array (AtomArray): Template atom array.
            entity_poly_type (dict[str, str]): Entity polymer types.
            seed (int): Prediction seed.
            sorted_indices (Optional[List[int]]): Indices for ranking.
            b_factor (Optional[List[np.ndarray]]): Predicted LDDT scores to be saved as B-factors.
        """
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        if sorted_indices is None:
            sorted_indices = list(range(N_sample))  # do not rank the output file
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_sample_{rank}.cif",
            )
            output_atom_array = atom_array
            if b_factor is not None:
                # b_factor.shape == [N_sample, N_atom]
                # The dumper is an I/O boundary; annotating the caller's
                # AtomArray makes a later reuse observe the last sample's pLDDT.
                output_atom_array = atom_array.copy()
                output_atom_array.set_annotation("b_factor", np.round(b_factor[idx], 2))

            save_structure_cif(
                atom_array=output_atom_array,
                pred_coordinate=pred_coordinates[idx],
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=sample_name,
            )

    def _get_ranker_indices(self, data: dict) -> List[int]:
        """
        Get indices for ranking predictions based on their confidence scores.

        Args:
            data (dict): Prediction results containing summary confidence.

        Returns:
            List[int]: List of indices sorted by ranking score.
        """
        N_sample = len(data["summary_confidence"])
        if self.sorted_by_ranking_score:
            score_key = (
                "final_score"
                if N_sample > 0 and "final_score" in data["summary_confidence"][0]
                else "ranking_score"
            )
            value = torch.tensor(
                [data["summary_confidence"][i][score_key] for i in range(N_sample)]
            )
            sorted_indices = [
                i for i in torch.argsort(torch.argsort(value, descending=True))
            ]
        else:
            sorted_indices = [i for i in range(N_sample)]
        return sorted_indices

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        seed: int,
        sorted_indices: Optional[List[int]],
    ):
        """
        Save confidence data to JSON files.

        Args:
            data (dict): Prediction results containing confidence scores.
            prediction_save_dir (str): Directory where to save the files.
            sample_name (str): Sample name.
            seed (int): Prediction seed.
            sorted_indices (Optional[List[int]]): Indices for ranking.
        """
        N_sample = len(data["summary_confidence"])
        if sorted_indices is None:
            sorted_indices = list(range(N_sample))
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{rank}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{rank}.json",
                )
                save_json(
                    get_clean_full_confidence(data["full_data"][idx]),
                    output_fpath,
                    indent=None,
                )
