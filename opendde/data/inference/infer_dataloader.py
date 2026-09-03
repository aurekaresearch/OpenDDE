# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
import logging
import os
import time
import traceback
import warnings
from collections.abc import Iterable, Iterator, Sized
from typing import Any, cast

import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, Sampler

from opendde.data.core import ccd
from opendde.data.inference.input_validation import validate_inference_jobs
from opendde.data.inference.json_to_feature import SampleDictToFeatures
from opendde.data.msa.msa_featurizer import InferenceMSAFeaturizer
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.data.template.template_utils import TemplateHitFeaturizer
from opendde.data.utils import data_type_transform, make_dummy_feature
from opendde.utils.distributed import DIST_WRAPPER
from opendde.utils.torch_utils import collate_fn_identity, dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


class InferenceJobSampler(Sampler[int]):
    """Shard whole inference jobs across DP groups and replicate within CP.

    All ranks in one Fold-CP group use the same ``rank`` here, so they receive
    the same job sequence. DP groups are padded to the same number of forwards
    because Fold-CP creates process groups collectively during model execution;
    ``owns`` identifies real assignments so padded work never writes outputs.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        num_replicas: int,
        rank: int,
        sample_indices: Iterable[int] | None = None,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.set_sample_indices(sample_indices)

    def set_sample_indices(self, sample_indices: Iterable[int] | None) -> None:
        indices = (
            list(range(len(self.data_source)))
            if sample_indices is None
            else [int(index) for index in sample_indices]
        )
        if any(index < 0 or index >= len(self.data_source) for index in indices):
            raise IndexError("inference sample index is out of range")
        self._sample_indices = indices
        if not indices:
            self._local_indices = []
            self._owned_indices: set[int] = set()
            return

        num_samples = (len(indices) + self.num_replicas - 1) // self.num_replicas
        total_size = num_samples * self.num_replicas
        padding_size = total_size - len(indices)
        padded = indices + (indices * (padding_size // len(indices) + 1))[:padding_size]
        positions = range(self.rank, total_size, self.num_replicas)
        self._local_indices = [padded[position] for position in positions]
        self._owned_indices = {
            padded[position]
            for position in range(self.rank, len(indices), self.num_replicas)
        }

    def __iter__(self) -> Iterator[int]:
        return iter(self._local_indices)

    def __len__(self) -> int:
        return len(self._local_indices)

    def owns(self, sample_index: int) -> bool:
        """Return whether this DP rank owns, rather than pads, this job."""
        return int(sample_index) in self._owned_indices


def _data_parallel_coordinates(configs: Any) -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()
    else:
        world_size = DIST_WRAPPER.world_size
        world_rank = DIST_WRAPPER.rank
    if getattr(configs, "foldcp_mode", "single") == "distributed":
        size_dp = int(getattr(configs, "foldcp_size_dp", 1))
        size_cp = int(getattr(configs, "foldcp_size_cp", 1))
        if size_dp != 1:
            raise ValueError(
                "Only the maintained 1 x P Fold-CP topology is supported; "
                "foldcp_size_dp must be 1."
            )
        if world_size != size_cp:
            raise ValueError(
                "Distributed 1 x P Fold-CP dataloading requires WORLD_SIZE to "
                f"equal foldcp_size_cp; got {world_size} vs {size_cp}."
            )
        # Every rank is one member of the same CP replica. All ranks must see
        # the same job sequence; there is no data-parallel job sharding.
        return 1, 0
    return world_size, world_rank


def get_inference_dataloader(
    configs: Any,
    *,
    inputs: list[dict[str, Any]] | None = None,
) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        configs=configs,
        inputs=inputs,
    )
    size_dp, dp_rank = _data_parallel_coordinates(configs)
    sampler = InferenceJobSampler(
        inference_dataset,
        num_replicas=size_dp,
        rank=dp_rank,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        configs,
        inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.configs = configs

        self.input_json_path = configs.input_json_path
        self.dump_dir = configs.dump_dir
        self.use_msa = configs.use_msa
        self.msa_pair_as_unpair = configs.get("msa_pair_as_unpair", True)
        self.use_rna_msa = configs.get("use_rna_msa", True)
        self.use_template = configs.get("use_template", True)
        ccd.set_ccd_cache_paths(
            components_file=configs.data.ccd_components_file,
            rdkit_mol_pkl=configs.data.ccd_components_rdkit_mol_file,
        )
        if inputs is None:
            with open(self.input_json_path, "r") as f:
                inputs = validate_inference_jobs(json.load(f))
        self.inputs = cast(list[dict[str, Any]], inputs)
        if self.use_template:
            template_mmcif_dir = configs.data.template.prot_template_mmcif_dir
            fetch_remote = configs.data.template.get("fetch_remote", True)
            if not fetch_remote:
                assert template_mmcif_dir is not None and os.path.exists(
                    template_mmcif_dir
                ), (
                    "Inference with template depends on the mmcif directory.\n"
                    "The mmcif directory containing cif files should be placed under $OPENDDE_ROOT_DIR/search_database/mmcif.\n"
                    "You can download it from PDB https://www.wwpdb.org/ftp/pdb-ftp-sites or\n"
                    "refer to scripts/download_opendde_data.sh to download inference dependency files, "
                    "set use_template=false for inference, or set data.template.fetch_remote=true "
                    "to download mmCIF files on demand from PDBe."
                )
            else:
                if template_mmcif_dir:
                    os.makedirs(template_mmcif_dir, exist_ok=True)
            self.online_template_featurizer = TemplateHitFeaturizer(
                mmcif_dir=configs.data.template.prot_template_mmcif_dir,
                template_cache_dir=configs.data.template.prot_template_cache_dir,
                max_hits=4,
                kalign_binary_path=configs.data.template.kalign_binary_path,
                max_template_date="2021-09-30",
                release_dates_path=configs.data.template.release_dates_path,
                obsolete_pdbs_path=configs.data.template.obsolete_pdbs_path,
                _shuffle_top_k_prefiltered=None,
                _max_template_candidates_num=20,
                fetch_remote=fetch_remote,
            )
        else:
            self.online_template_featurizer = None

    def process_one(
        self,
        single_sample_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample_diffusion_config = self.configs.sample_diffusion
        sample_diffusion_dict = sample_diffusion_config.to_dict()
        guidance_config = sample_diffusion_dict.get("guidance") or {}
        need_geometry_features = bool(guidance_config.get("enable", False))
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
            extract_features_for_tfg=need_geometry_features,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type_and_seqs = (
            sample2feat.entity_poly_type_and_seqs
        )  # we include ligand as well
        t1 = time.time()
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                atom_array=atom_array,
                msa_pair_as_unpair=self.msa_pair_as_unpair,
                use_rna_msa=self.use_rna_msa,
            )
            if self.use_msa
            else {}
        )
        template_features = InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=single_sample_dict["sequences"],
            atom_array=atom_array,
            use_template=self.use_template,
            online_template_featurizer=self.online_template_featurizer,
        )
        # Make dummy features for not implemented features
        dummy_feats = []
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data: dict[str, Any] = {}
        data["input_feature_dict"] = feat
        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]
        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )
        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type_and_seqs["entity_poly_type"]})
        time_tracker = {
            "parse": t1 - t0,
            "featurizer": t2 - t1,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], AtomArray | None, str]:
        sample_name = f"job_{index}"
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = sample_name
        data["sample_index"] = index
        return data, atom_array, error_message
