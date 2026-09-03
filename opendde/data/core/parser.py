# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import functools
import gzip
import logging
import random
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

warnings.filterwarnings(
    "ignore", message="Category 'chem_comp_bond' not found. No bonds will be parsed"
)
warnings.filterwarnings(
    "ignore",
    message="The coordinates are missing for some atoms. The fallback coordinates will be used instead",
)
warnings.filterwarnings(
    "ignore",
    message="UserWarning: Missing coordinates for some atoms. Those will be set to nan",
)


import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import networkx as nx
import numpy as np
import pandas as pd
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts
from biotite.structure.io.pdbx import convert as pdbx_convert
from biotite.structure.molecules import get_molecule_indices
from packaging import version
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import io

from opendde.data.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RES_ATOMS_DICT,
    RNA_STD_RESIDUES,
    STD_RESIDUES,
)
from opendde.data.core import ccd
from opendde.data.core.filter import Filter
from opendde.data.tools.rewrite_biotite import _parse_inter_residue_bonds, concatenate
from opendde.data.utils import (
    get_inter_residue_bonds,
    map_annotations_to_atom_indices,
)
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

# PDBX_COVALENT_TYPES was removed in biotite commit
# https://github.com/biotite-dev/biotite/commit/5f584ac3d73650ea7ec657df185f08ff55e8037f
if not hasattr(pdbx_convert, "PDBX_COVALENT_TYPES"):
    pdbx_convert.PDBX_COVALENT_TYPES = list(
        pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE.keys()
    )
# Ignore inter residue metal coordinate bonds in mmcif _struct_conn
pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE.pop("metalc", None)


class MMCIFParser:
    """
    Parsing and extracting information from mmCIF files.
    """

    def __init__(
        self,
        mmcif_file: Optional[Union[str, Path]] = None,
        mmcif_string: Optional[str] = None,
    ) -> None:
        self.mmcif_file = mmcif_file
        self.cif = self._parse(mmcif_file=mmcif_file, mmcif_string=mmcif_string)

    def _parse(
        self,
        mmcif_file: Optional[Union[str, Path]],
        mmcif_string: Optional[str] = None,
    ) -> pdbx.CIFFile:
        if mmcif_file is not None:
            mmcif_file = Path(mmcif_file)
            if mmcif_file.suffix == ".gz":
                with gzip.open(mmcif_file, "rt") as f:
                    cif_file = pdbx.CIFFile.read(f)
            elif mmcif_file.suffix == ".bcif":
                cif_file = pdbx.BinaryCIFFile.read(mmcif_file)
            else:
                with open(mmcif_file, "rt") as f:
                    cif_file = pdbx.CIFFile.read(f)
            return cif_file
        elif mmcif_string is not None:
            cif_file = io.StringIO(mmcif_string)
            cif_file = pdbx.CIFFile.read(cif_file)
            return cif_file
        else:
            raise ValueError("mmcif_file and mmcif_string are both None")

    def get_category_table(self, name: str) -> Union[pd.DataFrame, None]:
        """
        Retrieve a category table from the CIF block and return it as a pandas DataFrame.

        Args:
            name (str): The name of the category to retrieve from the CIF block.

        Returns:
            Union[pd.DataFrame, None]: A pandas DataFrame containing the category data if the category exists,
                                       otherwise None.
        """
        if name not in self.cif.block:
            return None
        category = self.cif.block[name]
        category_dict = {k: column.as_array() for k, column in category.items()}
        return pd.DataFrame(category_dict, dtype=str)

    @functools.cached_property
    def pdb_id(self) -> str:
        """
        Extracts and returns the PDB ID from the CIF block.

        Returns:
            str: The PDB ID in lowercase if present, otherwise an empty string.
        """

        if "entry" not in self.cif.block:
            return ""
        else:
            return self.cif.block["entry"]["id"].as_item().lower()

    def num_assembly_polymer_chains(self, assembly_id: str = "1") -> Optional[int]:
        """
        Calculate the number of polymer chains in a specified assembly.

        Args:
            assembly_id (str): The ID of the assembly to count polymer chains for.
                               Defaults to "1". If "all", counts chains for all assemblies.

        Returns:
            int: The total number of polymer chains in the specified assembly.
                 If the oligomeric count is invalid (e.g., '?'), the function returns None.
        """
        chain_count = 0
        for _assembly_id, _chain_count in zip(
            self.cif.block["pdbx_struct_assembly"]["id"].as_array(),
            self.cif.block["pdbx_struct_assembly"]["oligomeric_count"].as_array(),
        ):
            if assembly_id == "all" or _assembly_id == assembly_id:
                try:
                    chain_count += int(_chain_count)
                except ValueError:
                    # oligomeric_count == '?'.  e.g. 1hya.cif
                    return None
        return chain_count

    @functools.cached_property
    def resolution(self) -> float:
        """
        Get resolution for X-ray and cryoEM.
        Some methods don't have resolution, set as -1.0

        Returns:
            float: resolution (set to -1.0 if not found)
        """
        block = self.cif.block
        resolution_names = [
            "refine.ls_d_res_high",
            "em_3d_reconstruction.resolution",
            "reflns.d_resolution_high",
        ]
        for category_item in resolution_names:
            category, item = category_item.split(".")
            if category in block and item in block[category]:
                try:
                    resolution = block[category][item].as_array(float)[0]
                    # "." will be converted to 0.0, but it is not a valid resolution.
                    if resolution == 0.0:
                        continue
                    return resolution
                except ValueError:
                    # in some cases, resolution_str is "?"
                    continue
        return -1.0

    @functools.cached_property
    def release_date(self) -> str:
        """
        Get first release date.

        Returns:
            str: yyyy-mm-dd
        """

        def _is_valid_date_format(date_string):
            try:
                datetime.strptime(date_string, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        if "pdbx_audit_revision_history" in self.cif.block:
            history = self.cif.block["pdbx_audit_revision_history"]
            # np.str_ is inherit from str, so return is str
            date = history["revision_date"].as_array()[0]
        else:
            # no release date
            date = "9999-12-31"

        valid_date = _is_valid_date_format(date)
        assert valid_date, (
            f"Invalid date format: {date}, it should be yyyy-mm-dd format"
        )
        return date

    @functools.cached_property
    def methods(self) -> list[str]:
        """the methods to get the structure

        most of the time, methods only has one method, such as 'X-RAY DIFFRACTION',
        but about 233 entries have multi methods, such as ['X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION'].

        Allowed Values:
        https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html

        Returns:
            list[str]: such as ['X-RAY DIFFRACTION'], ['ELECTRON MICROSCOPY'], ['SOLUTION NMR', 'THEORETICAL MODEL'],
                ['X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION'], ['ELECTRON MICROSCOPY', 'SOLUTION NMR'], etc.
        """
        if "exptl" not in self.cif.block:
            return []
        else:
            methods = self.cif.block["exptl"]["method"]
            return methods.as_array()

    def get_poly_res_names(
        self, atom_array: Optional[AtomArray] = None
    ) -> dict[str, list[str]]:
        """get 3-letter residue names by combining mmcif._entity_poly_seq and atom_array

        if ref_atom_array is None: keep first altloc residue of the same res_id based in mmcif._entity_poly_seq
        if ref_atom_array is provided: keep same residue of ref_atom_array.

        Returns
            dict[str, list[str]]: label_entity_id --> [res_ids, res_names]
        """
        entity_res_names = {}
        if atom_array is not None:
            # build entity_id -> res_id -> res_name for input atom array
            res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)
            for start in res_starts:
                entity_id = atom_array.label_entity_id[start]
                res_id = atom_array.res_id[start]
                res_name = atom_array.res_name[start]
                if entity_id in entity_res_names:
                    entity_res_names[entity_id][res_id] = res_name
                else:
                    entity_res_names[entity_id] = {res_id: res_name}

        # build reference entity atom array, including missing residues
        entity_poly_seq = self.get_category_table("entity_poly_seq")
        if entity_poly_seq is None:
            return {}

        poly_res_names = {}
        for entity_id, poly_type in self.entity_poly_type.items():
            chain_mask = entity_poly_seq.entity_id == entity_id
            seq_mon_ids = entity_poly_seq.mon_id[chain_mask].to_numpy(dtype=str)

            # replace all MSE to MET in _entity_poly_seq.mon_id
            seq_mon_ids[seq_mon_ids == "MSE"] = "MET"

            seq_nums = entity_poly_seq.num[chain_mask].to_numpy(dtype=int)

            uniq_seq_num = np.unique(seq_nums).size

            if uniq_seq_num == seq_nums.size:
                # no altloc residues
                poly_res_names[entity_id] = seq_mon_ids
                continue

            # filter altloc residues, eg: 181 ALA (altloc A); 181 GLY (altloc B)
            select_mask = np.zeros(len(seq_nums), dtype=bool)
            matching_res_id = seq_nums[0]
            for i, res_id in enumerate(seq_nums):
                if res_id != matching_res_id:
                    continue

                res_name_in_atom_array = entity_res_names.get(entity_id, {}).get(res_id)
                if res_name_in_atom_array is None:
                    # res_name is mssing in atom_array,
                    # keep first altloc residue of the same res_id
                    select_mask[i] = True
                else:
                    # keep match residue to atom_array
                    if res_name_in_atom_array == seq_mon_ids[i]:
                        select_mask[i] = True

                if select_mask[i]:
                    matching_res_id += 1

            new_seq_mon_ids = seq_mon_ids[select_mask]
            new_seq_nums = seq_nums[select_mask]
            assert len(new_seq_nums) == uniq_seq_num, (
                f"seq_nums not match:\n{seq_nums=}\n{new_seq_nums=}\n{seq_mon_ids=}\n{new_seq_mon_ids=}"
            )
            poly_res_names[entity_id] = new_seq_mon_ids
        return poly_res_names

    def get_sequences(self, atom_array=None) -> dict:
        """get sequence by combining mmcif._entity_poly_seq and atom_array

        if ref_atom_array is None: keep first altloc residue of the same res_id based in mmcif._entity_poly_seq
        if ref_atom_array is provided: keep same residue of atom_array.

        Return
            Dict{str:str}: label_entity_id --> canonical_sequence
        """
        sequences = {}
        for entity_id, res_names in self.get_poly_res_names(atom_array).items():
            seq = ccd.res_names_to_sequence(res_names)
            sequences[entity_id] = seq
        return sequences

    @functools.cached_property
    def entity_poly_type(self) -> dict[str, str]:
        """
        Ref: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
        Map entity_id to entity_poly_type.

        Allowed Value:
        · cyclic-pseudo-peptide
        · other
        · peptide nucleic acid
        · polydeoxyribonucleotide
        · polydeoxyribonucleotide/polyribonucleotide hybrid
        · polypeptide(D)
        · polypeptide(L)
        · polyribonucleotide

        Returns:
            Dict: a dict of label_entity_id --> entity_poly_type.
        """
        entity_poly = self.get_category_table("entity_poly")
        if entity_poly is None:
            return {}

        return {i: t for i, t in zip(entity_poly.entity_id, entity_poly.type)}

    @functools.cached_property
    def entity_infos(self) -> dict:
        """
        Retrieves information about entities from the category table "entity".

        Returns:
            dict: A dictionary where each key is an entity ID and the value is another dictionary
                  containing the following keys:
                  - "type": The type of the entity.
                  - "pdbx_description": A description of the entity.
                  - "pdbx_number_of_molecules": The number of molecules for the entity.

        If the "entity" category table is not found, an empty dictionary is returned.
        """
        entity = self.get_category_table("entity")
        if entity is None:
            return {}
        else:
            return {
                _id: {
                    "type": _type,
                    "pdbx_description": _pdbx_description.strip(),
                    "pdbx_number_of_molecules": _pdbx_number_of_molecules,
                }
                for _id, _type, _pdbx_description, _pdbx_number_of_molecules in zip(
                    entity.id,
                    entity.type,
                    entity.pdbx_description,
                    entity.pdbx_number_of_molecules,
                )
            }

    @staticmethod
    def replace_auth_with_label(atom_array: AtomArray) -> AtomArray:
        """
        Replace the author-provided chain ID with the label asym ID in the given AtomArray.

        This function addresses the issue described in https://github.com/biotite-dev/biotite/issues/553.
        It updates the `chain_id` of the `atom_array` to match the `label_asym_id` and resets the ligand
        residue IDs (`res_id`) for chains where the `label_seq_id` is ".". The residue IDs are reset
        sequentially starting from 1 within each chain.

        Args:
            atom_array (AtomArray): The input AtomArray object to be modified.

        Returns:
            AtomArray: The modified AtomArray with updated chain IDs and residue IDs.
        """
        atom_array.chain_id = atom_array.label_asym_id

        # reset ligand res_id
        res_id = atom_array.label_seq_id.astype(object)
        chain_ids = np.unique(atom_array.chain_id)
        for chain_id in chain_ids:
            chain_mask = atom_array.chain_id == chain_id
            chain_res_id = res_id[chain_mask]
            if atom_array.label_seq_id[chain_mask][0] != ".":
                continue
            else:
                res_starts = get_residue_starts(
                    atom_array[chain_mask], add_exclusive_stop=True
                )
                num = 1
                for res_start, res_stop in zip(res_starts[:-1], res_starts[1:]):
                    chain_res_id[res_start:res_stop] = num
                    num += 1
            res_id[chain_mask] = chain_res_id

        atom_array.res_id = res_id.astype(int)
        return atom_array

    def get_structure(
        self,
        altloc: str = "local_largest",
        model: int = 1,
        bond_lenth_threshold: Union[float, None] = 2.4,
    ) -> AtomArray:
        """
        Get an AtomArray created by bioassembly of MMCIF.

        altloc: "local_largest", "first", "all", "A", "B", etc
        model: the model number of the structure.
        bond_lenth_threshold: the threshold of bond length. If None, no filter will be applied.
                              Default is 2.4 Angstroms.

        Returns:
            AtomArray: Biotite AtomArray object created by bioassembly of MMCIF.
        """
        use_author_fields = True
        extra_fields = ["label_asym_id", "label_entity_id", "auth_asym_id"]  # chain
        extra_fields += ["label_seq_id", "auth_seq_id"]  # residue
        atom_site_fields = {
            "occupancy": "occupancy",
            "pdbx_formal_charge": "charge",
            "B_iso_or_equiv": "b_factor",
            "label_alt_id": "label_alt_id",
        }  # atom
        for atom_site_name, alt_name in atom_site_fields.items():
            if atom_site_name in self.cif.block["atom_site"]:
                extra_fields.append(alt_name)

        block = self.cif.block

        extra_fields = set(extra_fields)

        atom_site = block.get("atom_site")
        if atom_site is None:
            raise ValueError("The file does not contain atom_site category table")

        if atom_site.row_count > 1000_000:
            raise ValueError("The file contains more than 1,000,000 atom_site rows")

        biotite_version = version.parse(biotite.__version__)
        if biotite_version >= version.parse("1.2.0"):
            model_atom_site = pdbx_convert._filter_model(atom_site, model)
        else:
            models = atom_site["pdbx_PDB_model_num"].as_array(np.int32)
            pdbx_convert_any = cast(Any, pdbx_convert)
            model_starts = pdbx_convert_any._get_model_starts(models)
            model_count = len(model_starts)

            if model == 0:
                raise ValueError("The model index must not be 0")
            # Negative models mean model indexing starting from last model

            model = model_count + model + 1 if model < 0 else model
            if model > model_count:
                raise ValueError(
                    f"The file has {model_count} models, "
                    f"the given model {model} does not exist"
                )

            model_atom_site = pdbx_convert_any._filter_model(
                atom_site, model_starts, model
            )

        # Any field of the category would work here to get the length
        model_length = model_atom_site.row_count
        atoms = AtomArray(model_length)

        atoms.coord[:, 0] = model_atom_site["Cartn_x"].as_array(np.float32)
        atoms.coord[:, 1] = model_atom_site["Cartn_y"].as_array(np.float32)
        atoms.coord[:, 2] = model_atom_site["Cartn_z"].as_array(np.float32)

        atoms.box = pdbx_convert._get_box(block)
        if atoms.box is not None and np.allclose(atoms.box, 0.0):
            # eg: 2z33, 3izz
            atoms.box = None

        # ensure the box computed from cell is consistent with fract_transf_matrix
        atom_sites = block.get("atom_sites")
        if atom_sites is not None and atoms.box is not None:
            fract_transf_matrix = np.zeros((3, 3))
            fract_transf_vector = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    fract_transf_matrix[i][j] = float(
                        atom_sites[f"fract_transf_matrix[{j + 1}][{i + 1}]"].as_item()
                    )
                fract_transf_vector[i] = float(
                    atom_sites[f"fract_transf_vector[{i + 1}]"].as_item()
                )

        # The below part is the same for both, AtomArray and AtomArrayStack
        pdbx_convert._fill_annotations(
            atoms, model_atom_site, extra_fields, use_author_fields
        )

        bonds = struc.connect_via_residue_names(atoms, inter_residue=False)

        if "struct_conn" in block:
            conn_bonds = _parse_inter_residue_bonds(
                model_atom_site, block["struct_conn"]
            )
            coord1 = atoms.coord[conn_bonds._bonds[:, 0]]
            coord2 = atoms.coord[conn_bonds._bonds[:, 1]]
            dist = np.linalg.norm(coord1 - coord2, axis=1)
            if bond_lenth_threshold is not None:
                conn_bonds._bonds = conn_bonds._bonds[dist < bond_lenth_threshold]
            bonds = bonds.merge(conn_bonds)
        atoms.bonds = bonds

        # inference inter residue bonds missing in struct_conn, based on res_id (auth_seq_id) and auth_asym_id, eg 5mfu
        atom_array = ccd.add_inter_residue_bonds(
            atoms,
            exclude_struct_conn_pairs=True,
            remove_far_inter_chain_pairs=True,
        )

        # use label_seq_id to match seq and structure
        atom_array = self.replace_auth_with_label(atom_array)

        # some pdb have insertion codes, such as 4v5s
        # so we use label_seq_id to iter res
        atom_array = Filter.filter_altloc(atom_array, altloc=altloc)

        # inference inter residue bonds based on new res_id (label_seq_id).
        # the auth_seq_id is not reliable, some are discontinuous (8bvh), some with insertion codes (6ydy).
        atom_array = ccd.add_inter_residue_bonds(
            atom_array, exclude_struct_conn_pairs=True
        )
        return atom_array

    def expand_assembly(
        self, structure: AtomArray, assembly_id: str = "1"
    ) -> AtomArray:
        """
        Expand the given assembly to all chains
        copy from biotite.structure.io.pdbx.get_assembly

        Args:
            structure (AtomArray): The AtomArray of the structure to expand.
            assembly_id (str, optional): The assembly ID in mmCIF file. Defaults to "1".
                                         If assembly_id is "all", all assemblies will be returned.

        Returns:
            AtomArray: The assembly AtomArray.
        """
        block = self.cif.block

        try:
            assembly_gen_category = block["pdbx_struct_assembly_gen"]
        except KeyError:
            logging.info(
                "File has no 'pdbx_struct_assembly_gen' category, return original structure."
            )
            return structure

        try:
            struct_oper_category = block["pdbx_struct_oper_list"]
        except KeyError:
            logging.info(
                "File has no 'pdbx_struct_oper_list' category, return original structure."
            )
            return structure

        assembly_ids = assembly_gen_category["assembly_id"].as_array(str)

        if assembly_id != "all":
            if assembly_id is None:
                assembly_id = assembly_ids[0]
            elif assembly_id not in assembly_ids:
                raise KeyError(f"File has no Assembly ID '{assembly_id}'")

        ### Calculate all possible transformations
        transformations = pdbx_convert._get_transformations(struct_oper_category)

        ### Get transformations and apply them to the affected asym IDs
        assembly = None
        assembly_1_mask = []
        for id, op_expr, asym_id_expr in zip(
            assembly_gen_category["assembly_id"].as_array(str),
            assembly_gen_category["oper_expression"].as_array(str),
            assembly_gen_category["asym_id_list"].as_array(str),
        ):
            # Find the operation expressions for given assembly ID
            # We already asserted that the ID is actually present
            if assembly_id == "all" or id == assembly_id:
                operations = pdbx_convert._parse_operation_expression(op_expr)
                asym_ids = asym_id_expr.split(",")
                # Filter affected asym IDs
                sub_structure = copy.deepcopy(
                    structure[..., np.isin(structure.label_asym_id, asym_ids)]
                )
                sub_assembly = pdbx_convert._apply_transformations(
                    sub_structure, transformations, operations
                )
                # Merge the chains with asym IDs for this operation
                # with chains from other operations
                if assembly is None:
                    assembly = sub_assembly
                else:
                    assembly += sub_assembly

                if id == "1":
                    assembly_1_mask.extend([True] * len(sub_assembly))
                else:
                    assembly_1_mask.extend([False] * len(sub_assembly))

        if assembly is None:
            raise ValueError(f"File has no Assembly ID '{assembly_id}'")

        if assembly_id == "1" or assembly_id == "all":
            assembly.set_annotation("assembly_1", np.array(assembly_1_mask))
        return assembly

    @staticmethod
    def mse_to_met(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI chapter 2.1
        MSE residues are converted to MET residues.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object after converted MSE to MET.
        """
        mse = atom_array.res_name == "MSE"
        se = mse & (atom_array.atom_name == "SE")
        atom_array.atom_name[se] = "SD"
        atom_array.element[se] = "S"
        atom_array.res_name[mse] = "MET"
        atom_array.hetero[mse] = False
        return atom_array

    @staticmethod
    def fix_arginine(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI chapter 2.1
        Arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2).

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object after fix arginine .
        """

        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start_i, stop_i in zip(starts[:-1], starts[1:]):
            if atom_array.res_name[start_i] != "ARG":
                continue
            cd_idx, nh1_idx, nh2_idx = None, None, None
            for idx in range(start_i, stop_i):
                if atom_array.atom_name[idx] == "CD":
                    cd_idx = idx
                if atom_array.atom_name[idx] == "NH1":
                    nh1_idx = idx
                if atom_array.atom_name[idx] == "NH2":
                    nh2_idx = idx
            if cd_idx and nh1_idx and nh2_idx:  # all not None
                cd_nh1 = atom_array.coord[nh1_idx] - atom_array.coord[cd_idx]
                d2_cd_nh1 = np.sum(cd_nh1**2)
                cd_nh2 = atom_array.coord[nh2_idx] - atom_array.coord[cd_idx]
                d2_cd_nh2 = np.sum(cd_nh2**2)
                if d2_cd_nh2 < d2_cd_nh1:
                    atom_array.coord[[nh1_idx, nh2_idx]] = atom_array.coord[
                        [nh2_idx, nh1_idx]
                    ]
        return atom_array

    @staticmethod
    def create_empty_annotation_like(
        source_array: AtomArray, target_array: AtomArray
    ) -> AtomArray:
        """create empty annotation like source_array"""
        # create empty annotation, atom array addition only keep common annotation
        for k, v in source_array._annot.items():
            if k not in target_array._annot:
                target_array._annot[k] = np.zeros(len(target_array), dtype=v.dtype)
        return target_array

    def find_non_ccd_leaving_atoms(
        self,
        atom_array: AtomArray,
        non_std_central_atom_name: str,
        indices_in_atom_array: list[int],
        component: AtomArray,
    ) -> list[str]:
        """ "
        handle mismatch bettween CCD and mmcif
        some residue has bond in non-central atom (without leaving atoms in CCD)
        and its neighbors should be removed like atom_array from mmcif.

        Args:
            atom_array (AtomArray): Biotite AtomArray object from mmcif.
            non_std_central_atom_name (str): non-CCD central atom name.
            indices_in_atom_array (list[int]): indices of equivalent non-CCD central atoms across different chains.
            component (AtomArray): CCD component AtomArray object.

        Returns:
            list[str]: list of atom_name to be removed.
        """
        if len(indices_in_atom_array) == 0:
            return []

        if component.bonds is None:
            return []

        # atom_name not in CCD component, return []
        idx_in_comp = np.where(component.atom_name == non_std_central_atom_name)[0]
        if len(idx_in_comp) == 0:
            return []
        idx_in_comp = idx_in_comp[0]

        # find non-CCD leaving atoms in atom_array
        remove_atom_names = []
        for idx in indices_in_atom_array:
            neighbor_idx = atom_array.bond_map[idx]
            ref_neighbor_idx, types = component.bonds.get_bonds(idx_in_comp)
            # neighbor_atom only bond to central atom in CCD component
            ref_neighbor_idx = [
                i for i in ref_neighbor_idx if len(component.bonds.get_bonds(i)[0]) == 1
            ]
            # atoms not exist in atom_array
            non_exist_mask = ~np.isin(
                component.atom_name[ref_neighbor_idx],
                atom_array.atom_name[neighbor_idx],
            )
            # remove single-bond neigbors not exist in atom_array
            remove_atom_names.append(
                component.atom_name[ref_neighbor_idx][non_exist_mask].tolist()
            )

        # remove atoms based on chain with most leaving atoms
        max_id = int(np.argmax([len(names) for names in remove_atom_names]))
        non_ccd_leaving_atoms = remove_atom_names[max_id]
        return non_ccd_leaving_atoms

    def build_ref_chain_with_atom_array(
        self, atom_array: AtomArray
    ) -> dict[str, dict[int, AtomArray]]:
        """
        build ref chain with atom_array and poly_res_names

        args:
            atom_array (AtomArray): Biotite AtomArray object from mmcif.
        returns:
            entity_residues (dict[str, dict[int, AtomArray]]):
                entity_id (str) -> res_id (int) -> residue (AtomArray)
        """
        # make entity-level annotations to atom indices mapping
        annots_to_indices = map_annotations_to_atom_indices(
            atom_array, annot_keys=["label_entity_id", "res_id", "atom_name"]
        )

        # count inter residue bonds of each potential central atom for removing leaving atoms later
        central_bond_count = Counter()  # (entity_id,res_id,atom_name) -> bond_count

        # build reference entity atom array, including missing residues
        poly_res_names = self.get_poly_res_names(atom_array)
        entity_chains: dict[str, AtomArray] = {}
        for entity_id, poly_type in self.entity_poly_type.items():
            residues = []
            res_ids = []
            for res_id, res_name in enumerate(poly_res_names[entity_id]):
                # keep all leaving atoms, will remove leaving atoms later in this function
                residue = ccd.get_component_atom_array(
                    res_name, keep_leaving_atoms=True, keep_hydrogens=False
                )  # return cached residue:atom_array for same res_name:str
                if residue is None:
                    raise ValueError(f"CCD component {res_name} could not be parsed")
                res_ids.extend([res_id + 1] * len(residue))
                residues.append(residue)
            chain = concatenate(residues)
            chain.res_id = np.array(res_ids)

            res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
            inter_bonds = ccd._connect_inter_residue(chain, res_starts)

            # skip std polymer bonds between residue with non-std polymer bonds
            bond_mask = np.ones(len(inter_bonds._bonds), dtype=bool)
            for b_idx, (atom_i, atom_j, b_type) in enumerate(inter_bonds._bonds):
                same_pos_i = annots_to_indices[
                    (entity_id, chain.res_id[atom_i], chain.atom_name[atom_i])
                ]
                same_pos_j = annots_to_indices[
                    (entity_id, chain.res_id[atom_j], chain.atom_name[atom_j])
                ]

                # When two atoms (same entity/residue) coexist in a chain:
                # 1. Standard polymer bond missing in atom_array suggests possible non-standard bonding
                # 2. Remove corresponding standard bond from inter_bonds
                same_pos_i_chain_id = atom_array.chain_id[same_pos_i].tolist()
                same_pos_j_chain_id = atom_array.chain_id[same_pos_j].tolist()
                for i, ci in zip(same_pos_i, same_pos_i_chain_id):
                    for j, cj in zip(same_pos_j, same_pos_j_chain_id):
                        if ci == cj:
                            bonds = atom_array.bond_map[i]
                            if j not in bonds:
                                bond_mask[b_idx] = False
                                break

                if bond_mask[b_idx]:
                    # keep this bond, add to central_bond_count
                    central_atom_idx = (
                        atom_i if chain.atom_name[atom_i] in ("C", "P") else atom_j
                    )
                    atom_key = (
                        entity_id,
                        chain.res_id[central_atom_idx],
                        chain.atom_name[central_atom_idx],
                    )
                    # use ref chain bond count if no inter bond in atom_array.
                    central_bond_count[atom_key] = 1

            inter_bonds._bonds = inter_bonds._bonds[bond_mask]
            chain.bonds = chain.bonds.merge(inter_bonds)

            chain.hetero[:] = False
            entity_chains[entity_id] = chain

        # remove leaving atoms of residues based on atom_array

        # count inter residue bonds from atom_array for removing leaving atoms later
        inter_residue_bonds = get_inter_residue_bonds(atom_array)
        for i in inter_residue_bonds.flat:
            bonds = atom_array.bond_map[i]
            bond_count = (
                (atom_array.res_id[bonds] != atom_array.res_id[i])
                | (atom_array.chain_id[bonds] != atom_array.chain_id[i])
            ).sum()
            atom_key = (
                atom_array.label_entity_id[i],
                atom_array.res_id[i],
                atom_array.atom_name[i],
            )
            # remove leaving atoms if central atom has inter residue bond in any copy of a entity
            central_bond_count[atom_key] = max(central_bond_count[atom_key], bond_count)

        # remove leaving atoms for each central atom based in atom_array info
        # so the residue in reference chain can be used directly.
        entity_residues: dict[str, dict[int, AtomArray]] = {}
        for entity_id, chain in entity_chains.items():
            keep_atom_mask = np.ones(len(chain), dtype=bool)
            starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
            for start, stop in zip(starts[:-1], starts[1:]):
                res_name = chain.res_name[start]
                remove_atom_names = []
                for i in range(start, stop):
                    central_atom_name = chain.atom_name[i]
                    central_atom_key = (entity_id, chain.res_id[i], central_atom_name)
                    inter_bond_count = central_bond_count[central_atom_key]

                    if inter_bond_count == 0:
                        continue

                    # num of remove leaving groups equals to num of inter residue bonds (inter_bond_count)
                    component = ccd.get_component_atom_array(
                        res_name, keep_leaving_atoms=True
                    )
                    if component is None:
                        break

                    if component.central_to_leaving_groups is None:
                        # The leaving atoms might be labeled wrongly. The residue remains as it is.
                        break

                    # central_to_leaving_groups:dict[str, list[list[str]]], central atom name to leaving atom groups (atom names).
                    if central_atom_name in component.central_to_leaving_groups:
                        leaving_groups = component.central_to_leaving_groups[
                            central_atom_name
                        ]
                        # removed only when there are leaving atoms.
                        if inter_bond_count >= len(leaving_groups):
                            remove_groups = leaving_groups
                        else:
                            # subsample leaving atoms, keep resolved leaving atoms first
                            exist_group = []
                            not_exist_group = []
                            for group in leaving_groups:
                                for leaving_atom_name in group:
                                    atom_idx = annots_to_indices[
                                        (entity_id, chain.res_id[i], leaving_atom_name)
                                    ]
                                    if len(atom_idx) > 0:  # resolved
                                        exist_group.append(group)
                                        break
                                else:
                                    not_exist_group.append(group)
                            if inter_bond_count <= len(not_exist_group):
                                remove_groups = random.sample(
                                    not_exist_group, inter_bond_count
                                )
                            else:
                                remove_groups = not_exist_group + random.sample(
                                    exist_group, inter_bond_count - len(not_exist_group)
                                )
                        names = [name for group in remove_groups for name in group]
                        remove_atom_names.extend(names)

                    else:
                        # may has non-std leaving atom
                        indices_in_atom_array = annots_to_indices[central_atom_key]
                        non_std_leaving_atoms = self.find_non_ccd_leaving_atoms(
                            atom_array=atom_array,
                            non_std_central_atom_name=central_atom_name,
                            indices_in_atom_array=indices_in_atom_array,
                            component=component,
                        )
                        if len(non_std_leaving_atoms) > 0:
                            remove_atom_names.extend(non_std_leaving_atoms)

                # remove leaving atoms of this residue
                remove_mask = np.isin(chain.atom_name[start:stop], remove_atom_names)
                keep_atom_mask[np.arange(start, stop)[remove_mask]] = False

            chain = chain[keep_atom_mask]
            chain = self.create_empty_annotation_like(atom_array, chain)
            entity_residues[entity_id] = {
                r.res_id[0]: r for r in struc.residue_iter(chain)
            }
        return entity_residues

    def make_new_residue(
        self, atom_array, res_start, res_stop, annots_to_indices
    ) -> AtomArray:
        """
        make new residue from atom_array[res_start:res_stop], ref_chain is the reference chain.
        only remove leavning atom when central atom covalent to other residue.
        Args:
            atom_array (AtomArray): Biotite AtomArray object from mmcif.
            res_start (int): start index of residue in atom_array.
            res_stop (int): stop index of residue in atom_array.
            annots_to_indices (dict[tuple, list]): entity_id, res_id, atom_name -> indices in atom_array.
        Returns:
            AtomArray: new residue AtomArray object which removes leaving atoms.
        """
        res_id = atom_array.res_id[res_start]
        res_name = atom_array.res_name[res_start]
        ref_residue = ccd.get_component_atom_array(
            res_name,
            keep_leaving_atoms=True,
            keep_hydrogens=False,
        )
        if ref_residue is None:  # only https://www.rcsb.org/ligand/UNL
            return atom_array[res_start:res_stop]

        if ref_residue.central_to_leaving_groups is None:
            # ambiguous: one leaving group bond to more than one central atom, keep same atoms with PDB entry.
            return atom_array[res_start:res_stop]

        keep_atom_mask = np.ones(len(ref_residue), dtype=bool)

        # remove leavning atoms when covalent to other residue
        chain_id = atom_array.chain_id[res_start]
        old_atom_names = atom_array.atom_name[res_start:res_stop]
        for i, central_atom_name in enumerate(old_atom_names):
            i += res_start
            bonds = atom_array.bond_map[i]
            # count inter residue bonds
            bond_count = sum([1 for b in bonds if (b < res_start or b >= res_stop)])
            if bond_count == 0:
                # central atom is not covalent to other residue, not remove leaving atoms
                continue

            central_atom_key = (
                chain_id,  # here is chain_id, will get only one atom.
                res_id,
                central_atom_name,
            )

            if central_atom_name in ref_residue.central_to_leaving_groups:
                leaving_groups = ref_residue.central_to_leaving_groups[
                    central_atom_name
                ]
                # removed only when there are leaving atoms.
                if bond_count >= len(leaving_groups):
                    remove_groups = leaving_groups
                else:
                    # subsample leaving atoms, remove unresolved leaving atoms first
                    exist_group = []
                    not_exist_group = []
                    for group in leaving_groups:
                        for leaving_atom_name in group:
                            atom_idx = annots_to_indices[
                                (
                                    chain_id,
                                    res_id,
                                    leaving_atom_name,
                                )
                            ]
                            if len(atom_idx) > 0:  # resolved
                                exist_group.append(group)
                                break
                        else:
                            not_exist_group.append(group)

                    # not remove leaving atoms of B and BE, if all leaving atoms is exist in atom_array
                    if central_atom_name in ["B", "BE"]:
                        if not not_exist_group:
                            continue

                    if bond_count <= len(not_exist_group):
                        remove_groups = random.sample(not_exist_group, bond_count)
                    else:
                        remove_groups = not_exist_group + random.sample(
                            exist_group, bond_count - len(not_exist_group)
                        )
            else:
                indices_in_atom_array = annots_to_indices[central_atom_key]
                leaving_atoms = self.find_non_ccd_leaving_atoms(
                    atom_array=atom_array,
                    non_std_central_atom_name=central_atom_name,
                    indices_in_atom_array=indices_in_atom_array,
                    component=ref_residue,
                )
                remove_groups = [leaving_atoms]

            names = [name for group in remove_groups for name in group]
            remove_mask = np.isin(ref_residue.atom_name, names)
            keep_atom_mask &= ~remove_mask

        new_residue = ref_residue[keep_atom_mask]
        new_residue = self.create_empty_annotation_like(atom_array, new_residue)
        return new_residue

    def add_missing_atoms_and_residues(self, atom_array: AtomArray) -> AtomArray:
        """add missing atoms and residues based on CCD and mmcif info.

        Args:
            atom_array (AtomArray): structure with missing residues and atoms, from PDB entry.

        Returns:
            AtomArray: structure added missing residues and atoms (label atom_array.is_resolved as False).
        """
        # build bond map for faster atom_array.bonds.get_bonds()
        bond_map = defaultdict(list)
        for atom_i, atom_j, b_type in atom_array.bonds._bonds:
            bond_map[atom_i].append(atom_j)
            bond_map[atom_j].append(atom_i)
        # used in build_ref_chain_with_atom_array() and make_new_residue()
        atom_array.bond_map = bond_map

        # build reference entity atom array, including missing residues
        entity_residues = self.build_ref_chain_with_atom_array(atom_array)

        # make chain-level annotations to atom indices mapping
        annots_to_indices = map_annotations_to_atom_indices(
            atom_array, annot_keys=["chain_id", "res_id", "atom_name"]
        )

        # build new atom array and copy info from input atom array to it (new_array).
        new_chains = []
        new_global_start = 0
        o2n_amap = {}  # old to new atom map
        chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
        res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for c_start, c_stop in zip(chain_starts[:-1], chain_starts[1:]):
            # get reference chain atom array
            entity_id = atom_array.label_entity_id[c_start]

            ref_chain_residues = entity_residues.get(entity_id)

            chain_residues = []
            c_res_starts = res_starts[(c_start <= res_starts) & (res_starts <= c_stop)]

            # add missing residues
            prev_res_id = 0
            for r_start, r_stop in zip(c_res_starts[:-1], c_res_starts[1:]):
                curr_res_id = atom_array.res_id[r_start]
                if ref_chain_residues is not None and curr_res_id - prev_res_id > 1:
                    # missing residue in head or middle, res_id is 1-based int.
                    for res_id in range(prev_res_id + 1, curr_res_id):
                        new_residue = ref_chain_residues[res_id]
                        chain_residues.append(new_residue)
                        new_global_start += len(new_residue)

                # add missing atoms of existing residue
                if ref_chain_residues is None:
                    new_residue = self.make_new_residue(
                        atom_array, r_start, r_stop, annots_to_indices
                    )
                else:
                    new_residue = ref_chain_residues[curr_res_id]

                # copy residue level info
                residue_fields = ["res_id", "hetero", "label_seq_id", "auth_seq_id"]
                for k in residue_fields:
                    v = atom_array._annot[k][r_start]
                    new_residue._annot[k][:] = v

                # make o2n_amap: old to new atom map
                name_to_index_new = {
                    name: idx for idx, name in enumerate(new_residue.atom_name)
                }
                res_o2n_amap = {}
                res_mismatch_idx = []
                for old_idx in range(r_start, r_stop):
                    old_name = atom_array.atom_name[old_idx]
                    if old_name not in name_to_index_new:
                        # AF3 SI 2.5.4 Filtering
                        # For residues or small molecules with CCD codes, atoms outside of the CCD code’s defined set of atom names are removed.
                        res_mismatch_idx.append(old_idx)
                    else:
                        new_idx = name_to_index_new[old_name]
                        res_o2n_amap[old_idx] = new_global_start + new_idx
                if len(res_o2n_amap) > len(res_mismatch_idx):
                    # Match residues only if more than half of their resolved atoms are matched.
                    # e.g. 1gbt GBS shows 2/12 match, not add to o2n_amap, all atoms are marked as is_resolved=False.
                    o2n_amap.update(res_o2n_amap)

                chain_residues.append(new_residue)

                prev_res_id = curr_res_id
                new_global_start += len(new_residue)

            # missing residue in tail
            if ref_chain_residues is not None:
                last_res_id = max(ref_chain_residues.keys())
                for res_id in range(curr_res_id + 1, last_res_id + 1):
                    new_residue = ref_chain_residues[res_id]
                    chain_residues.append(new_residue)
                    new_global_start += len(new_residue)

            chain_array = concatenate(chain_residues)

            # copy chain level info
            chain_fields = [
                "chain_id",
                "label_asym_id",
                "label_entity_id",
                "auth_asym_id",
                # "asym_id_int",
                # "entity_id_int",
                # "sym_id_int",
            ]
            for k in chain_fields:
                chain_array._annot[k][:] = atom_array._annot[k][c_start]

            new_chains.append(chain_array)

        new_array = concatenate(new_chains)

        # copy atom level info
        old_idx = list(o2n_amap.keys())
        new_idx = list(o2n_amap.values())
        atom_fields = ["b_factor", "occupancy", "charge", "label_alt_id"]
        for k in atom_fields:
            if k not in atom_array._annot:
                continue
            new_array._annot[k][new_idx] = atom_array._annot[k][old_idx]

        # add is_resolved annotation
        is_resolved = np.zeros(len(new_array), dtype=bool)
        is_resolved[new_idx] = True
        new_array.set_annotation("is_resolved", is_resolved)

        # copy coord
        new_array.coord[:] = 0.0
        new_array.coord[new_idx] = atom_array.coord[old_idx]
        # copy bonds
        old_bonds = atom_array.bonds.as_array()  # *n x 3* np.ndarray (i,j,bond_type)

        # some non-leaving atoms are not in the new_array for atom name mismatch, e.g. 4msw TYF
        # only keep bonds of matching atoms
        old_bonds = old_bonds[
            np.isin(old_bonds[:, 0], old_idx) & np.isin(old_bonds[:, 1], old_idx)
        ]

        old_bonds[:, 0] = [o2n_amap[i] for i in old_bonds[:, 0]]
        old_bonds[:, 1] = [o2n_amap[i] for i in old_bonds[:, 1]]
        new_bonds = struc.BondList(len(new_array), old_bonds)
        if new_array.bonds is None:
            new_array.bonds = new_bonds
        else:
            new_array.bonds = new_array.bonds.merge(new_bonds)

        del atom_array.bond_map

        # add peptide bonds and nucleic acid bonds based on CCD type
        new_array = ccd.add_inter_residue_bonds(
            new_array, exclude_struct_conn_pairs=True, remove_far_inter_chain_pairs=True
        )
        return new_array


class AddAtomArrayAnnot(object):
    """
    The methods in this class are all designed to add annotations to an AtomArray
    without altering the information in the original AtomArray.
    """

    @staticmethod
    def add_token_annotations(
        atom_array: AtomArray, entity_poly_type: dict[str, str]
    ) -> AtomArray:
        """
        Add the standard token-level annotations in a fixed order.

        Centralizes the annotation sequence shared by every parsing pipeline
        so all call sites stay consistent.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.
            entity_poly_type (dict[str, str]): label_entity_id -> poly type.

        Returns:
            AtomArray: with mol_type, atom/centre/distogram/plddt masks,
                canonical sequence resname, tokatom idx and modified-res mask.
        """
        atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        return atom_array

    @staticmethod
    def add_token_mol_type(
        atom_array: AtomArray, sequences: dict[str, str]
    ) -> AtomArray:
        """
        Add molecule types in atom_arry.mol_type based on ccd pdbx_type.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.
            sequences (dict[str, str]): A dict of label_entity_id --> canonical_sequence

        Return
            AtomArray: add atom_arry.mol_type = "protein" | "rna" | "dna" | "ligand"
        """
        mol_types = np.zeros(len(atom_array), dtype="U7")
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            entity_id = atom_array.label_entity_id[start]
            if entity_id not in sequences:
                # non-poly is ligand
                mol_types[start:stop] = "ligand"
                continue
            res_name = atom_array.res_name[start]

            mol_types[start:stop] = ccd.get_mol_type(res_name)

        atom_array.set_annotation("mol_type", mol_types)
        return atom_array

    @staticmethod
    def add_atom_mol_type_mask(atom_array: AtomArray) -> AtomArray:
        """
        Mask indicates is_protein / rna / dna / ligand.
        It is atom-level which is different with paper (token-level).
        The type of each atom is determined based on the most frequently
        occurring type in the chain to which it belongs.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with
                       "is_ligand", "is_dna", "is_rna", "is_protein" annotation added.
        """
        # it should be called after mmcif_parser.add_token_mol_type
        chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
        chain_mol_type = []
        for start, end in zip(chain_starts[:-1], chain_starts[1:]):
            mol_types = atom_array.mol_type[start:end]
            mol_type_count = Counter(mol_types)
            sorted_by_key = sorted(mol_type_count.items(), key=lambda x: x[0])
            sorted_by_value = sorted(sorted_by_key, key=lambda x: x[1])
            most_freq_mol_type = sorted_by_value[-1][0]
            chain_mol_type.extend([most_freq_mol_type] * (end - start))

        atom_array.set_annotation(
            "chain_mol_type", np.array(chain_mol_type, dtype=object)
        )

        for type_str in ["ligand", "dna", "rna", "protein"]:
            mask = (atom_array.chain_mol_type == type_str).astype(int)
            atom_array.set_annotation(f"is_{type_str}", mask)
        return atom_array

    @staticmethod
    def add_modified_res_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 5.9.3

        Determine if an atom belongs to a modified residue,
        which is used to calculate the Modified Residue Scores in sample ranking:
        Modified residue scores are ranked according to the average pLDDT of the modified residue.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with
                       "modified_res_mask" annotation added.
        """
        modified_res_mask = []
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_name = atom_array.res_name[start]
            mol_type = atom_array.mol_type[start]
            res_atom_nums = stop - start
            if res_name not in STD_RESIDUES and mol_type != "ligand":
                modified_res_mask.extend([1] * res_atom_nums)
            else:
                modified_res_mask.extend([0] * res_atom_nums)
        atom_array.set_annotation("modified_res_mask", modified_res_mask)
        return atom_array

    @staticmethod
    def add_centre_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 2.6
            • A standard amino acid residue (Table 13) is represented as a single token.
            • A standard nucleotide residue (Table 13) is represented as a single token.
            • A modified amino acid or nucleotide residue is tokenized per-atom (i.e. N tokens for an N-atom residue)
            • All ligands are tokenized per-atom
        For each token we also designate a token centre atom, used in various places below:
            • Cα for standard amino acids
            • C1′ for standard nucleotides
            • For other cases take the first and only atom as they are tokenized per-atom.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "centre_atom_mask" annotation added.
        """
        res_name = list(STD_RESIDUES.keys())
        std_res = np.isin(atom_array.res_name, res_name) & (
            atom_array.mol_type != "ligand"
        )
        prot_res = np.char.str_len(atom_array.res_name) == 3
        prot_centre_atom = prot_res & (atom_array.atom_name == "CA")
        nuc_centre_atom = (~prot_res) & (atom_array.atom_name == r"C1'")
        not_std_res = ~std_res
        centre_atom_mask = (
            std_res & (prot_centre_atom | nuc_centre_atom)
        ) | not_std_res
        centre_atom_mask = centre_atom_mask.astype(int)
        atom_array.set_annotation("centre_atom_mask", centre_atom_mask)
        return atom_array

    @staticmethod
    def add_distogram_rep_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 4.4
        the representative atom mask for each token for distogram head
        • Cβ for protein residues (Cα for glycine),
        • C4 for purines and C2 for pyrimidines.
        • All ligands already have a single atom per token.

        Due to the lack of explanation regarding the handling of "N" and "DN" in the article,
        it is impossible to determine the representative atom based on whether it is a purine or pyrimidine.
        Therefore, C1' is chosen as the representative atom for both "N" and "DN".

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "distogram_rep_atom_mask" annotation added.
        """
        std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
            atom_array.mol_type != "ligand"
        )

        # for protein std res
        std_prot_res = std_res & (np.char.str_len(atom_array.res_name) == 3)
        gly = atom_array.res_name == "GLY"
        prot_cb = std_prot_res & (~gly) & (atom_array.atom_name == "CB")
        prot_gly_ca = gly & (atom_array.atom_name == "CA")

        # for nucleotide std res
        purines_c4 = np.isin(atom_array.res_name, ["DA", "DG", "A", "G"]) & (
            atom_array.atom_name == "C4"
        )
        pyrimidines_c2 = np.isin(atom_array.res_name, ["DC", "DT", "C", "U"]) & (
            atom_array.atom_name == "C2"
        )

        # for nucleotide unk res
        unk_nuc = np.isin(atom_array.res_name, ["DN", "N"]) & (
            atom_array.atom_name == r"C1'"
        )

        distogram_rep_atom_mask = (
            prot_cb | prot_gly_ca | purines_c4 | pyrimidines_c2 | unk_nuc
        ) | (~std_res)
        distogram_rep_atom_mask = distogram_rep_atom_mask.astype(int)

        atom_array.set_annotation("distogram_rep_atom_mask", distogram_rep_atom_mask)

        if np.sum(atom_array.distogram_rep_atom_mask) != np.sum(
            atom_array.centre_atom_mask
        ):
            # some residue has no distogram_rep_atom
            starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
            for start, stop in zip(starts[:-1], starts[1:]):
                if ~np.any(atom_array.distogram_rep_atom_mask[start:stop]):
                    logging.warning(
                        "This residue has no distogram_rep_atom, use the first atom: "
                        "res_chain: %s, res_id: %s, res_name: %s",
                        atom_array.chain_id[start],
                        atom_array.res_id[start],
                        atom_array.res_name[start],
                    )
                    atom_array.distogram_rep_atom_mask[start] = 1
        return atom_array

    @staticmethod
    def add_plddt_m_rep_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 4.3.1
        the representative atom for pLDDT confidence
        • Atoms such that the distance in the ground truth between atom l and atom m is less than 15 Å
            if m is a protein atom or less than 30 Å if m is a nucleic acid atom.
        • Only atoms in polymer chains.
        • One atom per token - Cα for standard protein residues
            and C1′ for standard nucleic acid residues.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "plddt_m_rep_atom_mask" annotation added.
        """
        std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
            atom_array.mol_type != "ligand"
        )
        ca_or_c1 = (atom_array.atom_name == "CA") | (atom_array.atom_name == r"C1'")
        plddt_m_rep_atom_mask = (std_res & ca_or_c1).astype(int)
        atom_array.set_annotation("plddt_m_rep_atom_mask", plddt_m_rep_atom_mask)
        return atom_array

    @staticmethod
    def add_ref_space_uid(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 2.8 Table 5
        Numerical encoding of the chain id and residue index associated with this reference conformer.
        Each (chain id, residue index) tuple is assigned an integer on first appearance.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "ref_space_uid" annotation added.
        """
        # [N_atom, 2]
        chain_res_id = np.vstack((atom_array.asym_id_int, atom_array.res_id)).T
        unique_id = np.unique(chain_res_id, axis=0)

        mapping_dict = {}
        for idx, chain_res_id_pair in enumerate(unique_id):
            asym_id_int, res_id = chain_res_id_pair
            mapping_dict[(asym_id_int, res_id)] = idx

        ref_space_uid = [
            mapping_dict[(asym_id_int, res_id)] for asym_id_int, res_id in chain_res_id
        ]
        atom_array.set_annotation("ref_space_uid", ref_space_uid)
        return atom_array

    @staticmethod
    def add_cano_seq_resname(atom_array: AtomArray) -> AtomArray:
        """
        Assign to each atom the three-letter residue name (resname)
        corresponding to its place in the canonical sequences.
        Non-standard residues are mapped to standard ones.
        Residues that cannot be mapped to standard residues and ligands are all labeled as "UNK".

        Note: Some CCD Codes in the canonical sequence are mapped to three letters. It is labeled as one "UNK".

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "cano_seq_resname" annotation added.
        """
        cano_seq_resname = []
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_atom_nums = stop - start
            mol_type = atom_array.mol_type[start]
            resname = atom_array.res_name[start]

            one_letter_code = ccd.get_one_letter_code(resname)
            if one_letter_code is None or len(one_letter_code) != 1:
                # Some non-standard residues cannot be mapped back to one standard residue.
                one_letter_code = "X" if mol_type == "protein" else "N"

            if mol_type == "protein":
                res_name_in_cano_seq = PROT_STD_RESIDUES_ONE_TO_THREE.get(
                    one_letter_code, "UNK"
                )
            elif mol_type == "dna":
                res_name_in_cano_seq = "D" + one_letter_code
                if res_name_in_cano_seq not in DNA_STD_RESIDUES:
                    res_name_in_cano_seq = "DN"
            elif mol_type == "rna":
                res_name_in_cano_seq = one_letter_code
                if res_name_in_cano_seq not in RNA_STD_RESIDUES:
                    res_name_in_cano_seq = "N"
            else:
                # some molecules attached to a polymer like ATP-RNA. e.g.
                res_name_in_cano_seq = "UNK"

            cano_seq_resname.extend([res_name_in_cano_seq] * res_atom_nums)

        atom_array.set_annotation("cano_seq_resname", cano_seq_resname)
        return atom_array

    @staticmethod
    def remove_bonds_between_polymer_chains(
        atom_array: AtomArray, entity_poly_type: dict[str, str]
    ) -> struc.BondList:
        """
        Remove bonds between polymer chains based on entity_poly_type.
        Only remove bonds between different polymer chains.
        The primary purpose is to enable chains connected by disulfide bonds
        to be separated for chain permutation.

        Args:
            atom_array (AtomArray): Biotite AtomArray object
            entity_poly_type (dict[str, str]): entity_id to poly_type

        Returns:
            BondList: Biotite BondList object (copy) with bonds between polymer chains removed
        """
        bonds = atom_array.bonds.copy()
        polymer_mask = np.isin(
            atom_array.label_entity_id, list(entity_poly_type.keys())
        )
        i = bonds._bonds[:, 0]
        j = bonds._bonds[:, 1]
        pp_bond_mask = polymer_mask[i] & polymer_mask[j]
        diff_chain_mask = atom_array.chain_id[i] != atom_array.chain_id[j]
        pp_bond_mask = pp_bond_mask & diff_chain_mask
        bonds._bonds = bonds._bonds[~pp_bond_mask]

        # post-process after modified bonds manually
        # due to the extraction of bonds using a mask,
        # the lower one of the two atom indices is still in the first
        bonds._remove_redundant_bonds()
        bonds._max_bonds_per_atom = bonds._get_max_bonds_per_atom()
        return bonds

    @staticmethod
    def find_equiv_mol_and_assign_ids(
        atom_array: AtomArray,
        entity_poly_type: Optional[dict[str, str]] = None,
        pdb_id: Optional[str] = None,
    ) -> AtomArray:
        """
        Assign a unique integer to each molecule in the structure.
        All atoms connected by covalent bonds are considered as a molecule, with unique mol_id (int).
        different copies of same molecule will assign same entity_mol_id (int).
        for each mol, assign mol_atom_index starting from 0.

        Args:
            atom_array (AtomArray): Biotite AtomArray object
            entity_poly_type (Optional[dict[str, str]]): label_entity_id to entity.poly_type.
                              Defaults to None.

        Returns:
            AtomArray: Biotite AtomArray object with new annotations
            - mol_id: atoms with covalent bonds connected, 0-based int
            - entity_mol_id: equivalent molecules will assign same entity_mol_id, 0-based int
            - mol_residue_index: mol_atom_index for each mol, 0-based int
        """
        # Re-assign mol_id to AtomArray after break asym bonds
        # Only use resolved atoms to find molecules.
        # because excessive atomic quantities can trigger recursion limits. (e.g. 6ydp)
        if hasattr(atom_array, "is_resolved"):
            valid_mask = atom_array.is_resolved
        else:
            valid_mask = np.ones(len(atom_array), dtype=bool)

        if entity_poly_type is None:
            mol_indices: list[np.ndarray] = get_molecule_indices(atom_array[valid_mask])
        else:
            bonds_filtered = AddAtomArrayAnnot.remove_bonds_between_polymer_chains(
                atom_array[valid_mask], entity_poly_type
            )
            mol_indices: list[np.ndarray] = get_molecule_indices(bonds_filtered)

        # assign mol_id
        mol_ids = np.array([-1] * len(atom_array), dtype=int)
        chain_graph = nx.Graph()
        chain_graph.add_nodes_from(np.unique(atom_array.chain_id))
        for atom_indices in mol_indices:
            chain_ids_in_mol = np.unique(atom_array.chain_id[valid_mask][atom_indices])
            for i in zip(chain_ids_in_mol[:-1], chain_ids_in_mol[1:]):
                chain_graph.add_edge(i[0], i[1])

        for mol_id, subgraph in enumerate(nx.connected_components(chain_graph)):
            atom_indices = np.where(np.isin(atom_array.chain_id, list(subgraph)))[0]
            mol_ids[atom_indices] = mol_id
        atom_array.set_annotation("mol_id", mol_ids)
        assert ~np.isin(-1, atom_array.mol_id), "Some mol_id is not assigned."

        # assign entity_mol_id
        mol_id_to_atom_name = {}
        entity_mol_dict = defaultdict(list)
        for mol_id in np.unique(atom_array.mol_id):
            mol_mask = atom_array.mol_id == mol_id
            _, chain_starts = np.unique(
                atom_array.chain_id[mol_mask], return_index=True
            )
            entity_ids = atom_array.label_entity_id[mol_mask][chain_starts].tolist()
            entity_mol_dict[tuple(sorted(entity_ids))].append(mol_id)
            mol_id_to_atom_name[mol_id] = atom_array.atom_name[mol_mask]

        entity_mol_id_num = 0
        mol_id_to_entity_mol_ids = {}
        for entity_ids, mol_ids in entity_mol_dict.items():
            checked_mol_id = []
            for mol_id in mol_ids:
                mol_atom_name = mol_id_to_atom_name[mol_id]
                if checked_mol_id:
                    for ref_mol_id in checked_mol_id:
                        ref_atom_name = mol_id_to_atom_name[ref_mol_id]
                        if len(mol_atom_name) == len(ref_atom_name):
                            if np.all(ref_atom_name == mol_atom_name):
                                mol_id_to_entity_mol_ids[mol_id] = (
                                    mol_id_to_entity_mol_ids[ref_mol_id]
                                )
                                break
                            else:
                                warning_msg = (
                                    "Two mols have same entity_ids, but diff atom name:\n"
                                    f"ref_atom_name={ref_atom_name[:5]}\n"
                                    f"atom_name={mol_atom_name[:5]}"
                                )
                                if pdb_id is not None:
                                    warning_msg = f"PDB ID: {pdb_id} - " + warning_msg
                                logging.warning(warning_msg)
                                continue
                        else:
                            warning_msg = (
                                "Two mols have same entity_ids, but diff atom num:\n"
                                f"ref_atom_num={len(ref_atom_name)}\n"
                                f"atom_num={len(mol_atom_name)}"
                            )
                            if pdb_id is not None:
                                warning_msg = f"PDB ID: {pdb_id} - " + warning_msg
                                logging.warning(warning_msg)
                            continue
                    else:
                        # Same mol not be found, create a new entity_mol_id
                        entity_mol_id_num += 1
                        mol_id_to_entity_mol_ids[mol_id] = entity_mol_id_num

                else:
                    # First mol for this entity_ids
                    mol_id_to_entity_mol_ids[mol_id] = entity_mol_id_num

                checked_mol_id.append(mol_id)

            # Add 1 to entity_mol_id of new group
            entity_mol_id_num += 1

        entity_mol_ids = np.array(
            [mol_id_to_entity_mol_ids[mol_id] for mol_id in atom_array.mol_id],
            dtype=np.int32,
        )
        atom_array.set_annotation("entity_mol_id", entity_mol_ids)

        # assign mol_atom_index
        # e.g. mol_id = [1, 1, 2, 2, 1, 3] -> mol_atom_index = [0, 1, 0, 1, 2, 0]
        unique, indices = np.unique(atom_array.mol_id, return_inverse=True)
        counts = np.bincount(indices)

        mol_atom_index = np.zeros_like(atom_array.mol_id)
        for i in range(len(unique)):
            mol_atom_index[indices == i] = np.arange(counts[i])
        atom_array.set_annotation("mol_atom_index", mol_atom_index)
        return atom_array

    @staticmethod
    def add_tokatom_idx(atom_array: AtomArray) -> AtomArray:
        """
        Add a tokatom_idx corresponding to the residue and atom name for each atom.
        For non-standard residues or ligands, the tokatom_idx should be set to 0.

        Parameters:
        atom_array (AtomArray): The AtomArray object to which the annotation will be added.

        Returns:
        AtomArray: The AtomArray object with the 'tokatom_idx' annotation added.
        """
        # pre-defined atom name order for tokatom_idx
        tokatom_idx_list = []
        for atom in atom_array:
            atom_name_position = RES_ATOMS_DICT.get(atom.res_name, None)
            if atom.mol_type == "ligand" or atom_name_position is None:
                tokatom_idx = 0
            else:
                tokatom_idx = atom_name_position[atom.atom_name]
            tokatom_idx_list.append(tokatom_idx)
        atom_array.set_annotation("tokatom_idx", tokatom_idx_list)
        return atom_array

    @staticmethod
    def unique_chain_and_add_ids(atom_array: AtomArray) -> AtomArray:
        """
        Unique chain ID and add asym_id, entity_id, sym_id.
        Adds a number to the chain ID to make chain IDs in the assembly unique.
        Example: [A, B, A, B, C] -> [A, B, A.1, B.1, C]

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object with new annotations:
                - asym_id_int: np.array(int)
                - entity_id_int: np.array(int)
                - sym_id_int: np.array(int)
        """
        chain_ids = np.zeros(len(atom_array), dtype="<U16")
        chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)

        chain_counter = Counter()
        for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
            ori_chain_id = atom_array.chain_id[start]
            cnt = chain_counter[ori_chain_id]
            if cnt == 0:
                new_chain_id = ori_chain_id
            else:
                new_chain_id = f"{ori_chain_id}.{chain_counter[ori_chain_id]}"

            chain_ids[start:stop] = new_chain_id
            chain_counter[ori_chain_id] += 1

        assert "" not in chain_ids
        # reset chain id
        atom_array.del_annotation("chain_id")
        atom_array.set_annotation("chain_id", chain_ids)

        entity_id_uniq = np.sort(np.unique(atom_array.label_entity_id))
        entity_id_dict = {e: i for i, e in enumerate(entity_id_uniq)}
        asym_ids = np.zeros(len(atom_array), dtype=int)
        entity_ids = np.zeros(len(atom_array), dtype=int)
        sym_ids = np.zeros(len(atom_array), dtype=int)
        counter = Counter()
        start_indices = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
        for i in range(len(start_indices) - 1):
            start_i = start_indices[i]
            stop_i = start_indices[i + 1]
            asym_ids[start_i:stop_i] = i

            entity_id = atom_array.label_entity_id[start_i]
            entity_ids[start_i:stop_i] = entity_id_dict[entity_id]

            sym_ids[start_i:stop_i] = counter[entity_id]
            counter[entity_id] += 1

        atom_array.set_annotation("asym_id_int", asym_ids)
        atom_array.set_annotation("entity_id_int", entity_ids)
        atom_array.set_annotation("sym_id_int", sym_ids)
        return atom_array
