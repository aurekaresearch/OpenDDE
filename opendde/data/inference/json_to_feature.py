# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import math
import re
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray

from opendde.data.core.featurizer import Featurizer
from opendde.data.core.geometry_featurizer import GeometryFeaturizer
from opendde.data.core.parser import AddAtomArrayAnnot
from opendde.data.inference.json_parser import (
    add_entity_atom_array,
    remove_leaving_atoms,
)
from opendde.data.tokenizer import AtomArrayTokenizer, TokenArray
from opendde.data.utils import int_to_letters
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


class SampleDictToFeatures:
    def __init__(
        self, single_sample_dict: dict[str, Any], extract_features_for_tfg: bool = False
    ) -> None:
        self.extract_features_for_tfg = extract_features_for_tfg
        self.single_sample_dict = single_sample_dict
        self._handle_constraint_field(single_sample_dict, extract_features_for_tfg)
        self.input_dict = add_entity_atom_array(single_sample_dict)
        self.entity_poly_type_and_seqs = self.get_entity_poly_type_and_seqs()
        self.entity_poly_type = self.entity_poly_type_and_seqs["entity_poly_type"]
        self.entity_to_sequences = self.entity_poly_type_and_seqs["entity_to_sequences"]
        self.contact_restraints = self.parse_contact_restraints()

    def get_entity_poly_type_and_seqs(self) -> dict[str, dict[str, str]]:
        """
        Get the entity type for each entity.

        Allowed Value for "_entity_poly.type":
        · cyclic-pseudo-peptide
        · other
        · peptide nucleic acid
        · polydeoxyribonucleotide
        · polydeoxyribonucleotide/polyribonucleotide hybrid
        · polypeptide(D)
        · polypeptide(L)
        · polyribonucleotide

        Returns:
            dict[str, dict[str, str]]: entity polymer types and source sequences.
        """
        entity_type_mapping_dict = {
            "proteinChain": "polypeptide(L)",
            "dnaSequence": "polydeoxyribonucleotide",
            "rnaSequence": "polyribonucleotide",
            "ligand": "non-polymer",
            "ion": "non-polymer",
        }
        entity_poly_type = {}
        entity_to_sequences = {}
        for idx, type2entity_dict in enumerate(self.input_dict["sequences"]):
            assert len(type2entity_dict) == 1, "Only one entity type is allowed."
            for entity_type, entity in type2entity_dict.items():
                if "sequence" in entity:
                    assert entity_type in [
                        "proteinChain",
                        "dnaSequence",
                        "rnaSequence",
                        "ligand",
                        "ion",
                    ], (
                        'The "sequences" field accepts only these entity types: ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"].'
                    )
                    entity_poly_type[str(idx + 1)] = entity_type_mapping_dict[
                        entity_type
                    ]
                    entity_to_sequences[str(idx + 1)] = entity["sequence"]
        return {
            "entity_poly_type": entity_poly_type,
            "entity_to_sequences": entity_to_sequences,
        }

    def build_full_atom_array(self) -> AtomArray:
        """
        By assembling the AtomArray of each entity, a complete AtomArray is created.

        Returns:
            AtomArray: Biotite Atom array.
        """
        used_chain_ids = set()
        for seq_idx, type2entity_dict in enumerate(self.input_dict["sequences"]):
            for entity in type2entity_dict.values():
                if entity.get("id"):
                    ids = entity["id"]
                    if not isinstance(ids, list) or not all(
                        isinstance(x, str) for x in ids
                    ):
                        raise ValueError(
                            f'Invalid "id" field in sequences[{seq_idx}]: expecting list[str].'
                        )
                    if len(ids) != entity["count"]:
                        raise ValueError(
                            f'Invalid "id" field in sequences[{seq_idx}]: len(id)({len(ids)}) != count({entity["count"]}).'
                        )
                    if len(set(ids)) != len(ids):
                        raise ValueError(
                            f'Invalid "id" field in sequences[{seq_idx}]: duplicated chain IDs in the same entity.'
                        )
                    duplicated = set(ids) & used_chain_ids
                    if duplicated:
                        raise ValueError(
                            f'Invalid "id" field in sequences[{seq_idx}]: duplicated chain IDs across entities: {sorted(duplicated)}.'
                        )
                    used_chain_ids.update(ids)

        atom_array = None
        asym_chain_idx = 0
        for idx, type2entity_dict in enumerate(self.input_dict["sequences"]):
            for entity_type, entity in type2entity_dict.items():
                entity_id = str(idx + 1)

                entity_atom_array = None
                ids = entity.get("id")

                for asym_chain_count in range(1, entity["count"] + 1):
                    if ids:
                        asym_id_str = str(ids[asym_chain_count - 1])
                    else:
                        while True:
                            candidate = int_to_letters(asym_chain_idx + 1)
                            if candidate not in used_chain_ids:
                                asym_id_str = candidate
                                used_chain_ids.add(asym_id_str)
                                asym_chain_idx += 1
                                break
                            asym_chain_idx += 1

                    asym_chain = copy.deepcopy(entity["atom_array"])
                    chain_id = [asym_id_str] * len(asym_chain)
                    copy_id = [asym_chain_count] * len(asym_chain)
                    asym_chain.set_annotation("label_asym_id", chain_id)
                    asym_chain.set_annotation("auth_asym_id", chain_id)
                    asym_chain.set_annotation("chain_id", chain_id)
                    asym_chain.set_annotation("label_seq_id", asym_chain.res_id)
                    asym_chain.set_annotation("copy_id", copy_id)
                    if entity_atom_array is None:
                        entity_atom_array = asym_chain
                    else:
                        entity_atom_array += asym_chain

                if entity_atom_array is None:
                    raise ValueError(
                        f'Invalid "count" field in sequences[{idx}]: expecting a positive integer.'
                    )

                entity_atom_array.set_annotation(
                    "label_entity_id", [entity_id] * len(entity_atom_array)
                )

                if entity_type in ["proteinChain", "dnaSequence", "rnaSequence"]:
                    entity_atom_array.hetero[:] = False
                else:
                    entity_atom_array.hetero[:] = True

                if atom_array is None:
                    atom_array = entity_atom_array
                else:
                    atom_array += entity_atom_array
        if atom_array is None:
            raise ValueError('Input JSON must contain at least one "sequences" entity.')
        return atom_array

    @staticmethod
    def _handle_constraint_field(
        single_sample_dict: dict[str, Any], extract_features_for_tfg: bool
    ) -> None:
        """Warn about unsupported constraint keys; hint when contact needs TFG."""
        if "constraint" not in single_sample_dict:
            return
        constraint = single_sample_dict["constraint"]
        if constraint is None:
            return
        if not isinstance(constraint, dict):
            raise ValueError(
                "The 'constraint' field must be an object with the keys contact, "
                "movable_chains and epitope."
            )
        supported = {"contact", "movable_chains", "epitope"}
        if "epitope" in constraint:
            if constraint["epitope"] is None:
                raise ValueError(
                    "constraint.epitope is null; give an object or remove the key."
                )
            if "contact" in constraint:
                raise ValueError(
                    "constraint.epitope and constraint.contact cannot be combined in one request."
                )
            if not extract_features_for_tfg:
                raise ValueError(
                    "constraint.epitope is applied only through TFG guidance; "
                    "pass --use_tfg_guidance true or remove constraint.epitope."
                )
        unknown = sorted(set(constraint.keys()) - supported)
        if unknown:
            # An unknown key is an error: a misspelled key would otherwise run as an unguided prediction under a guided label
            raise ValueError(
                "Unsupported constraint field(s): %s. Supported: contact, movable_chains, epitope."
                % ", ".join(unknown)
            )
        contacts = constraint.get("contact")
        if contacts is not None and not isinstance(contacts, list):
            raise ValueError(
                f"constraint.contact must be a list; got {type(contacts).__name__}."
            )
        if contacts and not extract_features_for_tfg:
            logger.warning(
                "constraint.contact is present but TFG guidance is disabled. "
                "Enable with --use_tfg_guidance true to apply epitope distance restraints."
            )

    def parse_contact_restraints(self) -> list[dict[str, Any]]:
        """Parse constraint.contact into normalized restraint dicts.

        Schema mirrors covalent_bonds entity/copy/position/atom addressing
        (entity1/2 or left_*/right_* aliases) plus min_distance/max_distance in Å.
        Does not write covalent bonds.
        """
        constraint = self.single_sample_dict.get("constraint")
        if not isinstance(constraint, dict):
            return []
        contacts = constraint.get("contact") or []
        if not isinstance(contacts, list):
            raise ValueError(
                f"constraint.contact must be a list; got {type(contacts).__name__}."
            )

        parsed: list[dict[str, Any]] = []
        for contact_idx, bond_info_dict in enumerate(contacts):
            if not isinstance(bond_info_dict, dict):
                raise ValueError(
                    f"constraint.contact[{contact_idx}] must be an object."
                )
            sides: list[dict[str, Any]] = []
            for idx, side in enumerate(["left", "right"]):
                entity_raw = bond_info_dict.get(
                    f"{side}_entity", bond_info_dict.get(f"entity{idx + 1}")
                )
                if entity_raw is None:
                    raise ValueError(
                        f"constraint.contact[{contact_idx}] missing "
                        f"entity{idx + 1}/{side}_entity."
                    )
                entity_id = self._strict_int(
                    entity_raw, f"constraint.contact[{contact_idx}].entity{idx + 1}"
                )
                copy_id = bond_info_dict.get(
                    f"{side}_copy", bond_info_dict.get(f"copy{idx + 1}")
                )
                if copy_id is not None:
                    copy_id = self._strict_int(
                        copy_id, f"constraint.contact[{contact_idx}].copy{idx + 1}"
                    )
                position_raw = bond_info_dict.get(
                    f"{side}_position", bond_info_dict.get(f"position{idx + 1}")
                )
                if position_raw is None:
                    raise ValueError(
                        f"constraint.contact[{contact_idx}] missing "
                        f"position{idx + 1}/{side}_position."
                    )
                position = self._strict_int(
                    position_raw, f"constraint.contact[{contact_idx}].position{idx + 1}"
                )
                atom_name = bond_info_dict.get(
                    f"{side}_atom", bond_info_dict.get(f"atom{idx + 1}")
                )
                sides.append(
                    {
                        "entity_id": entity_id,
                        "copy_id": copy_id,
                        "position": position,
                        "atom_name": atom_name,
                    }
                )
            min_distance = self._strict_number(
                bond_info_dict.get("min_distance", 3.5),
                f"constraint.contact[{contact_idx}].min_distance",
            )
            max_distance = self._strict_number(
                bond_info_dict.get("max_distance", 8.0),
                f"constraint.contact[{contact_idx}].max_distance",
            )
            if min_distance < 0.0:
                raise ValueError(
                    f"constraint.contact[{contact_idx}]: min_distance "
                    f"({min_distance}) must be >= 0."
                )
            if max_distance < min_distance:
                raise ValueError(
                    f"constraint.contact[{contact_idx}]: max_distance "
                    f"({max_distance}) < min_distance ({min_distance})."
                )
            parsed.append(
                {
                    "left": sides[0],
                    "right": sides[1],
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                }
            )
        return parsed

    def _resolve_contact_atom_name(
        self, entity_id: int, atom_name: Any, contact_idx: int, side: str
    ) -> str:
        """Default omitted protein atom names to CA; otherwise require a name."""
        if atom_name is not None and str(atom_name).strip() != "":
            return str(atom_name)
        poly_type = self.entity_poly_type.get(str(entity_id), "")
        if poly_type.startswith("polypeptide"):
            return "CA"
        raise ValueError(
            f"constraint.contact[{contact_idx}] {side}_atom/atom is required "
            f"for non-protein entity {entity_id} (type={poly_type!r})."
        )

    def build_rigid_movable_features(
        self, atom_array: AtomArray
    ) -> dict[str, torch.Tensor]:
        """Resolve constraint.movable_chains into a per-atom mask.

        The mask names the chains that rigid contact guidance may move together,
        so that a heavy and light chain travel as one body and every antigen
        chain, ligand, glycan and extra copy stays where the sampler put it.
        An absent field yields an empty tensor; rigid contact guidance then
        supports exactly two chains.
        """
        constraint = self.single_sample_dict.get("constraint")
        chains = (
            constraint.get("movable_chains") if isinstance(constraint, dict) else None
        )
        if chains is None:
            return {"user_rigid_movable_atom": torch.empty((0,), dtype=torch.bool)}
        if not isinstance(chains, list) or not chains:
            raise ValueError(
                "constraint.movable_chains must be a non-empty list of chain ids."
            )
        wanted = [str(c) for c in chains]
        if len(set(wanted)) != len(wanted):
            raise ValueError("constraint.movable_chains contains a repeated chain id.")
        present = set(np.unique(atom_array.chain_id).tolist())
        missing = [c for c in wanted if c not in present]
        if missing:
            raise ValueError(
                f"constraint.movable_chains names chains that are absent: {missing}. "
                f"Chains present: {sorted(present)}."
            )
        mask = np.isin(atom_array.chain_id, wanted)
        if mask.all():
            raise ValueError(
                "constraint.movable_chains names every chain; nothing would stay fixed."
            )
        return {"user_rigid_movable_atom": torch.as_tensor(mask, dtype=torch.bool)}

    def build_user_distance_restraint_features(
        self, atom_array: AtomArray
    ) -> dict[str, torch.Tensor]:
        """Resolve contact restraints to atom-index distance features for TFG.

        Returns tensors shaped for UserDistanceRestraintPotential:
            user_distance_restraint_index: [2, M]
            user_distance_restraint_lower_bound: [M]
            user_distance_restraint_upper_bound: [M]
        """
        pairs: list[list[int]] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for contact_idx, restraint in enumerate(self.contact_restraints):
            side_indices: list[np.ndarray] = []
            for side_name in ("left", "right"):
                side = restraint[side_name]
                atom_name = self._resolve_contact_atom_name(
                    side["entity_id"], side["atom_name"], contact_idx, side_name
                )
                atom_indices = self.get_a_bond_atom(
                    atom_array,
                    side["entity_id"],
                    side["position"],
                    atom_name,
                    side["copy_id"],
                )
                if atom_indices.size == 0:
                    raise ValueError(
                        f"No atom found for constraint.contact[{contact_idx}] "
                        f"{side_name}: entity={side['entity_id']} "
                        f"position={side['position']} atom={atom_name!r} "
                        f"copy={side['copy_id']}."
                    )
                side_indices.append(atom_indices)

            if len(side_indices[0]) != len(side_indices[1]):
                raise ValueError(
                    f"constraint.contact[{contact_idx}]: asymmetric copy counts "
                    f"({len(side_indices[0])} vs {len(side_indices[1])}); "
                    "specify copy1/copy2 explicitly or keep entity counts equal."
                )
            for atom_idx1, atom_idx2 in zip(side_indices[0], side_indices[1]):
                pairs.append([int(atom_idx1), int(atom_idx2)])
                lowers.append(float(restraint["min_distance"]))
                uppers.append(float(restraint["max_distance"]))

        if pairs:
            index = torch.as_tensor(pairs, dtype=torch.int64).T  # [2, M]
        else:
            index = torch.empty((2, 0), dtype=torch.int64)
        lower = torch.as_tensor(lowers, dtype=torch.float32)
        upper = torch.as_tensor(uppers, dtype=torch.float32)
        return {
            "user_distance_restraint_index": index,
            "user_distance_restraint_lower_bound": lower,
            "user_distance_restraint_upper_bound": upper,
        }

    @staticmethod
    def _strict_int(value: Any, label: str) -> int:
        """An integer, or a string of digits. Booleans, fractional numbers and anything else are refused (int() would truncate 9.9 to 9)."""
        if isinstance(value, bool):
            raise ValueError(f"{label} must be an integer, got the boolean {value!r}.")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
            return int(value.strip())
        raise ValueError(
            f"{label} must be an integer (or a string of digits), got {value!r}."
        )

    @staticmethod
    def _strict_number(value: Any, label: str) -> float:
        """A finite number. Booleans, strings, null, NaN and infinities are refused."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must be a finite number, got {value!r}.")
        return float(value)

    _EPITOPE_KEYS = {"residues", "paratope", "min_fraction"}
    _CDR_WINDOWS = [[24, 40], [50, 66], [88, 115]]
    _AA3TO1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    def build_epitope_features(
        self, atom_array: AtomArray, movable_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Resolve constraint.epitope (antigen residues only) to atom-index features.

        The request is "at least K of the n distinct resolved epitope residues have a heavy
        atom within the design gate of a paratope heavy atom", with K = max(1, ceil(min_fraction * n)).
        Any malformed field is an error: this path never falls back to an unguided run.
        Returns an empty dict when the constraint carries no epitope.
        """
        constraint = self.single_sample_dict.get("constraint")
        epitope = constraint.get("epitope") if isinstance(constraint, dict) else None
        if epitope is None:
            return {}
        if not isinstance(epitope, dict):
            raise ValueError("constraint.epitope must be an object.")
        unknown = sorted(set(epitope.keys()) - self._EPITOPE_KEYS)
        if unknown:
            raise ValueError(f"Unknown constraint.epitope field(s): {unknown}.")
        if movable_mask.numel() == 0:
            raise ValueError(
                "constraint.epitope requires constraint.movable_chains (the antibody chains that move)."
            )
        residues = epitope.get("residues")
        if not isinstance(residues, list) or not residues:
            raise ValueError("constraint.epitope.residues must be a non-empty list.")
        min_fraction = self._strict_number(
            epitope.get("min_fraction", 0.5), "constraint.epitope.min_fraction"
        )
        if not 0.0 < min_fraction <= 1.0:
            raise ValueError("constraint.epitope.min_fraction must be in (0, 1].")
        movable = movable_mask.numpy().astype(bool)
        seen: dict[tuple[str, int, int], list[int]] = {}
        for idx, item in enumerate(residues):
            if not isinstance(item, dict):
                raise ValueError(
                    f"constraint.epitope.residues[{idx}] must be an object."
                )
            extra = sorted(set(item.keys()) - {"entity", "copy", "position", "residue"})
            if extra:
                raise ValueError(
                    f"constraint.epitope.residues[{idx}]: unknown field(s) {extra}."
                )
            for key in ("entity", "copy", "position"):
                if item.get(key) is None:
                    raise ValueError(
                        f"constraint.epitope.residues[{idx}] missing {key}."
                    )
            entity = str(
                self._strict_int(
                    item["entity"], f"constraint.epitope.residues[{idx}].entity"
                )
            )
            copy_id = self._strict_int(
                item["copy"], f"constraint.epitope.residues[{idx}].copy"
            )
            position = self._strict_int(
                item["position"], f"constraint.epitope.residues[{idx}].position"
            )
            key = (entity, copy_id, position)
            mask = (
                (atom_array.label_entity_id == entity)
                & (atom_array.copy_id == copy_id)
                & (atom_array.res_id == position)
            )
            atoms = np.where(mask)[0]
            if atoms.size == 0:
                raise ValueError(
                    f"constraint.epitope.residues[{idx}]: no atom found for entity={entity} "
                    f"copy={copy_id} position={position}."
                )
            names = set(atom_array.res_name[atoms].tolist())
            if len(names) != 1:
                raise ValueError(
                    f"constraint.epitope.residues[{idx}] resolves to several residue types {sorted(names)}."
                )
            name = names.pop()
            if name not in self._AA3TO1:
                raise ValueError(
                    f"constraint.epitope.residues[{idx}] is {name!r}, not a standard amino acid."
                )
            expected = item.get("residue")
            if expected is not None and self._AA3TO1[name] != str(expected).upper():
                raise ValueError(
                    f"constraint.epitope.residues[{idx}]: residue {expected!r} given, "
                    f"the sequence has {self._AA3TO1[name]!r} at entity={entity} copy={copy_id} position={position}."
                )
            if movable[atoms].any():
                raise ValueError(
                    f"constraint.epitope.residues[{idx}] lies in a movable chain; the epitope must stay fixed."
                )
            # every record is validated before duplicates are dropped; a repeated residue counts once, the counts are logged below
            seen[key] = atoms.tolist()
        seen = dict(
            sorted(seen.items(), key=lambda kv: (int(kv[0][0]), kv[0][1], kv[0][2]))
        )  # canonical order: the list order never matters
        n = len(seen)
        from fractions import Fraction

        k = max(
            1, math.ceil(Fraction(repr(min_fraction)) * n)
        )  # exact rational ceiling
        # paratope: the heavy atoms of the moving group that count as the binding face
        spec = epitope.get("paratope", "all")
        chain_ids = atom_array.chain_id
        moving_chains = sorted(set(chain_ids[movable].tolist()))
        if spec == "all":
            windows = None
        elif spec == "cdr":
            windows = {c: self._CDR_WINDOWS for c in moving_chains}
        elif (
            isinstance(spec, dict)
            and set(spec.keys()) == {"windows"}
            and isinstance(spec["windows"], dict)
        ):
            windows = {}
            for chain, spans in spec["windows"].items():
                if str(chain) not in moving_chains:
                    raise ValueError(
                        f"constraint.epitope.paratope.windows names {chain!r}, which is not a movable chain."
                    )
                if (
                    not isinstance(spans, list)
                    or not spans
                    or any(not isinstance(w, list) or len(w) != 2 for w in spans)
                ):
                    raise ValueError(
                        f"constraint.epitope.paratope.windows[{chain!r}] must be a list of [low, high] pairs."
                    )
                pairs = [
                    [
                        self._strict_int(w[0], f"windows[{chain!r}] low"),
                        self._strict_int(w[1], f"windows[{chain!r}] high"),
                    ]
                    for w in spans
                ]
                if any(lo > hi for lo, hi in pairs):
                    raise ValueError(
                        f"constraint.epitope.paratope.windows[{chain!r}] has a window with low > high."
                    )
                windows[str(chain)] = pairs
        else:
            raise ValueError(
                'constraint.epitope.paratope must be "cdr", "all" or {"windows": {chain: [[low, high], ...]}}.'
            )
        para = np.zeros(atom_array.shape[0], dtype=bool)
        if windows is None:
            para = movable.copy()
        else:
            for chain, spans in windows.items():
                for lo, hi in spans:
                    para |= (
                        movable
                        & (chain_ids == chain)
                        & (atom_array.res_id >= lo)
                        & (atom_array.res_id <= hi)
                    )
        if not para.any():
            raise ValueError(
                "constraint.epitope: the paratope selects no atom of the movable chains."
            )
        width = max(len(v) for v in seen.values())
        index = np.full((n, width), -1, dtype=np.int64)
        for row, atoms in enumerate(seen.values()):
            index[row, : len(atoms)] = atoms
        para_res = {
            c: len({int(r) for r in atom_array.res_id[para & (chain_ids == c)]})
            for c in moving_chains
        }
        logger.info(
            "EPITOPE_INPUT declared=%d unique_resolved=%d K=%d min_fraction=%s paratope_atoms=%d paratope_residues_per_chain=%s movable_chains=%s",
            len(residues),
            n,
            k,
            min_fraction,
            int(para.sum()),
            para_res,
            moving_chains,
        )
        return {
            "user_epitope_atom_index": torch.as_tensor(index, dtype=torch.int64),
            "user_epitope_paratope_atom": torch.as_tensor(para, dtype=torch.bool),
            "user_epitope_k": torch.as_tensor([k], dtype=torch.int64),
        }

    @staticmethod
    def get_a_bond_atom(
        atom_array: AtomArray,
        entity_id: int,
        position: int,
        atom_name: str,
        copy_id: int | None = None,
    ) -> np.ndarray:
        """
        Get the atom index of a bond atom.

        Args:
            atom_array (AtomArray): Biotite Atom array.
            entity_id (int): Entity id.
            position (int): Residue index of the atom.
            atom_name (str): Atom name.
            copy_id (copy_id): A asym chain id in N copies of an entity.

        Returns:
            np.ndarray: Array of indices for specified atoms on each asym chain.
        """
        entity_mask = atom_array.label_entity_id == str(entity_id)
        position_mask = atom_array.res_id == int(position)
        atom_name_mask = atom_array.atom_name == str(atom_name)

        if copy_id is not None:
            copy_mask = atom_array.copy_id == int(copy_id)
            mask = entity_mask & position_mask & atom_name_mask & copy_mask
        else:
            mask = entity_mask & position_mask & atom_name_mask
        atom_indices = np.where(mask)[0]
        return atom_indices

    def add_bonds_between_entities(self, atom_array: AtomArray) -> AtomArray:
        """
        Based on the information in the "covalent_bonds",
        add a bond between specified atoms on each pair of asymmetric chains of the two entities.
        Note that this requires the number of asymmetric chains in both entities to be equal.

        Args:
            atom_array (AtomArray): Biotite Atom array.

        Returns:
            AtomArray: Biotite Atom array with bonds added.
        """
        if "covalent_bonds" not in self.input_dict:
            return atom_array

        bond_count = {}
        for bond_info_dict in self.input_dict["covalent_bonds"]:
            bond_atoms = []
            for idx, i in enumerate(["left", "right"]):
                entity_id = int(
                    bond_info_dict.get(
                        f"{i}_entity", bond_info_dict.get(f"entity{idx + 1}")
                    )
                )
                copy_id = bond_info_dict.get(
                    f"{i}_copy", bond_info_dict.get(f"copy{idx + 1}")
                )
                position = int(
                    bond_info_dict.get(
                        f"{i}_position", bond_info_dict.get(f"position{idx + 1}")
                    )
                )
                atom_name = bond_info_dict.get(
                    f"{i}_atom", bond_info_dict.get(f"atom{idx + 1}")
                )

                if copy_id is not None:
                    copy_id = int(copy_id)

                if isinstance(atom_name, str):
                    if atom_name.isdigit():
                        # Convert SMILES atom index to int
                        atom_name = int(atom_name)

                if isinstance(atom_name, int):
                    # Convert AtomMap in SMILES to atom name in AtomArray
                    entity_dict = list(
                        self.input_dict["sequences"][int(entity_id - 1)].values()
                    )[0]
                    assert "atom_map_to_atom_name" in entity_dict
                    atom_name = entity_dict["atom_map_to_atom_name"][atom_name]

                # Get bond atoms by entity_id, position, atom_name
                atom_indices = self.get_a_bond_atom(
                    atom_array, entity_id, position, atom_name, copy_id
                )
                assert atom_indices.size > 0, (
                    f"No atom found for {atom_name} in entity {entity_id} at position {position}."
                )
                bond_atoms.append(atom_indices)
            assert len(bond_atoms[0]) == len(bond_atoms[1]), (
                'Can not create bonds because the "count" of entity1 '
                f"({bond_info_dict.get('left_entity', bond_info_dict.get('entity1'))}) "
                f"and entity2 ({bond_info_dict.get('right_entity', bond_info_dict.get('entity2'))}) are not equal. "
            )

            # Create bond between each asym chain pair
            for atom_idx1, atom_idx2 in zip(bond_atoms[0], bond_atoms[1]):
                atom_array.bonds.add_bond(atom_idx1, atom_idx2, 1)
                bond_count[atom_idx1] = bond_count.get(atom_idx1, 0) + 1
                bond_count[atom_idx2] = bond_count.get(atom_idx2, 0) + 1

        atom_array = remove_leaving_atoms(atom_array, bond_count)

        return atom_array

    @staticmethod
    def add_atom_array_attributes(
        atom_array: AtomArray, entity_poly_type: dict[str, str]
    ) -> AtomArray:
        """
        Add attributes to the Biotite AtomArray.

        Args:
            atom_array (AtomArray): Biotite Atom array.
            entity_poly_type (dict[str, str]): a dict of polymer entity id to entity type.

        Returns:
            AtomArray: Biotite Atom array with attributes added.
        """
        atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        atom_array.set_annotation("is_resolved", np.ones(len(atom_array), dtype=bool))
        atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
        atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
            atom_array, entity_poly_type=entity_poly_type
        )
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)
        return atom_array

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

    def get_atom_array(self) -> AtomArray:
        """
        Create a Biotite AtomArray and add attributes from the input dict.

        Returns:
            AtomArray: Biotite Atom array.
        """
        atom_array = self.build_full_atom_array()
        atom_array = self.add_bonds_between_entities(atom_array)
        atom_array = self.mse_to_met(atom_array)
        atom_array = self.add_atom_array_attributes(atom_array, self.entity_poly_type)
        return atom_array

    def get_feature_dict(self) -> tuple[dict[str, torch.Tensor], AtomArray, TokenArray]:
        """
        Generates a feature dictionary from the input sample dictionary.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A TokenArray object.
        """
        atom_array = self.get_atom_array()

        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        featurizer = Featurizer(
            token_array, atom_array, include_discont_poly_poly_bonds=True
        )
        feature_dict = featurizer.get_all_input_features()

        token_array_with_frame = featurizer.get_token_frame(
            token_array=token_array,
            atom_array=atom_array,
            ref_pos=feature_dict["ref_pos"],
            ref_mask=feature_dict["ref_mask"],
        )

        # [N_token]
        feature_dict["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()

        # [N_token, 3]
        feature_dict["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()
        if self.extract_features_for_tfg:
            geometry_featurizer = GeometryFeaturizer(
                atom_array,
                ccd_mols=self.input_dict.get("ccd_mols"),
                exclude_std_residue=True,
            )
            feature_dict.update(geometry_featurizer.get_features())
            # User epitope / contact distance restraints (do not touch bond graph).
            feature_dict.update(self.build_user_distance_restraint_features(atom_array))
            feature_dict.update(self.build_rigid_movable_features(atom_array))
            feature_dict.update(
                self.build_epitope_features(
                    atom_array, feature_dict["user_rigid_movable_atom"]
                )
            )
        return feature_dict, atom_array, token_array
