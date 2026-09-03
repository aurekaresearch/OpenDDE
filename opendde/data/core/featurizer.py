# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from typing import Optional, Union

import numpy as np
import torch
from biotite.structure import Atom, AtomArray
from sklearn.neighbors import KDTree

from opendde.data.constants import STD_RESIDUES, STD_RESIDUES_WITH_GAP, get_all_elems
from opendde.data.tokenizer import AtomArrayTokenizer, Token, TokenArray
from opendde.data.utils import get_atom_level_token_mask, get_ligand_polymer_bond_mask
from opendde.utils.geometry import angle_3p, random_transform


class Featurizer(object):
    """
    Args:
        cropped_token_array (TokenArray): TokenArray object after cropping
        cropped_atom_array (AtomArray): AtomArray object after cropping
        ref_pos_augment (bool): Boolean indicating whether apply random rotation and translation on ref_pos
        include_discont_poly_poly_bonds (bool): Boolean indicating whether
                                        include discontinuous polymer-polymer bonds
    """

    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        ref_pos_augment: bool = True,
        include_discont_poly_poly_bonds: bool = False,
    ) -> None:
        self.cropped_token_array = cropped_token_array

        self.cropped_atom_array = cropped_atom_array
        self.ref_pos_augment = ref_pos_augment
        self.include_discont_poly_poly_bonds = include_discont_poly_poly_bonds

    @staticmethod
    def encoder(
        encode_def_dict_or_list: Optional[Union[dict, list[str]]], input_list: list[str]
    ) -> torch.Tensor:
        """
        Encode a list of input values into a binary format using a specified encoding definition list.

        Args:
            encode_def_dict_or_list (list or dict): A list or dict of encoding definitions.
            input_list (list): A list of input values to be encoded.

        Returns:
            torch.Tensor: A tensor representing the binary encoding of the input values.
        """
        if isinstance(encode_def_dict_or_list, dict):
            num_keys = len(encode_def_dict_or_list)
            assert num_keys == max(encode_def_dict_or_list.values()) + 1, (
                "Do not use discontinuous number, which might causing potential bugs in the code"
            )
            idx_map = encode_def_dict_or_list
        elif isinstance(encode_def_dict_or_list, list):
            num_keys = len(encode_def_dict_or_list)
            idx_map = {key: idx for idx, key in enumerate(encode_def_dict_or_list)}
        else:
            raise TypeError(
                "encode_def_dict_or_list must be a list or dict, "
                f"but got {type(encode_def_dict_or_list)}"
            )
        indices = torch.tensor([idx_map[item] for item in input_list], dtype=torch.long)
        return torch.nn.functional.one_hot(indices, num_classes=num_keys).float()

    @staticmethod
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "restype"
        One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
        4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap.
        Ligands represented as “unknown amino acid”.

        Args:
            restype_list (List[str]): A list of residue types.
                                      The residue type of ligand should be "UNK" in the input list.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded residue types
        """

        return Featurizer.encoder(STD_RESIDUES_WITH_GAP, restype_list)

    @staticmethod
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_element"
        One-hot encoding of the element atomic number for each atom
        in the reference conformer, up to atomic number 128.

        Args:
            elem_list (List[str]): A list of element symbols.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded elements
        """
        return Featurizer.encoder(get_all_elems(), elem_list)

    @staticmethod
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) − 32, and names are padded to length 4.

        Args:
            atom_name_list (List[str]): A list of atom names.

        Returns:
            torch.Tensor:  A Tensor of character encoded atom names
        """
        n = len(atom_names)
        padded = "".join(name.ljust(4)[:4] for name in atom_names)
        char_codes = np.frombuffer(padded.encode("ascii"), dtype=np.uint8)
        raw_indices = char_codes.astype(np.int64) - 32
        if not ((raw_indices >= 0) & (raw_indices < 64)).all():
            import logging

            logging.getLogger(__name__).warning(
                "ref_atom_name_chars_encoded: character codes outside [32, 95] "
                "detected and will be clipped to [0, 63]."
            )
        indices = raw_indices.clip(0, 63)
        indices_tensor = torch.from_numpy(indices).reshape(n, 4)
        return torch.nn.functional.one_hot(indices_tensor, num_classes=64).float()

    @staticmethod
    def get_prot_nuc_frame(token: Token, centre_atom: Atom) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For proteins/DNA/RNA, we use the three atoms [N, CA, C] / [C1', C3', C4']

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.

        Returns:
            has_frame (int): 1 if the token has frame, 0 otherwise.
            frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        if centre_atom.mol_type == "protein":
            # For protein
            abc_atom_name = ["N", "CA", "C"]
        else:
            # For DNA and RNA
            abc_atom_name = [r"C1'", r"C3'", r"C4'"]

        idx_in_atom_indices = []
        for i in abc_atom_name:
            if centre_atom.mol_type == "protein" and "N" not in token.atom_names:
                return 0, [-1, -1, -1]
            elif centre_atom.mol_type != "protein" and "C1'" not in token.atom_names:
                return 0, [-1, -1, -1]
            idx_in_atom_indices.append(token.atom_names.index(i))
        # Protein/DNA/RNA always has frame
        has_frame = 1
        frame_atom_index = [token.atom_indices[i] for i in idx_in_atom_indices]
        return has_frame, frame_atom_index

    @staticmethod
    def get_lig_frame(
        token: Token,
        centre_atom: Atom,
        lig_res_ref_conf_kdtree: dict[str, tuple[Optional[KDTree], np.ndarray]],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For ligands, we use the reference conformer of the ligand to construct the frame.

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.
            lig_res_ref_conf_kdtree (Dict[str, Tuple[KDTree, List[int]]]): A dictionary of KDTree objects and atom indices.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            tuple[int, List[int]]:
                has_frame (int): 1 if the token has frame, 0 otherwise.
                frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        kdtree, atom_ids = lig_res_ref_conf_kdtree[centre_atom.ref_space_uid]
        b_ref_pos = ref_pos[token.centre_atom_index]
        b_idx = token.centre_atom_index
        if kdtree is None:
            # Atom num < 3
            frame_atom_index = [-1, b_idx, -1]
            has_frame = 0
        else:
            _dist, ind = kdtree.query([b_ref_pos], k=3)
            a_idx, c_idx = atom_ids[ind[0][1]], atom_ids[ind[0][2]]
            frame_atom_index = [a_idx, b_idx, c_idx]

            # Check if reference confomrer vaild
            has_frame = all([ref_mask[idx] for idx in frame_atom_index])

            # Colinear check
            if has_frame:
                vec1 = ref_pos[frame_atom_index[1]] - ref_pos[frame_atom_index[0]]
                vec2 = ref_pos[frame_atom_index[2]] - ref_pos[frame_atom_index[1]]
                # ref_pos can be all zeros, in which case has_frame=0
                is_zero_norm = np.isclose(
                    np.linalg.norm(vec1, axis=-1), 0
                ) or np.isclose(np.linalg.norm(vec2, axis=-1), 0)
                if is_zero_norm:
                    has_frame = 0
                else:
                    theta_degrees = angle_3p(
                        *[ref_pos[idx] for idx in frame_atom_index]
                    )
                    is_colinear = theta_degrees <= 25 or theta_degrees >= 155
                    if is_colinear:
                        has_frame = 0
        return int(has_frame), frame_atom_index

    @staticmethod
    def get_token_frame(
        token_array: TokenArray,
        atom_array: AtomArray,
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> TokenArray:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        The atoms (a_i, b_i, c_i) used to construct token i’s frame depend on the chain type of i:
        Protein tokens use their residue’s backbone (N, Cα, C),
        while DNA and RNA tokens use (C1′, C3′, C4′) atoms of their residue.
        All other tokens (small molecules, glycans, ions) contain only one atom per token.
        The token atom is assigned to b_i, the closest atom to the token atom is a_i,
        and the second closest atom to the token atom is c_i.
        If this set of three atoms is close to colinear (less than 25 degree deviation),
        or if three atoms do not exist in the chain (e.g. a sodium ion),
        then the frame is marked as invalid.

        Note: frames constucted from reference conformer

        Args:
            token_array (TokenArray): A list of tokens.
            atom_array (AtomArray): An atom array.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            TokenArray: A TokenArray with updated frame annotations.
                        - has_frame: 1 if the token has frame, 0 otherwise.
                        - frame_atom_index: The index of the atoms used to construct the frame.
        """
        token_array_w_frame = token_array
        atom_level_token_mask = get_atom_level_token_mask(token_array, atom_array)

        # Construct a KDTree for queries to avoid redundant distance calculations
        lig_res_ref_conf_kdtree = {}
        # Ligand and non-standard residues need to use ref to identify frames
        lig_atom_array = atom_array[
            (atom_array.mol_type == "ligand")
            | (~np.isin(atom_array.res_name, list(STD_RESIDUES.keys())))
            | atom_level_token_mask
        ]
        for ref_space_uid in np.unique(lig_atom_array.ref_space_uid):
            # The ref_space_uid is the unique identifier ID for each residue.
            atom_ids = np.where(atom_array.ref_space_uid == ref_space_uid)[0]
            if len(atom_ids) >= 3:
                kdtree = KDTree(ref_pos[atom_ids], metric="euclidean")
            else:
                # Invalid frame
                kdtree = None
            lig_res_ref_conf_kdtree[ref_space_uid] = (kdtree, atom_ids)

        has_frame = []
        for token in token_array_w_frame:
            centre_atom = atom_array[token.centre_atom_index]
            if (
                centre_atom.mol_type != "ligand"
                and centre_atom.res_name in STD_RESIDUES
                and len(token.atom_indices) > 1
            ):
                has_frame, frame_atom_index = Featurizer.get_prot_nuc_frame(
                    token, centre_atom
                )

            else:
                has_frame, frame_atom_index = Featurizer.get_lig_frame(
                    token, centre_atom, lig_res_ref_conf_kdtree, ref_pos, ref_mask
                )

            token.has_frame = has_frame
            token.frame_atom_index = frame_atom_index
        return token_array_w_frame

    def get_token_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token features.
        The size of these features is [N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of token features.
        """
        token_features = {}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        restype = centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)

        token_features["token_index"] = torch.arange(0, len(self.cropped_token_array))
        token_features["residue_index"] = torch.from_numpy(
            centre_atoms.res_id.astype(np.int64)
        )
        token_features["asym_id"] = torch.from_numpy(
            centre_atoms.asym_id_int.astype(np.int64)
        )
        token_features["entity_id"] = torch.from_numpy(
            centre_atoms.entity_id_int.astype(np.int64)
        )
        token_features["sym_id"] = torch.from_numpy(
            centre_atoms.sym_id_int.astype(np.int64)
        )
        token_features["restype"] = restype_onehot

        return token_features

    def get_chain_perm_features(self) -> dict[str, torch.Tensor]:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: A dict of chain permutation features.
        """

        chain_perm_features = {}
        chain_perm_features["mol_id"] = torch.from_numpy(
            self.cropped_atom_array.mol_id.astype(np.int64)
        )
        chain_perm_features["mol_atom_index"] = torch.from_numpy(
            self.cropped_atom_array.mol_atom_index.astype(np.int64)
        )
        chain_perm_features["entity_mol_id"] = torch.from_numpy(
            self.cropped_atom_array.entity_mol_id.astype(np.int64)
        )
        return chain_perm_features

    def get_reference_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get reference features.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of reference features.
        """
        ref_pos = np.empty_like(self.cropped_atom_array.ref_pos)
        for ref_space_uid in np.unique(self.cropped_atom_array.ref_space_uid):
            mask = self.cropped_atom_array.ref_space_uid == ref_space_uid
            ref_pos[mask] = random_transform(
                self.cropped_atom_array.ref_pos[mask],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )

        ref_features = {}
        ref_features["ref_pos"] = torch.Tensor(ref_pos)
        ref_features["ref_mask"] = torch.from_numpy(
            self.cropped_atom_array.ref_mask.astype(np.int64)
        )
        ref_features["ref_element"] = Featurizer.elem_onehot_encoded(
            self.cropped_atom_array.element
        ).long()
        ref_features["ref_charge"] = torch.from_numpy(
            self.cropped_atom_array.ref_charge.astype(np.int64)
        )

        atom_names = self.cropped_atom_array.atom_name

        ref_features["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
            atom_names
        ).long()
        ref_features["ref_space_uid"] = torch.from_numpy(
            self.cropped_atom_array.ref_space_uid.astype(np.int64)
        )

        token_array_with_frame = self.get_token_frame(
            token_array=self.cropped_token_array,
            atom_array=self.cropped_atom_array,
            ref_pos=ref_features["ref_pos"],
            ref_mask=ref_features["ref_mask"],
        )
        ref_features["has_frame"] = torch.from_numpy(
            np.array(token_array_with_frame.get_annotation("has_frame")).astype(
                np.int64
            )
        )  # [N_token]
        ref_features["frame_atom_index"] = torch.from_numpy(
            np.array(token_array_with_frame.get_annotation("frame_atom_index")).astype(
                np.int64
            )
        )  # [N_token, 3]
        return ref_features

    def _get_atom_to_token_idx(self) -> np.ndarray:
        """Build atom-to-token index mapping. Cached after first call.

        Returns:
            np.ndarray: array of shape [N_atom] mapping each atom to its token index.
        """
        if hasattr(self, "_atom_to_token_idx_cache"):
            return self._atom_to_token_idx_cache
        n_atoms = len(self.cropped_atom_array)
        atom_idx_to_token_idx = np.full(n_atoms, -1, dtype=int)
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_idx_to_token_idx[atom_idx] = idx
        self._atom_to_token_idx_cache = atom_idx_to_token_idx
        return atom_idx_to_token_idx

    def get_bond_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to polymer-ligand and ligand-ligand bonds shorter than 2.4 Å.
        The size of bond feature is [N_token, N_token].
        Returns:
            Dict[str, torch.Tensor]: A dict of bond features.
        """
        bond_array = self.cropped_atom_array.bonds.as_array()
        bond_atom_i = bond_array[:, 0]
        bond_atom_j = bond_array[:, 1]
        ref_space_uid = self.cropped_atom_array.ref_space_uid
        polymer_mask = np.isin(
            self.cropped_atom_array.mol_type, ["protein", "dna", "rna"]
        )
        std_res_mask = (
            np.isin(self.cropped_atom_array.res_name, list(STD_RESIDUES.keys()))
            & polymer_mask
        )
        unstd_res_mask = ~std_res_mask & polymer_mask
        # the polymer-polymer (std-std, std-unstd, and inter-unstd) bond will not be included in token_bonds.
        std_std_bond_mask = std_res_mask[bond_atom_i] & std_res_mask[bond_atom_j]
        std_unstd_bond_mask = (
            std_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) | (std_res_mask[bond_atom_j] & unstd_res_mask[bond_atom_i])
        inter_unstd_bond_mask = (
            unstd_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) & (ref_space_uid[bond_atom_i] != ref_space_uid[bond_atom_j])

        kept_mask = ~(std_std_bond_mask | std_unstd_bond_mask | inter_unstd_bond_mask)
        if self.include_discont_poly_poly_bonds:
            # include discontinuous polymer-polymer bonds
            res_id_i = self.cropped_atom_array.res_id[bond_atom_i]
            res_id_j = self.cropped_atom_array.res_id[bond_atom_j]
            chain_i = self.cropped_atom_array.chain_id[bond_atom_i]
            chain_j = self.cropped_atom_array.chain_id[bond_atom_j]
            is_discont = (np.abs(res_id_i - res_id_j) > 1) | (chain_i != chain_j)
            kept_mask |= is_discont

        kept_bonds = bond_array[kept_mask]

        # -1 means the atom is not in any token
        atom_idx_to_token_idx = self._get_atom_to_token_idx()
        assert np.all(atom_idx_to_token_idx >= 0), "Some atoms are not in any token"
        num_tokens = len(self.cropped_token_array)
        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        bond_token_i = atom_idx_to_token_idx[kept_bonds[:, 0]]
        bond_token_j = atom_idx_to_token_idx[kept_bonds[:, 1]]
        token_adj_matrix[bond_token_i, bond_token_j] = 1
        token_adj_matrix[bond_token_j, bond_token_i] = 1
        bond_features = {"token_bonds": torch.Tensor(token_adj_matrix)}
        return bond_features

    def get_extra_features(self) -> dict[str, torch.Tensor]:
        """
        Get other features not listed in AlphaFold3 SI Chapter 2.8 Table 5.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of extra features.
        """
        atom_to_token_idx = self._get_atom_to_token_idx()

        extra_features = {}
        extra_features["atom_to_token_idx"] = torch.from_numpy(
            atom_to_token_idx.astype(np.int64)
        )
        extra_features["atom_to_tokatom_idx"] = torch.from_numpy(
            self.cropped_atom_array.tokatom_idx.astype(np.int64)
        )

        extra_features["is_protein"] = torch.from_numpy(
            self.cropped_atom_array.is_protein.astype(np.int64)
        )
        extra_features["is_ligand"] = torch.from_numpy(
            self.cropped_atom_array.is_ligand.astype(np.int64)
        )
        extra_features["is_dna"] = torch.from_numpy(
            self.cropped_atom_array.is_dna.astype(np.int64)
        )
        extra_features["is_rna"] = torch.from_numpy(
            self.cropped_atom_array.is_rna.astype(np.int64)
        )
        if "resolution" in self.cropped_atom_array._annot:
            extra_features["resolution"] = torch.Tensor(
                [self.cropped_atom_array.resolution[0]]
            )
        else:
            extra_features["resolution"] = torch.Tensor([-1])
        return extra_features

    def get_structural_token_features(
        self, ref_pos: torch.Tensor, ref_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Get the residue-to-structural-token mapping for the late structure branch.

        The current residue token space is kept unchanged for MSA/template/trunk
        features. These tensors describe the optional expanded structural token
        space so downstream modules can switch atom/token mappings after trunk.
        """
        structural_token_array = AtomArrayTokenizer(
            self.cropped_atom_array
        ).get_structural_token_array(self.cropped_token_array)
        structural_token_array = self.get_token_frame(
            token_array=structural_token_array,
            atom_array=self.cropped_atom_array,
            ref_pos=ref_pos,
            ref_mask=ref_mask,
        )
        structural_token_array = self.inherit_parent_backbone_frames(
            structural_token_array
        )

        atom_to_structural_token_idx = np.full(
            len(self.cropped_atom_array), -1, dtype=np.int64
        )
        atom_to_structural_tokatom_idx = np.full(
            len(self.cropped_atom_array), -1, dtype=np.int64
        )
        structural_rep_atom_mask = np.zeros(
            len(self.cropped_atom_array), dtype=np.int64
        )
        prev_parent_residue_idx, next_parent_residue_idx = (
            self.get_polymer_residue_graph()
        )
        structural_prev_parent_residue_idx = []
        structural_next_parent_residue_idx = []
        structural_is_polymer = []
        structural_polymer_type = []
        structural_seq_pos = []
        for token_idx, token in enumerate(structural_token_array):
            for tokatom_idx, atom_idx in enumerate(token.atom_indices):
                atom_to_structural_token_idx[atom_idx] = token_idx
                atom_to_structural_tokatom_idx[atom_idx] = tokatom_idx
            structural_rep_atom_mask[token.centre_atom_index] = 1
            parent_idx = token.parent_residue_idx
            centre_atom = self.cropped_atom_array[token.centre_atom_index]
            structural_prev_parent_residue_idx.append(
                prev_parent_residue_idx[parent_idx]
            )
            structural_next_parent_residue_idx.append(
                next_parent_residue_idx[parent_idx]
            )
            is_polymer = centre_atom.mol_type in ["protein", "dna", "rna"]
            structural_is_polymer.append(int(is_polymer))
            structural_polymer_type.append(
                {"protein": 1, "dna": 2, "rna": 3}.get(centre_atom.mol_type, 0)
            )
            structural_seq_pos.append(centre_atom.res_id)

        assert np.all(atom_to_structural_token_idx >= 0), (
            "Some atoms are not in any structural token"
        )
        assert np.all(atom_to_structural_tokatom_idx >= 0), (
            "Some atoms are not assigned a structural token atom slot"
        )

        subtoken_role_id = np.array(
            structural_token_array.get_annotation("subtoken_role_id"), dtype=np.int64
        )
        return {
            "structural_token_index": torch.arange(0, len(structural_token_array)),
            "residue_token_group_id": torch.from_numpy(
                np.array(
                    structural_token_array.get_annotation("residue_token_group_id"),
                    dtype=np.int64,
                )
            ),
            "subtoken_role": torch.from_numpy(subtoken_role_id),
            "subtoken_role_id": torch.from_numpy(subtoken_role_id.copy()),
            "twin_token_idx": torch.from_numpy(
                np.array(
                    structural_token_array.get_annotation("twin_token_idx"),
                    dtype=np.int64,
                )
            ),
            "parent_residue_idx": torch.from_numpy(
                np.array(
                    structural_token_array.get_annotation("parent_residue_idx"),
                    dtype=np.int64,
                )
            ),
            "atom_to_structural_token_idx": torch.from_numpy(
                atom_to_structural_token_idx
            ),
            "atom_to_structural_tokatom_idx": torch.from_numpy(
                atom_to_structural_tokatom_idx
            ),
            "structural_distogram_rep_atom_mask": torch.from_numpy(
                structural_rep_atom_mask
            ),
            "structural_pae_rep_atom_mask": torch.from_numpy(
                structural_rep_atom_mask.copy()
            ),
            "structural_has_frame": torch.from_numpy(
                np.array(
                    structural_token_array.get_annotation("has_frame"),
                    dtype=np.int64,
                )
            ),
            "structural_frame_atom_index": torch.from_numpy(
                np.array(
                    structural_token_array.get_annotation("frame_atom_index"),
                    dtype=np.int64,
                )
            ),
            "prev_parent_residue_idx": torch.from_numpy(
                np.array(structural_prev_parent_residue_idx, dtype=np.int64)
            ),
            "next_parent_residue_idx": torch.from_numpy(
                np.array(structural_next_parent_residue_idx, dtype=np.int64)
            ),
            "structural_is_polymer": torch.from_numpy(
                np.array(structural_is_polymer, dtype=np.int64)
            ),
            "structural_polymer_type": torch.from_numpy(
                np.array(structural_polymer_type, dtype=np.int64)
            ),
            "structural_seq_pos": torch.from_numpy(
                np.array(structural_seq_pos, dtype=np.int64)
            ),
        }

    @staticmethod
    def _is_polymer_backbone_bond(
        mol_type: str, atom_name_a: str, atom_name_b: str
    ) -> bool:
        atom_pair = {atom_name_a, atom_name_b}
        if mol_type == "protein":
            return atom_pair == {"C", "N"}
        if mol_type in ["dna", "rna"]:
            return "P" in atom_pair and bool(
                atom_pair.intersection({r"O3'", "O3*", "O3T"})
            )
        return False

    def get_polymer_residue_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build parent-residue predecessor/successor indices from polymer bonds.

        The graph is defined over the residue-level token array. Real inter-residue
        backbone bonds are used first whenever a bond graph is available. A strict
        same-chain, adjacent-res_id fallback fills missing adjacent edges.
        """
        n_residue_tokens = len(self.cropped_token_array)
        prev_parent = np.full(n_residue_tokens, -1, dtype=np.int64)
        next_parent = np.full(n_residue_tokens, -1, dtype=np.int64)
        atom_to_parent = np.full(len(self.cropped_atom_array), -1, dtype=np.int64)
        for parent_idx, token in enumerate(self.cropped_token_array):
            for atom_idx in token.atom_indices:
                atom_to_parent[atom_idx] = parent_idx

        centre_atom_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atom_indices]
        parent_is_std_polymer = np.array(
            [
                atom.mol_type in ["protein", "dna", "rna"]
                and atom.res_name in STD_RESIDUES
                for atom in centre_atoms
            ],
            dtype=bool,
        )
        has_bond_graph = self.cropped_atom_array.bonds is not None
        graph_edges = set()
        if has_bond_graph:
            for bond in self.cropped_atom_array.bonds.as_array():
                atom_idx_a, atom_idx_b = int(bond[0]), int(bond[1])
                parent_a = atom_to_parent[atom_idx_a]
                parent_b = atom_to_parent[atom_idx_b]
                if parent_a < 0 or parent_b < 0 or parent_a == parent_b:
                    continue
                if not (
                    parent_is_std_polymer[parent_a] and parent_is_std_polymer[parent_b]
                ):
                    continue
                atom_a = self.cropped_atom_array[atom_idx_a]
                atom_b = self.cropped_atom_array[atom_idx_b]
                if atom_a.asym_id_int != atom_b.asym_id_int:
                    continue
                if atom_a.mol_type != atom_b.mol_type:
                    continue
                if not self._is_polymer_backbone_bond(
                    atom_a.mol_type, atom_a.atom_name, atom_b.atom_name
                ):
                    continue
                first_parent, second_parent = (
                    (parent_a, parent_b)
                    if parent_a < parent_b
                    else (parent_b, parent_a)
                )
                next_parent[first_parent] = second_parent
                prev_parent[second_parent] = first_parent
                graph_edges.add((int(first_parent), int(second_parent)))

        for parent_idx in range(n_residue_tokens - 1):
            next_idx = parent_idx + 1
            atom_i = centre_atoms[parent_idx]
            atom_j = centre_atoms[next_idx]
            if (parent_idx, next_idx) in graph_edges:
                continue
            if next_parent[parent_idx] >= 0 or prev_parent[next_idx] >= 0:
                continue
            if not (
                parent_is_std_polymer[parent_idx]
                and parent_is_std_polymer[next_idx]
                and atom_i.asym_id_int == atom_j.asym_id_int
                and atom_i.mol_type == atom_j.mol_type
                and atom_j.res_id - atom_i.res_id == 1
            ):
                continue
            next_parent[parent_idx] = next_idx
            prev_parent[next_idx] = parent_idx

        return prev_parent, next_parent

    @staticmethod
    def get_lig_pocket_mask(
        atom_array: AtomArray, lig_label_asym_id: Union[str, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ref: AlphaFold3 Chapter Methods.Metrics

        the pocket is defined as all heavy atoms within 10 Å of any heavy atom of the ligand,
        restricted to the primary polymer chain for the ligand or modified residue being scored,
        and further restricted to only backbone atoms for proteins. The primary polymer chain is defined variously:
        for PoseBusters it is the protein chain with the most atoms within 10 Å of the ligand,
        for bonded ligand scores it is the bonded polymer chain and for modified residues it
        is the chain that the residue is contained in (minus that residue).

        Args:
            atom_array (AtomArray): atoms in the complex.
            lig_label_asym_id (Union[str, List]): The label_asym_id of the ligand of interest.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of ligand pocket mask and pocket mask.
        """

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        prot_backbone = (
            atom_array.is_protein & np.isin(atom_array.atom_name, ["C", "N", "CA"])
        ).astype(bool)

        kdtree = KDTree(atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(lig_label_asym_id, atom_array.label_asym_id), (
                f"{lig_label_asym_id} is not in the label_asym_id of the cropped atom array."
            )

            ligand_mask = atom_array.label_asym_id == lig_label_asym_id
            lig_pos = atom_array.coord[ligand_mask & atom_array.is_resolved]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                np.concatenate(kdtree.query_radius(lig_pos, 10.0))
            )
            near_atoms = np.isin(
                np.arange(len(atom_array)), near_atom_indices
            ) & atom_array.is_resolved.astype(bool)

            # Get primary chain (protein backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & prot_backbone
            primary_chain_candidates_atoms = atom_array[primary_chain_candidates]

            max_atom = 0
            primary_chain_asym_id_int = None
            for asym_id_int in np.unique(primary_chain_candidates_atoms.asym_id_int):
                n_atoms = np.sum(
                    primary_chain_candidates_atoms.asym_id_int == asym_id_int
                )
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_asym_id_int = asym_id_int
            assert primary_chain_asym_id_int is not None, (
                f"No primary chain found for ligand ({lig_label_asym_id=})."
            )

            pocket_mask = primary_chain_candidates & (
                atom_array.asym_id_int == primary_chain_asym_id_int
            )
            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = torch.from_numpy(
            np.array(ligand_mask_list).astype(np.int64)
        )
        pocket_mask_by_pockets = torch.from_numpy(
            np.array(pocket_mask_list).astype(np.int64)
        )
        return ligand_mask_by_pockets, pocket_mask_by_pockets

    def get_mask_features(self) -> dict[str, torch.Tensor]:
        """
        Generate mask features for the cropped atom array.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various mask features.
        """
        mask_features = {}

        mask_features["pae_rep_atom_mask"] = torch.from_numpy(
            self.cropped_atom_array.centre_atom_mask.astype(np.int64)
        )

        mask_features["plddt_m_rep_atom_mask"] = torch.from_numpy(
            self.cropped_atom_array.plddt_m_rep_atom_mask.astype(np.int64)
        )  # [N_atom]

        mask_features["distogram_rep_atom_mask"] = torch.from_numpy(
            self.cropped_atom_array.distogram_rep_atom_mask.astype(np.int64)
        )  # [N_atom]

        mask_features["modified_res_mask"] = torch.from_numpy(
            self.cropped_atom_array.modified_res_mask.astype(np.int64)
        )

        lig_polymer_bonds = get_ligand_polymer_bond_mask(self.cropped_atom_array)
        num_atoms = len(self.cropped_atom_array)
        bond_mask_mat = np.zeros((num_atoms, num_atoms))
        for i, j, _ in lig_polymer_bonds:
            bond_mask_mat[i, j] = 1
            bond_mask_mat[j, i] = 1
        mask_features["bond_mask"] = torch.Tensor(
            bond_mask_mat
        ).long()  # [N_atom, N_atom]
        return mask_features

    def get_all_input_features(self):
        """
        Get input features from cropped data.

        Returns:
            Dict[str, torch.Tensor]: a dict of features.
        """
        features = {}
        token_features = self.get_token_features()
        features.update(token_features)

        bond_features = self.get_bond_features()
        features.update(bond_features)

        reference_features = self.get_reference_features()
        features.update(reference_features)

        extra_features = self.get_extra_features()
        features.update(extra_features)

        structural_token_features = self.get_structural_token_features(
            ref_pos=reference_features["ref_pos"],
            ref_mask=reference_features["ref_mask"],
        )
        features.update(structural_token_features)

        chain_perm_features = self.get_chain_perm_features()
        features.update(chain_perm_features)

        mask_features = self.get_mask_features()
        features.update(mask_features)
        return features

    @staticmethod
    def inherit_parent_backbone_frames(
        structural_token_array: TokenArray,
    ) -> TokenArray:
        """
        Give SC/BASE subtokens the same backbone-derived frame as their BB twin.

        PAE for polymers is defined relative to a backbone-derived residue frame.
        SC/BASE structural tokens are another view of the same parent residue, so
        they should use the parent BB frame instead of being left unsupervised.
        """
        backbone_roles = {"protein_bb", "dna_bb", "rna_bb"}
        child_roles = {"protein_sc", "dna_base", "rna_base"}
        parent_to_backbone_frame = {}
        for token in structural_token_array:
            if token.subtoken_role not in backbone_roles:
                continue
            parent_to_backbone_frame[token.parent_residue_idx] = (
                token.has_frame,
                list(token.frame_atom_index),
            )

        for token in structural_token_array:
            if token.subtoken_role not in child_roles:
                continue
            frame = parent_to_backbone_frame.get(token.parent_residue_idx)
            if frame is None:
                token.has_frame = 0
                token.frame_atom_index = [-1, -1, -1]
                continue
            token.has_frame = frame[0]
            token.frame_atom_index = list(frame[1])
        return structural_token_array
