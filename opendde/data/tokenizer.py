# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from opendde.data.constants import ELEMS, STD_RESIDUES


STRUCTURAL_TOKEN_ROLES = {
    "atom": 0,
    "protein_bb": 1,
    "protein_sc": 2,
    "dna_bb": 3,
    "dna_base": 4,
    "rna_bb": 5,
    "rna_base": 6,
}

NO_TWIN_TOKEN_IDX = -1
PROTEIN_BACKBONE_ATOMS = frozenset(["N", "CA", "C", "O", "OXT"])
NUCLEIC_BACKBONE_ATOMS = frozenset(
    [
        "P",
        "OP1",
        "OP2",
        "OP3",
        "O1P",
        "O2P",
        "O3P",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O5*",
        "C5*",
        "C4*",
        "O4*",
        "C3*",
        "O3*",
        "C2*",
        "O2*",
        "C1*",
        "O5T",
        "O3T",
    ]
)
PURINE_RESIDUES = frozenset(["DA", "DG", "A", "G"])
PYRIMIDINE_RESIDUES = frozenset(["DC", "DT", "C", "U"])


class Token(object):
    """
    Used to store information related to Tokens.

    Example:
    >>> token = Token(1)
    >>> token.value
    1
    >>> token.atom_indices = [1, 2, 3]
    """

    def __init__(self, value, **kwargs):
        self.value = value
        self._annot = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr):
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __repr__(self):
        annot_lst = []
        for k, v in self._annot.items():
            annot_lst.append(f"{k}={v}")
        return f"Token({self.value}, {','.join(annot_lst)})"

    def __setattr__(self, attr, value):
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value


class TokenArray(object):
    """
    A group of Token objects used for batch operations.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def __repr__(self):
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tokens[index]
        else:
            return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category):
        return [token._annot[category] for token in self.tokens]

    def set_annotation(self, category, values):
        assert len(values) == len(self.tokens), (
            "Length of values must match the number of tokens"
        )
        for token, value in zip(self.tokens, values):
            token._annot[category] = value

    def get_values(self):
        return [token.value for token in self.tokens]


class AtomArrayTokenizer(object):
    """
    Tokenize an AtomArray object into a list of Token object.
    """

    def __init__(self, atom_array: AtomArray):
        self.atom_array = atom_array

    @staticmethod
    def _make_structural_token(
        value: int,
        atom_indices: list[int],
        atom_names: list[str],
        centre_atom_index: int,
        subtoken_role: str,
        parent_residue_idx: int,
        residue_token_group_id: int,
        twin_token_idx: int = NO_TWIN_TOKEN_IDX,
    ) -> Token:
        token = Token(value)
        token.atom_indices = atom_indices
        token.atom_names = atom_names
        token.centre_atom_index = centre_atom_index
        token.subtoken_role = subtoken_role
        token.subtoken_role_id = STRUCTURAL_TOKEN_ROLES[subtoken_role]
        token.parent_residue_idx = parent_residue_idx
        token.residue_token_group_id = residue_token_group_id
        token.twin_token_idx = twin_token_idx
        return token

    @staticmethod
    def _split_atom_pairs(
        atom_indices: list[int],
        atom_names: list[str],
        group_atom_names: frozenset[str],
    ) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        in_group = []
        out_group = []
        for atom_idx, atom_name in zip(atom_indices, atom_names):
            if atom_name in group_atom_names:
                in_group.append((atom_idx, atom_name))
            else:
                out_group.append((atom_idx, atom_name))
        return in_group, out_group

    @staticmethod
    def _choose_centre_atom_index(
        atom_pairs: list[tuple[int, str]], preferred_atom_names: list[str]
    ) -> int:
        for preferred_atom_name in preferred_atom_names:
            for atom_idx, atom_name in atom_pairs:
                if atom_name == preferred_atom_name:
                    return atom_idx
        return atom_pairs[0][0]

    @staticmethod
    def _pairs_to_lists(
        atom_pairs: list[tuple[int, str]],
    ) -> tuple[list[int], list[str]]:
        atom_indices = [atom_idx for atom_idx, _ in atom_pairs]
        atom_names = [atom_name for _, atom_name in atom_pairs]
        return atom_indices, atom_names

    @staticmethod
    def _base_centre_atom_preferences(res_name: str) -> list[str]:
        if res_name in PURINE_RESIDUES:
            return ["N9", "C4", "C8", "N7", "C5"]
        if res_name in PYRIMIDINE_RESIDUES:
            return ["N1", "C2", "C6", "C5", "C4"]
        return ["C1'", "C1*", "N9", "N1"]

    def tokenize(self) -> list[Token]:
        """
        Ref: AlphaFold3 SI Chapter 2.6
        Tokenize an AtomArray object into a list of Token object.

        Returns:
           list : a list of Token object.
        """
        tokens = []
        total_atom_num = 0
        for res in struc.residue_iter(self.atom_array):
            atom_num = len(res)
            first_atom = res[0]
            res_name = first_atom.res_name
            mol_type = first_atom.mol_type
            res_token = STD_RESIDUES.get(res_name, None)
            if res_token is not None and mol_type != "ligand":
                # for std residues
                token = Token(res_token)
                atom_indices = [
                    i for i in range(total_atom_num, total_atom_num + atom_num)
                ]
                atom_names = [self.atom_array[i].atom_name for i in atom_indices]
                token.atom_indices = atom_indices
                token.atom_names = atom_names
                tokens.append(token)
                total_atom_num += atom_num
            else:
                # for ligand and non-std residues
                for atom in res:
                    atom_elem = atom.element
                    atom_token = ELEMS.get(atom_elem, None)
                    if atom_token is None:
                        raise ValueError(f"Unknown atom element: {atom_elem}")
                    token = Token(atom_token)
                    token.atom_indices = [total_atom_num]
                    token.atom_names = [
                        self.atom_array[token.atom_indices[0]].atom_name
                    ]
                    tokens.append(token)
                    total_atom_num += 1

        assert total_atom_num == len(self.atom_array)
        return tokens

    def _get_atom_token_from_parent(self, token: Token, parent_idx: int) -> Token:
        return self._make_structural_token(
            value=token.value,
            atom_indices=list(token.atom_indices),
            atom_names=list(token.atom_names),
            centre_atom_index=token.centre_atom_index,
            subtoken_role="atom",
            parent_residue_idx=parent_idx,
            residue_token_group_id=parent_idx,
        )

    def _split_standard_protein_token(
        self, token: Token, parent_idx: int
    ) -> list[Token]:
        centre_atom = self.atom_array[token.centre_atom_index]
        atom_indices = list(token.atom_indices)
        atom_names = list(token.atom_names)
        backbone_pairs, sidechain_pairs = self._split_atom_pairs(
            atom_indices, atom_names, PROTEIN_BACKBONE_ATOMS
        )

        # Glycine remains a single structural token in v1: empty sidechain tokens
        # would require invasive empty-token handling in downstream atom layouts.
        if centre_atom.res_name == "GLY":
            backbone_pairs = list(zip(atom_indices, atom_names))
            sidechain_pairs = []
        elif len(backbone_pairs) == 0:
            backbone_pairs = list(zip(atom_indices, atom_names))
            sidechain_pairs = []

        structural_tokens = []
        if backbone_pairs:
            bb_atom_indices, bb_atom_names = self._pairs_to_lists(backbone_pairs)
            structural_tokens.append(
                self._make_structural_token(
                    value=token.value,
                    atom_indices=bb_atom_indices,
                    atom_names=bb_atom_names,
                    centre_atom_index=self._choose_centre_atom_index(
                        backbone_pairs, ["CA", "N", "C"]
                    ),
                    subtoken_role="protein_bb",
                    parent_residue_idx=parent_idx,
                    residue_token_group_id=parent_idx,
                )
            )
        if sidechain_pairs:
            sc_atom_indices, sc_atom_names = self._pairs_to_lists(sidechain_pairs)
            structural_tokens.append(
                self._make_structural_token(
                    value=token.value,
                    atom_indices=sc_atom_indices,
                    atom_names=sc_atom_names,
                    centre_atom_index=self._choose_centre_atom_index(
                        sidechain_pairs, ["CB"]
                    ),
                    subtoken_role="protein_sc",
                    parent_residue_idx=parent_idx,
                    residue_token_group_id=parent_idx,
                )
            )
        return self._set_twin_indices(structural_tokens)

    def _split_standard_nucleic_token(
        self, token: Token, parent_idx: int, mol_type: str
    ) -> list[Token]:
        centre_atom = self.atom_array[token.centre_atom_index]
        atom_indices = list(token.atom_indices)
        atom_names = list(token.atom_names)
        backbone_pairs, base_pairs = self._split_atom_pairs(
            atom_indices, atom_names, NUCLEIC_BACKBONE_ATOMS
        )
        if len(backbone_pairs) == 0:
            backbone_pairs = list(zip(atom_indices, atom_names))
            base_pairs = []

        bb_role = f"{mol_type}_bb"
        base_role = f"{mol_type}_base"
        structural_tokens = []
        if backbone_pairs:
            bb_atom_indices, bb_atom_names = self._pairs_to_lists(backbone_pairs)
            structural_tokens.append(
                self._make_structural_token(
                    value=token.value,
                    atom_indices=bb_atom_indices,
                    atom_names=bb_atom_names,
                    centre_atom_index=self._choose_centre_atom_index(
                        backbone_pairs, ["C4'", "C4*", "C1'", "C1*"]
                    ),
                    subtoken_role=bb_role,
                    parent_residue_idx=parent_idx,
                    residue_token_group_id=parent_idx,
                )
            )
        if base_pairs:
            base_atom_indices, base_atom_names = self._pairs_to_lists(base_pairs)
            structural_tokens.append(
                self._make_structural_token(
                    value=token.value,
                    atom_indices=base_atom_indices,
                    atom_names=base_atom_names,
                    centre_atom_index=self._choose_centre_atom_index(
                        base_pairs,
                        self._base_centre_atom_preferences(centre_atom.res_name),
                    ),
                    subtoken_role=base_role,
                    parent_residue_idx=parent_idx,
                    residue_token_group_id=parent_idx,
                )
            )
        return self._set_twin_indices(structural_tokens)

    @staticmethod
    def _set_twin_indices(tokens: list[Token]) -> list[Token]:
        if len(tokens) == 2:
            tokens[0].twin_token_idx = 1
            tokens[1].twin_token_idx = 0
        return tokens

    def tokenize_structural(
        self, residue_token_array: TokenArray | None = None
    ) -> list[Token]:
        """
        Expand residue-level tokens into structural tokens.

        Standard protein residues become backbone/sidechain tokens, except glycine
        and residues without sidechain atoms, which stay as a single backbone token.
        Standard DNA/RNA residues become backbone/base tokens when both atom groups
        are present. Ligands, modifications, and pre-expanded atom-level tokens stay
        as atom tokens.
        """
        if residue_token_array is None:
            residue_token_array = self.get_token_array()

        structural_tokens = []
        for parent_idx, token in enumerate(residue_token_array):
            centre_atom = self.atom_array[token.centre_atom_index]
            if (
                centre_atom.res_name in STD_RESIDUES
                and centre_atom.mol_type != "ligand"
                and len(token.atom_indices) > 1
            ):
                if centre_atom.mol_type == "protein":
                    new_tokens = self._split_standard_protein_token(token, parent_idx)
                elif centre_atom.mol_type in ["dna", "rna"]:
                    new_tokens = self._split_standard_nucleic_token(
                        token, parent_idx, centre_atom.mol_type
                    )
                else:
                    new_tokens = [self._get_atom_token_from_parent(token, parent_idx)]
            else:
                new_tokens = [self._get_atom_token_from_parent(token, parent_idx)]

            token_offset = len(structural_tokens)
            for new_token in new_tokens:
                if new_token.twin_token_idx != NO_TWIN_TOKEN_IDX:
                    new_token.twin_token_idx += token_offset
                structural_tokens.append(new_token)

        return structural_tokens

    def _set_token_annotations(self, token_array: TokenArray) -> TokenArray:
        """
        Set annotations for the token_array.

        The annotations include:
            - centre_atom_index: the atom indices of the token in the atom array

        Args:
            token_array (TokenArray): TokenArray object created by tokenize bioassembly AtomArray.

        Returns:
            TokenArray: TokenArray object with annotations.
        """
        centre_atom_indices = np.where(self.atom_array.centre_atom_mask == 1)[0]
        token_array.set_annotation("centre_atom_index", centre_atom_indices)
        assert len(token_array) == len(centre_atom_indices)
        return token_array

    def get_token_array(self) -> TokenArray:
        """
        Get TokenArray object with annotations (atom_indices, centre_atom_index).

        Returns:
            TokenArray: The TokenArray object with annotations.
                TokenArray(
                Token(1, atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],centre_atom_index=2,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'])
                Token(15, atom_indices=[11, 12, 13, 14, 15, 16],centre_atom_index=13,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'OG'])
                Token(15, atom_indices=[17, 18, 19, 20, 21, 22],centre_atom_index=19,
                    atom_names=['N', 'CA', 'C', 'O', 'CB', 'OG'])
                    )
                it satisfy the following format
                Token($token_index,  atom_indices=[global_atom_indexs],
                    centre_atom_index=global_atom_indexs,atom_names=[names])
        """
        tokens = self.tokenize()
        token_array = TokenArray(tokens=tokens)
        token_array = self._set_token_annotations(token_array=token_array)
        return token_array

    def get_structural_token_array(
        self, residue_token_array: TokenArray | None = None
    ) -> TokenArray:
        """
        Get structural tokens for the post-trunk structure branch.

        The returned TokenArray carries these annotations per structural token:
            - centre_atom_index
            - subtoken_role / subtoken_role_id
            - parent_residue_idx
            - residue_token_group_id
            - twin_token_idx
        """
        tokens = self.tokenize_structural(residue_token_array=residue_token_array)
        return TokenArray(tokens=tokens)
