# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Self, Sequence, TypeAlias

import numpy as np
from biotite.structure import AtomArray

from opendde.data.constants import (
    DNA_CHAIN,
    LIGAND_CHAIN_TYPES,
    PROTEIN_CHAIN,
    RNA_CHAIN,
)
from opendde.data.msa.msa_utils import map_to_standard
from opendde.data.template.template_parser import HHRParser, HmmsearchA3MParser
from opendde.data.template.template_utils import (
    TEMPLATE_FEATURES,
    DistogramFeaturesConfig,
    TemplateFeatures,
    TemplateHitFeaturizer,
)
from opendde.data.utils import pad_to
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

BatchDict: TypeAlias = dict[str, np.ndarray]
FeatureDict: TypeAlias = Mapping[str, np.ndarray]
MutableFeatureDict: TypeAlias = dict[str, Any]


class TemplateFeatureAssemblyLine:
    """
    Orchestrates the conversion of raw templates into finalized OpenDDE features.

    Args:
        max_templates: Maximum number of templates to include in the features.
    """

    def __init__(self, max_templates: int = 4) -> None:
        self.max_templates = max_templates

    def assemble(
        self,
        bioassembly: Mapping[int, Mapping[str, Any]],
        standard_token_idxs: np.ndarray,
    ) -> "Templates":
        """
        Executes the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            standard_token_idxs: Array of standardized residue indices.

        Returns:
            An assembled Templates object.
        """
        np_chains_list: list[MutableFeatureDict] = []
        polymer_entity_features: dict[bool, dict[Any, FeatureDict]] = {
            True: {},
            False: {},
        }
        # Identify entities where template features can be safely copied (same sequence)
        safe_entity_ids = get_safe_entity_id_for_template_copy(bioassembly)

        for asym_id, info in bioassembly.items():
            chain_id = info["chain_id"]
            entity_id = info["entity_id"]
            chain_type = info["chain_entity_type"]
            num_tokens = len(info["sequence"])

            # Templates are currently only supported for protein chains with sufficient length
            skip_chain = chain_type != PROTEIN_CHAIN or num_tokens <= 4

            if (entity_id not in polymer_entity_features[skip_chain]) or (
                entity_id not in safe_entity_ids
            ):
                templates = info["templates"]
                if skip_chain or not templates:
                    template_features = TemplateFeatures.empty_template_features(
                        num_tokens
                    )
                else:
                    # Package and fix template features
                    template_features = TemplateFeatures.package_template_features(
                        hit_features=templates
                    )
                    template_features = TemplateFeatures.fix_template_features(
                        template_features=template_features,
                        num_res=num_tokens,
                    )
                # Reduce to requested maximum number of templates
                template_features = _reduce_template_features(
                    template_features, self.max_templates
                )
                if entity_id in safe_entity_ids:
                    polymer_entity_features[skip_chain][entity_id] = template_features

            if entity_id in safe_entity_ids:
                feats: MutableFeatureDict = dict(
                    polymer_entity_features[skip_chain][entity_id]
                )
            else:
                feats = dict(template_features)

            feats["chain_id"] = chain_id
            np_chains_list.append(feats)

        # Pad the number of templates to max_templates for each chain to allow concatenation
        for chain in np_chains_list:
            chain["template_aatype"] = pad_to(
                chain["template_aatype"], (self.max_templates, None)
            )
            chain["template_atom_positions"] = pad_to(
                chain["template_atom_positions"],
                (self.max_templates, None, None, None),
            )
            chain["template_atom_mask"] = pad_to(
                chain["template_atom_mask"], (self.max_templates, None, None)
            )

        # Concatenate features along the residue dimension
        merged_example = {
            ft: np.concatenate([c[ft] for c in np_chains_list], axis=1)
            for ft in np_chains_list[0]
            if ft in TEMPLATE_FEATURES
        }

        # Crop/index merged features using standard token indices
        for feature_name, v in merged_example.items():
            merged_example[feature_name] = v[
                : self.max_templates, standard_token_idxs, ...
            ]

        return Templates(
            aatype=merged_example["template_aatype"],
            atom_positions=merged_example["template_atom_positions"],
            atom_mask=merged_example["template_atom_mask"].astype(bool),
        )


@dataclasses.dataclass(frozen=True)
class Templates:
    """Dataclass containing template features."""

    # aatype: [num_templates, num_res]
    aatype: np.ndarray
    # atom_positions: [num_templates, num_res, 24, 3]
    atom_positions: np.ndarray
    # atom_mask: [num_templates, num_res, 24]
    atom_mask: np.ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Construct instance from a data dictionary."""
        return cls(
            aatype=batch["template_aatype"],
            atom_positions=batch["template_atom_positions"],
            atom_mask=batch["template_atom_mask"],
        )

    def as_data_dict(self) -> BatchDict:
        """Convert to a standard data dictionary."""
        return {
            "template_aatype": self.aatype,
            "template_atom_positions": self.atom_positions,
            "template_atom_mask": self.atom_mask,
        }

    # Shared config instance to avoid repeated object creation
    _DGRAM_CONFIG = DistogramFeaturesConfig(min_bin=3.25, max_bin=50.75, num_bins=39)

    def as_opendde_dict(self) -> BatchDict:
        """Compute additional features and return as OpenDDE dictionary."""
        features = self.as_data_dict()
        num_templates = self.aatype.shape[0]
        num_res = self.aatype.shape[1]

        # Pre-allocate output arrays instead of list append + stack
        all_pb_masks = np.empty((num_templates, num_res, num_res), dtype=np.float32)
        all_dgrams = np.empty((num_templates, num_res, num_res, 39), dtype=np.float32)
        all_unit_vectors = np.empty(
            (num_templates, num_res, num_res, 3), dtype=np.float32
        )
        all_bb_masks = np.empty((num_templates, num_res, num_res), dtype=np.float32)

        config = Templates._DGRAM_CONFIG
        for i in range(num_templates):
            aatype = self.aatype[i]
            mask = self.atom_mask[i]
            pos = self.atom_positions[i] * mask[..., None]

            pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(aatype, pos, mask)
            pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]

            dgram = TemplateFeatures.dgram_from_positions(pb_pos, config=config)
            all_dgrams[i] = dgram * pb_mask_2d[..., None]
            all_pb_masks[i] = pb_mask_2d

            uv, bb_mask_2d = TemplateFeatures.compute_template_unit_vector(
                aatype, pos, mask
            )
            all_unit_vectors[i] = uv * bb_mask_2d[..., None]
            all_bb_masks[i] = bb_mask_2d

        features.update(
            {
                "template_pseudo_beta_mask": all_pb_masks,
                "template_distogram": all_dgrams,
                "template_unit_vector": all_unit_vectors,
                "template_backbone_frame_mask": all_bb_masks,
            }
        )
        return features


def _reduce_template_features(
    template_features: FeatureDict, max_templates: int
) -> FeatureDict:
    """Reduces templates to the requested maximum number."""
    num_t = template_features["template_aatype"].shape[0]
    keep_mask = np.arange(num_t) < max_templates
    fields = TEMPLATE_FEATURES
    return {k: v[keep_mask] for k, v in template_features.items() if k in fields}


def get_safe_entity_id_for_template_copy(
    bioassembly: Mapping[int, Mapping[str, Any]],
) -> List[str]:
    """Identifies entity IDs that have consistent sequences across all chains."""
    eid_to_seqs = {}
    for aid, info in bioassembly.items():
        eid = info["entity_id"]
        seq = info["sequence"]
        eid_to_seqs.setdefault(eid, set()).add(seq)
    return [eid for eid, seqs in eid_to_seqs.items() if len(seqs) == 1]


class InferenceTemplateFeaturizer:
    """Simplified featurizer for inference, leveraging the same assembly logic."""

    @staticmethod
    def make_template_feature(
        bioassembly: Sequence[Mapping[str, Any]],
        atom_array: AtomArray,
        use_template: bool = True,
        online_template_featurizer: Optional[TemplateHitFeaturizer] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generates template features during inference.

        Args:
            bioassembly: List of entity information from the input JSON.
            atom_array: Parsed atom structure.
            use_template: Whether to use templates.
            online_template_featurizer: Featurizer for processing template hits.

        Returns:
            Dictionary of template features.
        """
        if use_template:
            logger.info("Building template features for inference")
        else:
            logger.debug("Building empty template feature placeholders")
        template_meta_infos = {}
        curr_asym_id = 0

        for eid, info in enumerate(bioassembly):
            seq, count, ctype, t_path = "", 0, LIGAND_CHAIN_TYPES, ""

            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype, t_path = (
                    c["sequence"],
                    c["count"],
                    PROTEIN_CHAIN,
                    c.get("templatesPath", ""),
                )
            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            elif "ligand" in info:
                count, ctype = info["ligand"]["count"], LIGAND_CHAIN_TYPES
                seq = "X" * (atom_array.asym_id_int == curr_asym_id).sum()

            templates = []
            if t_path and use_template and online_template_featurizer:
                assert ctype == PROTEIN_CHAIN, "Only protein templates are supported."
                with open(t_path, "r") as f:
                    content = f.read()

                if t_path.endswith(".hhr"):
                    hits = HHRParser.parse(hhr_string=content)
                elif t_path.endswith(".a3m"):
                    hits = HmmsearchA3MParser.parse(
                        query_seq=seq, a3m_str=content, skip_first=False
                    )
                else:
                    raise ValueError(f"Unsupported template format: {t_path}")

                result, _ = online_template_featurizer.get_templates(
                    sequence_uid=seq,
                    query_sequence=seq,
                    hits=hits,
                    max_template_date=None,
                )
                templates = result.features
                logger.info(f"Found {len(templates)} templates for sequence {seq}")

            for i in range(count):
                aid = curr_asym_id + i
                template_meta_infos[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "templates": templates,
                }
            curr_asym_id += count

        # Coordinate mapping
        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, template_meta_infos)

        # Assemble features
        return (
            TemplateFeatureAssemblyLine(max_templates=4)
            .assemble(template_meta_infos, std_idxs)
            .as_opendde_dict()
        )
