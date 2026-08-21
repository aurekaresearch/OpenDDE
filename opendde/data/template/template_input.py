# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from collections.abc import Mapping
from typing import Any


def get_explicit_templates(protein_chain: Mapping[str, Any]) -> list[Any] | None:
    """Return explicit templates, or None when the search pipeline applies."""
    if "templates" not in protein_chain:
        return None
    if protein_chain.get("templatesPath"):
        raise ValueError(
            "proteinChain accepts either explicit templates or templatesPath, not both."
        )
    explicit_templates = protein_chain["templates"]
    if not isinstance(explicit_templates, list):
        raise TypeError("proteinChain.templates must be a list.")
    return explicit_templates


def needs_template_search(input_json_data: Any) -> bool:
    """Return whether any protein chain needs search template infrastructure."""
    jobs = input_json_data if isinstance(input_json_data, list) else [input_json_data]

    for infer_data in jobs:
        if not isinstance(infer_data, Mapping):
            continue
        for sequence in infer_data.get("sequences", []):
            if not isinstance(sequence, Mapping):
                continue
            protein_chain = sequence.get("proteinChain")
            if not isinstance(protein_chain, Mapping):
                continue
            if "templates" not in protein_chain:
                return True
    return False
