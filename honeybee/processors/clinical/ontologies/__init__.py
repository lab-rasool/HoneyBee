"""
Ontology mappings for clinical entity normalization.

Provides aggregated ONTOLOGY_MAPPINGS and lookup utilities.
"""

import difflib
from typing import Dict, List, Optional

from .loinc import LOINC_MAPPINGS
from .rxnorm import RXNORM_MAPPINGS
from .snomed_ct import SNOMED_CT_MAPPINGS

# Aggregated ontology mappings (same structure as original ONTOLOGY_MAPPINGS)
ONTOLOGY_MAPPINGS = {
    "snomed_ct": SNOMED_CT_MAPPINGS,
    "rxnorm": RXNORM_MAPPINGS,
    "loinc": LOINC_MAPPINGS,
}


def lookup(text: str, ontology: Optional[str] = None) -> List[Dict]:
    """Exact case-insensitive lookup across ontologies.

    Args:
        text: Term to look up.
        ontology: Optional ontology name to restrict search.

    Returns:
        List of matching entries with ontology, concept_id, concept_name.
    """
    normalized = text.lower().strip()
    results = []

    ontologies = (
        {ontology: ONTOLOGY_MAPPINGS[ontology]}
        if ontology and ontology in ONTOLOGY_MAPPINGS
        else ONTOLOGY_MAPPINGS
    )

    for ont_name, ont_dict in ontologies.items():
        mapping = ont_dict.get(normalized)
        if mapping:
            results.append(
                {
                    "ontology": ont_name,
                    "concept_id": mapping["id"],
                    "concept_name": mapping["name"],
                    "match_type": "exact",
                }
            )

    return results


def fuzzy_lookup(
    text: str,
    ontology: Optional[str] = None,
    threshold: float = 0.85,
    max_results: int = 5,
) -> List[Dict]:
    """Fuzzy lookup using difflib.SequenceMatcher (no extra deps).

    Args:
        text: Term to look up.
        ontology: Optional ontology name to restrict search.
        threshold: Minimum similarity ratio (0.0-1.0).
        max_results: Maximum number of results to return.

    Returns:
        List of matching entries sorted by similarity score.
    """
    normalized = text.lower().strip()
    results = []

    ontologies = (
        {ontology: ONTOLOGY_MAPPINGS[ontology]}
        if ontology and ontology in ONTOLOGY_MAPPINGS
        else ONTOLOGY_MAPPINGS
    )

    for ont_name, ont_dict in ontologies.items():
        for term, mapping in ont_dict.items():
            ratio = difflib.SequenceMatcher(
                None, normalized, term
            ).ratio()
            if ratio >= threshold:
                results.append(
                    {
                        "ontology": ont_name,
                        "concept_id": mapping["id"],
                        "concept_name": mapping["name"],
                        "match_type": "fuzzy",
                        "similarity": round(ratio, 3),
                        "matched_term": term,
                    }
                )

    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]
