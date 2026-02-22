"""
Unit tests for ontology mappings subpackage.

Tests SNOMED-CT, RxNorm, LOINC mappings, lookup, and fuzzy_lookup.
"""

import pytest

from honeybee.processors.clinical.ontologies import (
    ONTOLOGY_MAPPINGS,
    fuzzy_lookup,
    lookup,
)
from honeybee.processors.clinical.ontologies.loinc import LOINC_MAPPINGS
from honeybee.processors.clinical.ontologies.rxnorm import RXNORM_MAPPINGS
from honeybee.processors.clinical.ontologies.snomed_ct import SNOMED_CT_MAPPINGS


class TestSNOMEDCTMappings:
    """Test SNOMED-CT ontology mappings."""

    def test_minimum_entry_count(self):
        """SNOMED-CT should have at least 150 entries."""
        assert len(SNOMED_CT_MAPPINGS) >= 150

    def test_all_entries_have_id(self):
        """Every entry must have an 'id' field."""
        for term, mapping in SNOMED_CT_MAPPINGS.items():
            assert "id" in mapping, f"'{term}' missing 'id'"

    def test_all_entries_have_name(self):
        """Every entry must have a 'name' field."""
        for term, mapping in SNOMED_CT_MAPPINGS.items():
            assert "name" in mapping, f"'{term}' missing 'name'"

    def test_ids_are_strings(self):
        """All IDs should be strings."""
        for term, mapping in SNOMED_CT_MAPPINGS.items():
            assert isinstance(mapping["id"], str), f"'{term}' id not str"

    def test_names_are_strings(self):
        """All names should be strings."""
        for term, mapping in SNOMED_CT_MAPPINGS.items():
            assert isinstance(mapping["name"], str), f"'{term}' name not str"

    def test_known_cancer_types(self):
        """Spot-check known cancer types."""
        assert "breast cancer" in SNOMED_CT_MAPPINGS
        assert SNOMED_CT_MAPPINGS["breast cancer"]["id"] == "254838004"
        assert "lung cancer" in SNOMED_CT_MAPPINGS
        assert "colon cancer" in SNOMED_CT_MAPPINGS
        assert "prostate cancer" in SNOMED_CT_MAPPINGS
        assert "melanoma" in SNOMED_CT_MAPPINGS
        assert "lymphoma" in SNOMED_CT_MAPPINGS
        assert "leukemia" in SNOMED_CT_MAPPINGS

    def test_known_procedures(self):
        """Spot-check known procedures."""
        assert "chemotherapy" in SNOMED_CT_MAPPINGS
        assert SNOMED_CT_MAPPINGS["chemotherapy"]["id"] == "367336001"
        assert "radiation therapy" in SNOMED_CT_MAPPINGS
        assert "mastectomy" in SNOMED_CT_MAPPINGS
        assert "biopsy" in SNOMED_CT_MAPPINGS
        assert "immunotherapy" in SNOMED_CT_MAPPINGS

    def test_known_symptoms(self):
        """Spot-check known symptoms."""
        assert "pain" in SNOMED_CT_MAPPINGS
        assert "fatigue" in SNOMED_CT_MAPPINGS
        assert "nausea" in SNOMED_CT_MAPPINGS
        assert "fever" in SNOMED_CT_MAPPINGS
        assert "weight loss" in SNOMED_CT_MAPPINGS

    def test_known_anatomy(self):
        """Spot-check known anatomical sites."""
        assert "lung" in SNOMED_CT_MAPPINGS
        assert "breast" in SNOMED_CT_MAPPINGS
        assert "liver" in SNOMED_CT_MAPPINGS
        assert "brain" in SNOMED_CT_MAPPINGS

    def test_known_staging(self):
        """Spot-check staging entries."""
        assert "stage I" in SNOMED_CT_MAPPINGS
        assert "stage IV" in SNOMED_CT_MAPPINGS
        assert "grade 1" in SNOMED_CT_MAPPINGS

    def test_known_biomarkers(self):
        """Spot-check biomarker entries."""
        assert "HER2" in SNOMED_CT_MAPPINGS
        assert "EGFR" in SNOMED_CT_MAPPINGS
        assert "PD-L1" in SNOMED_CT_MAPPINGS
        assert "Ki-67" in SNOMED_CT_MAPPINGS

    def test_backward_compat_original_entries(self):
        """Original 4 SNOMED-CT entries must still exist."""
        assert "breast cancer" in SNOMED_CT_MAPPINGS
        assert "lung cancer" in SNOMED_CT_MAPPINGS
        assert "chemotherapy" in SNOMED_CT_MAPPINGS
        assert "tamoxifen" in SNOMED_CT_MAPPINGS

    def test_has_cancer_categories(self):
        """Should have entries across multiple cancer categories."""
        cancer_terms = [k for k in SNOMED_CT_MAPPINGS if "cancer" in k or "carcinoma" in k]
        assert len(cancer_terms) >= 10

    def test_has_procedure_entries(self):
        """Should have at least 20 procedure entries."""
        procedure_terms = [
            k for k in SNOMED_CT_MAPPINGS
            if any(p in k for p in ["therapy", "ectomy", "biopsy", "surgery", "ablation"])
        ]
        assert len(procedure_terms) >= 10


class TestRxNormMappings:
    """Test RxNorm ontology mappings."""

    def test_minimum_entry_count(self):
        """RxNorm should have at least 60 entries."""
        assert len(RXNORM_MAPPINGS) >= 60

    def test_all_entries_have_id(self):
        """Every entry must have an 'id' field."""
        for term, mapping in RXNORM_MAPPINGS.items():
            assert "id" in mapping, f"'{term}' missing 'id'"

    def test_all_entries_have_name(self):
        """Every entry must have a 'name' field."""
        for term, mapping in RXNORM_MAPPINGS.items():
            assert "name" in mapping, f"'{term}' missing 'name'"

    def test_known_chemo_agents(self):
        """Spot-check known chemotherapy agents."""
        assert "cisplatin" in RXNORM_MAPPINGS
        assert "doxorubicin" in RXNORM_MAPPINGS
        assert "paclitaxel" in RXNORM_MAPPINGS
        assert "cyclophosphamide" in RXNORM_MAPPINGS
        assert "gemcitabine" in RXNORM_MAPPINGS
        assert "fluorouracil" in RXNORM_MAPPINGS

    def test_known_targeted_therapy(self):
        """Spot-check targeted therapy agents."""
        assert "trastuzumab" in RXNORM_MAPPINGS
        assert "bevacizumab" in RXNORM_MAPPINGS
        assert "imatinib" in RXNORM_MAPPINGS
        assert "erlotinib" in RXNORM_MAPPINGS

    def test_known_immunotherapy(self):
        """Spot-check immunotherapy agents."""
        assert "pembrolizumab" in RXNORM_MAPPINGS
        assert "ipilimumab" in RXNORM_MAPPINGS
        assert "atezolizumab" in RXNORM_MAPPINGS

    def test_known_hormone_therapy(self):
        """Spot-check hormone therapy agents."""
        assert "tamoxifen" in RXNORM_MAPPINGS
        assert "letrozole" in RXNORM_MAPPINGS
        assert "anastrozole" in RXNORM_MAPPINGS

    def test_known_supportive_care(self):
        """Spot-check supportive care medications."""
        assert "ondansetron" in RXNORM_MAPPINGS
        assert "dexamethasone" in RXNORM_MAPPINGS
        assert "filgrastim" in RXNORM_MAPPINGS
        assert "morphine" in RXNORM_MAPPINGS

    def test_backward_compat_original_entries(self):
        """Original 3 RxNorm entries must still exist."""
        assert "tamoxifen" in RXNORM_MAPPINGS
        assert "carboplatin" in RXNORM_MAPPINGS
        assert "paclitaxel" in RXNORM_MAPPINGS


class TestLOINCMappings:
    """Test LOINC ontology mappings."""

    def test_minimum_entry_count(self):
        """LOINC should have at least 10 entries (original)."""
        assert len(LOINC_MAPPINGS) >= 10

    def test_all_entries_have_id(self):
        """Every entry must have an 'id' field."""
        for term, mapping in LOINC_MAPPINGS.items():
            assert "id" in mapping, f"'{term}' missing 'id'"

    def test_all_entries_have_name(self):
        """Every entry must have a 'name' field."""
        for term, mapping in LOINC_MAPPINGS.items():
            assert "name" in mapping, f"'{term}' missing 'name'"

    def test_original_entries_preserved(self):
        """Original 10 LOINC entries must be preserved."""
        expected = [
            "hemoglobin", "white blood cell count", "platelet count",
            "creatinine", "glucose", "albumin", "bilirubin",
            "calcium", "potassium", "sodium",
        ]
        for term in expected:
            assert term in LOINC_MAPPINGS, f"Original LOINC entry '{term}' missing"

    def test_hemoglobin_id(self):
        """Hemoglobin should have correct LOINC ID."""
        assert LOINC_MAPPINGS["hemoglobin"]["id"] == "718-7"

    def test_creatinine_id(self):
        """Creatinine should have correct LOINC ID."""
        assert LOINC_MAPPINGS["creatinine"]["id"] == "2160-0"

    def test_has_tumor_markers(self):
        """Should include tumor marker entries."""
        tumor_markers = ["carcinoembryonic antigen", "alpha-fetoprotein", "prostate specific antigen"]
        for marker in tumor_markers:
            assert marker in LOINC_MAPPINGS, f"Tumor marker '{marker}' missing"


class TestOntologyLookup:
    """Test exact lookup function."""

    def test_exact_lookup_snomed(self):
        """Exact lookup in SNOMED-CT."""
        results = lookup("breast cancer", ontology="snomed_ct")
        assert len(results) >= 1
        assert results[0]["ontology"] == "snomed_ct"
        assert results[0]["concept_id"] == "254838004"
        assert results[0]["match_type"] == "exact"

    def test_exact_lookup_rxnorm(self):
        """Exact lookup in RxNorm."""
        results = lookup("cisplatin", ontology="rxnorm")
        assert len(results) >= 1
        assert results[0]["ontology"] == "rxnorm"

    def test_exact_lookup_loinc(self):
        """Exact lookup in LOINC."""
        results = lookup("hemoglobin", ontology="loinc")
        assert len(results) >= 1
        assert results[0]["ontology"] == "loinc"
        assert results[0]["concept_id"] == "718-7"

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        results = lookup("Breast Cancer")
        assert len(results) >= 1

    def test_lookup_across_ontologies(self):
        """Lookup without ontology filter searches all."""
        # tamoxifen should be in both SNOMED and RxNorm
        results = lookup("tamoxifen")
        ontologies = {r["ontology"] for r in results}
        assert len(ontologies) >= 2

    def test_lookup_not_found(self):
        """Lookup for non-existent term returns empty."""
        results = lookup("xyznonexistent")
        assert results == []

    def test_lookup_returns_list(self):
        """Lookup always returns a list."""
        results = lookup("breast cancer")
        assert isinstance(results, list)


class TestFuzzyLookup:
    """Test fuzzy lookup function."""

    def test_fuzzy_close_match(self):
        """Fuzzy lookup should find close matches."""
        results = fuzzy_lookup("brest cancer", threshold=0.8)
        assert len(results) >= 1
        terms = [r.get("matched_term", "") for r in results]
        assert "breast cancer" in terms

    def test_fuzzy_threshold(self):
        """Higher threshold should return fewer results."""
        low = fuzzy_lookup("cancer", threshold=0.5)
        high = fuzzy_lookup("cancer", threshold=0.95)
        assert len(low) >= len(high)

    def test_fuzzy_returns_similarity(self):
        """Fuzzy results should include similarity score."""
        results = fuzzy_lookup("breast cancer", threshold=0.8)
        if results:
            assert "similarity" in results[0]
            assert results[0]["similarity"] >= 0.8

    def test_fuzzy_match_type(self):
        """Fuzzy results should have match_type=fuzzy."""
        results = fuzzy_lookup("breast cancer", threshold=0.8)
        if results:
            assert results[0]["match_type"] == "fuzzy"

    def test_fuzzy_max_results(self):
        """Should respect max_results parameter."""
        results = fuzzy_lookup("cancer", threshold=0.5, max_results=3)
        assert len(results) <= 3

    def test_fuzzy_sorted_by_similarity(self):
        """Results should be sorted by similarity descending."""
        results = fuzzy_lookup("cancer", threshold=0.5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_fuzzy_ontology_filter(self):
        """Fuzzy lookup with ontology filter."""
        results = fuzzy_lookup("cisplatin", ontology="rxnorm", threshold=0.8)
        for r in results:
            assert r["ontology"] == "rxnorm"


class TestBackwardCompatibility:
    """Test backward compatibility of ontology mappings."""

    def test_ontology_mappings_has_three_ontologies(self):
        """ONTOLOGY_MAPPINGS should have snomed_ct, rxnorm, loinc."""
        assert "snomed_ct" in ONTOLOGY_MAPPINGS
        assert "rxnorm" in ONTOLOGY_MAPPINGS
        assert "loinc" in ONTOLOGY_MAPPINGS

    def test_same_structure_as_original(self):
        """Each ontology should be a dict of str -> dict with id/name."""
        for ont_name, ont_dict in ONTOLOGY_MAPPINGS.items():
            assert isinstance(ont_dict, dict), f"{ont_name} not a dict"
            for term, mapping in ont_dict.items():
                assert isinstance(term, str)
                assert isinstance(mapping, dict)
                assert "id" in mapping
                assert "name" in mapping

    def test_original_snomed_entries(self):
        """All original SNOMED-CT entries should be present."""
        snomed = ONTOLOGY_MAPPINGS["snomed_ct"]
        assert "breast cancer" in snomed
        assert snomed["breast cancer"]["id"] == "254838004"
        assert "lung cancer" in snomed
        assert snomed["lung cancer"]["id"] == "254637007"
        assert "chemotherapy" in snomed
        assert "tamoxifen" in snomed

    def test_original_rxnorm_entries(self):
        """All original RxNorm entries should be present."""
        rxnorm = ONTOLOGY_MAPPINGS["rxnorm"]
        assert "tamoxifen" in rxnorm
        assert "carboplatin" in rxnorm
        assert "paclitaxel" in rxnorm

    def test_original_loinc_entries(self):
        """All original LOINC entries should be present."""
        loinc = ONTOLOGY_MAPPINGS["loinc"]
        assert "hemoglobin" in loinc
        assert loinc["hemoglobin"]["id"] == "718-7"
        assert "creatinine" in loinc
        assert "glucose" in loinc
