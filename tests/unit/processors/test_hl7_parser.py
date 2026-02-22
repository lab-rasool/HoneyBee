"""
Unit tests for HL7 v2 parser.

All tests use unittest.mock so they work without hl7apy installed.
"""

from unittest.mock import patch

import pytest


class TestHL7ImportGuard:
    """Test HL7 import guard."""

    def test_import_error_without_hl7apy(self):
        """HL7Parser should raise ImportError without hl7apy."""
        import sys
        hl7_modules = {k: v for k, v in sys.modules.items() if k.startswith("hl7")}
        for k in hl7_modules:
            sys.modules[k] = None
        try:
            from honeybee.processors.clinical.interop import hl7_parser
            if not hl7_parser._HL7_AVAILABLE:
                with pytest.raises(ImportError, match="hl7apy"):
                    hl7_parser.HL7Parser()
        finally:
            for k, v in hl7_modules.items():
                sys.modules[k] = v
            for k in list(sys.modules):
                if k.startswith("hl7") and sys.modules[k] is None:
                    del sys.modules[k]


class TestHL7SegmentParsers:
    """Test individual segment parsing functions (no hl7apy needed)."""

    def test_parse_pid(self):
        """PID segment parser should extract patient fields."""
        from honeybee.processors.clinical.interop.hl7_parser import _parse_pid
        segment = "PID|1||P001||Doe^John||19670101|M"
        result = _parse_pid(segment)
        assert result["patient_id"] == "P001"
        assert result["patient_name"] == "Doe^John"
        assert result["date_of_birth"] == "19670101"
        assert result["sex"] == "M"

    def test_parse_obx(self):
        """OBX segment parser should extract observation fields."""
        from honeybee.processors.clinical.interop.hl7_parser import _parse_obx
        segment = "OBX|1|NM|Hemoglobin||12.5|g/dL|12-16|N|||F"
        result = _parse_obx(segment)
        assert result["observation_id"] == "Hemoglobin"
        assert result["value"] == "12.5"
        assert result["units"] == "g/dL"
        assert result["reference_range"] == "12-16"

    def test_parse_dg1(self):
        """DG1 segment parser should extract diagnosis fields."""
        from honeybee.processors.clinical.interop.hl7_parser import _parse_dg1
        segment = "DG1|1||C50.9|Breast Cancer||A"
        result = _parse_dg1(segment)
        assert result["diagnosis_code"] == "C50.9"
        assert result["description"] == "Breast Cancer"

    def test_parse_rxa(self):
        """RXA segment parser should extract medication fields."""
        from honeybee.processors.clinical.interop.hl7_parser import _parse_rxa
        segment = "RXA|0|1|202401150800||Tamoxifen|20|mg"
        result = _parse_rxa(segment)
        assert result["administered_code"] == "Tamoxifen"
        assert result["administered_amount"] == "20"
        assert result["administered_units"] == "mg"

    def test_safe_field_out_of_range(self):
        """_safe_field should return default for out-of-range index."""
        from honeybee.processors.clinical.interop.hl7_parser import _safe_field
        assert _safe_field("A|B|C", 10) == ""
        assert _safe_field("A|B|C", 10, "default") == "default"


class TestHL7Parser:
    """Test HL7Parser class (requires hl7apy or mock)."""

    @patch("honeybee.processors.clinical.interop.hl7_parser._HL7_AVAILABLE", True)
    def test_parse_adt_message(self):
        """Parse ADT message with PID segment."""
        from honeybee.processors.clinical.interop.hl7_parser import HL7Parser
        parser = HL7Parser()

        message = (
            "MSH|^~\\&|HIS|Hospital|Lab|Lab|202401150800||ADT^A01|12345|P|2.5\r"
            "PID|1||P001||Doe^John||19670101|M\r"
            "DG1|1||C50.9|Breast Cancer||A"
        )
        result = parser.parse(message)

        assert result["message_type"] == "ADT^A01"
        assert result["patient"]["patient_id"] == "P001"
        assert len(result["diagnoses"]) == 1
        assert result["diagnoses"][0]["description"] == "Breast Cancer"
        assert "text" in result
        assert len(result["text"]) > 0

    @patch("honeybee.processors.clinical.interop.hl7_parser._HL7_AVAILABLE", True)
    def test_parse_oru_message(self):
        """Parse ORU message with OBX segments."""
        from honeybee.processors.clinical.interop.hl7_parser import HL7Parser
        parser = HL7Parser()

        message = (
            "MSH|^~\\&|Lab|Hospital|HIS|Hospital|202401150900||ORU^R01|12346|P|2.5\r"
            "PID|1||P002||Smith^Jane||19800515|F\r"
            "OBX|1|NM|Hemoglobin||12.5|g/dL|12-16|N|||F\r"
            "OBX|2|NM|WBC||5.0|K/uL|4-11|N|||F"
        )
        result = parser.parse(message)

        assert result["message_type"] == "ORU^R01"
        assert len(result["observations"]) == 2
        assert result["observations"][0]["value"] == "12.5"

    @patch("honeybee.processors.clinical.interop.hl7_parser._HL7_AVAILABLE", True)
    def test_parse_medication_message(self):
        """Parse message with RXA segment."""
        from honeybee.processors.clinical.interop.hl7_parser import HL7Parser
        parser = HL7Parser()

        message = (
            "MSH|^~\\&|Pharm|Hospital|HIS|Hospital|202401151000||ORM^O01|12347|P|2.5\r"
            "PID|1||P003||Brown^Bob||19551220|M\r"
            "RXA|0|1|202401150800||Tamoxifen|20|mg"
        )
        result = parser.parse(message)

        assert len(result["medications"]) == 1
        assert result["medications"][0]["administered_code"] == "Tamoxifen"

    @patch("honeybee.processors.clinical.interop.hl7_parser._HL7_AVAILABLE", True)
    def test_to_text(self):
        """to_text should produce clinical text from parsed data."""
        from honeybee.processors.clinical.interop.hl7_parser import HL7Parser
        parser = HL7Parser()

        parsed = {
            "patient": {
                "patient_name": "Doe^John",
                "date_of_birth": "19670101",
                "sex": "M",
            },
            "diagnoses": [
                {"description": "Breast Cancer", "diagnosis_code": "C50.9"},
            ],
            "observations": [
                {"observation_id": "Hemoglobin", "value": "12.5", "units": "g/dL"},
            ],
            "medications": [
                {
                    "administered_code": "Tamoxifen",
                    "administered_amount": "20",
                    "administered_units": "mg",
                },
            ],
        }
        text = parser.to_text(parsed)

        assert "Doe^John" in text
        assert "Breast Cancer" in text
        assert "Hemoglobin" in text
        assert "Tamoxifen" in text

    @patch("honeybee.processors.clinical.interop.hl7_parser._HL7_AVAILABLE", True)
    def test_empty_message(self):
        """Parsing empty message should not crash."""
        from honeybee.processors.clinical.interop.hl7_parser import HL7Parser
        parser = HL7Parser()
        result = parser.parse("")
        assert result["text"] == ""
        assert result["observations"] == []
