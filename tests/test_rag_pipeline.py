"""
Tests for rag_pipeline.py
===========================
Tests document chunking, query building, and brief generation
without requiring ChromaDB or OpenAI.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Tests: document chunking
# ============================================================
class TestDocumentChunking:

    def test_chunk_returns_list(self):
        from rag_pipeline import chunk_document
        doc = {"filename": "test.md", "content": "# Section\n\nSome content here.\n\nMore content."}
        chunks = chunk_document(doc)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_required_keys(self):
        from rag_pipeline import chunk_document
        doc = {"filename": "test.md", "content": "# Section\n\nSome content here."}
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "chunk_id" in chunk

    def test_chunk_source_matches_filename(self):
        from rag_pipeline import chunk_document
        doc = {"filename": "fraud_red_flags.md", "content": "# Test\n\nContent."}
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk["source"] == "fraud_red_flags.md"

    def test_large_document_splits_into_multiple_chunks(self):
        from rag_pipeline import chunk_document
        # Create content larger than CHUNK_SIZE (500 chars)
        long_content = "## Section 1\n\n" + ("A" * 300) + "\n\n## Section 2\n\n" + ("B" * 300)
        doc = {"filename": "large.md", "content": long_content}
        chunks = chunk_document(doc)
        assert len(chunks) >= 2

    def test_chunk_text_is_non_empty(self):
        from rag_pipeline import chunk_document
        doc = {"filename": "test.md", "content": "# Title\n\nActual content here.\n\nMore content."}
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0

    def test_chunk_ids_are_unique(self):
        from rag_pipeline import chunk_document
        long_content = "\n\n".join([f"## Section {i}\n\n" + "X" * 200 for i in range(10)])
        doc = {"filename": "test.md", "content": long_content}
        chunks = chunk_document(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"


# ============================================================
# Tests: retrieval query building
# ============================================================
class TestQueryBuilding:

    def test_query_is_string(self):
        from rag_pipeline import build_retrieval_query
        claim_data = {"BasePolicy": "Liability", "Fault": "Policy Holder"}
        reason_codes = ["[MILD] High risk: Deductible (SHAP: +0.12)"]
        query = build_retrieval_query(claim_data, reason_codes)
        assert isinstance(query, str)

    def test_query_contains_claim_fields(self):
        from rag_pipeline import build_retrieval_query
        claim_data = {"BasePolicy": "Liability", "Fault": "Policy Holder",
                      "PoliceReportFiled": "No"}
        query = build_retrieval_query(claim_data, [])
        assert "Liability" in query
        assert "Policy Holder" in query

    def test_query_contains_reason_codes(self):
        from rag_pipeline import build_retrieval_query
        claim_data = {}
        reason_codes = ["High risk: NoPoliceReport", "High risk: ExternalAgent"]
        query = build_retrieval_query(claim_data, reason_codes)
        assert "NoPoliceReport" in query or "Risk factors" in query

    def test_empty_inputs_returns_string(self):
        from rag_pipeline import build_retrieval_query
        query = build_retrieval_query({}, [])
        assert isinstance(query, str)

    def test_query_length_reasonable(self):
        from rag_pipeline import build_retrieval_query
        claim_data = {
            "BasePolicy": "Liability", "Fault": "Policy Holder",
            "PoliceReportFiled": "No", "WitnessPresent": "No",
            "AgentType": "External", "Deductible": 400,
        }
        reason_codes = [f"[MILD] High risk: feature_{i}" for i in range(5)]
        query = build_retrieval_query(claim_data, reason_codes)
        assert len(query) > 10
        assert len(query) < 5000


# ============================================================
# Tests: template brief generation (no LLM needed)
# ============================================================
class TestTemplateBrief:

    @pytest.fixture
    def sample_passages(self):
        return [
            {"text": "Liability coverage fraud includes staged accidents.", "source": "fraud_red_flags.md"},
            {"text": "No police report is a primary red flag.", "source": "staged_accident_patterns.md"},
            {"text": "SIU escalation required for top 5% risk claims.", "source": "triage_procedures.md"},
        ]

    @pytest.fixture
    def sample_claim(self):
        return {
            "BasePolicy": "Liability",
            "Fault": "Policy Holder",
            "PoliceReportFiled": "No",
            "WitnessPresent": "No",
            "AgentType": "External",
            "Deductible": 400,
            "Age": 23,
        }

    @pytest.fixture
    def sample_reason_codes(self):
        return [
            "[MILD] High risk: BasePolicy = 'Liability' (SHAP: +0.07)",
            "[MILD] High risk: Deductible (SHAP: +0.12)",
            "[MILD] Low risk: Age (SHAP: -0.05)",
        ]

    def test_siu_brief_is_string(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert isinstance(brief, str)
        assert len(brief) > 100

    def test_siu_brief_contains_siu_action(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert "SIU" in brief

    def test_manual_brief_contains_manual_action(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.35, triage_bucket="Manual Review"
        )
        assert "Manual" in brief

    def test_approve_brief_contains_approve_action(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.05, triage_bucket="Approve"
        )
        assert "Approval" in brief or "Approve" in brief or "Standard" in brief

    def test_brief_contains_risk_score(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert "0.65" in brief

    def test_brief_cites_passages(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert "Source" in brief

    def test_generate_brief_returns_tuple(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief
        # No OpenAI key set → should use template
        os.environ.pop("OPENAI_API_KEY", None)
        brief, method = generate_brief(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert isinstance(brief, str)
        assert method in ["llm", "template"]

    def test_template_fallback_when_no_api_key(self, sample_claim, sample_reason_codes, sample_passages):
        from rag_pipeline import generate_brief
        os.environ.pop("OPENAI_API_KEY", None)
        brief, method = generate_brief(
            sample_claim, sample_reason_codes, sample_passages,
            risk_score=0.65, triage_bucket="SIU"
        )
        assert method == "template"
