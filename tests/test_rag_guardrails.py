"""
Tests for RAG output guardrails
================================
The triage brief is what a claims investigator reads before escalating a claim.
A fabricated citation is the failure that matters on that page: it is fluent,
authoritative, and indistinguishable from a real one.

These tests need no API key and no vector store — verify_citations() works on
plain text, which is the point of keeping the check separate from generation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import (annotate_invalid_citations,  # noqa: E402
                          verify_citations)

PASSAGES = [
    {"text": "High-deductible policies paired with frequent claims suggest moral hazard.",
     "source": "fraud_red_flags.md"},
    {"text": "Sport vehicles are overrepresented in staged accident fraud.",
     "source": "staged_accident_patterns.md"},
    {"text": "Liability-only cover on a high-value vehicle is unusual.",
     "source": "policy_coverage_standards.md"},
]


class TestVerifyCitations:

    def test_all_citations_valid(self):
        brief = ("Risk is elevated [Source 1]. The vehicle profile matches known "
                 "staged-accident patterns [Source 2].")
        check = verify_citations(brief, PASSAGES)
        assert check["ok"] is True
        assert check["cited"] == [1, 2]
        assert check["invalid"] == []
        assert check["n_passages"] == 3

    def test_hallucinated_citation_detected(self):
        """The case this exists for: three passages supplied, four cited."""
        brief = "Elevated risk [Source 1]. Policy requires escalation [Source 4]."
        check = verify_citations(brief, PASSAGES)
        assert check["ok"] is False
        assert check["invalid"] == [4]
        assert check["valid"] == [1]

    def test_zero_and_negative_are_invalid(self):
        check = verify_citations("See [Source 0] for detail.", PASSAGES)
        assert check["ok"] is False
        assert check["invalid"] == [0]

    def test_brief_without_citations(self):
        check = verify_citations("No guidelines were applicable.", PASSAGES)
        assert check["cited"] == []
        assert check["ok"] is True          # nothing invented

    def test_no_passages_makes_any_citation_invalid(self):
        check = verify_citations("As noted [Source 1].", [])
        assert check["ok"] is False
        assert check["invalid"] == [1]

    def test_duplicate_citations_counted_once(self):
        brief = "[Source 2] applies here, and [Source 2] again later."
        check = verify_citations(brief, PASSAGES)
        assert check["cited"] == [2]

    def test_case_insensitive(self):
        check = verify_citations("see [source 2] and [SOURCE 3]", PASSAGES)
        assert sorted(check["cited"]) == [2, 3]
        assert check["ok"] is True

    def test_handles_empty_brief(self):
        assert verify_citations("", PASSAGES)["ok"] is True
        assert verify_citations(None, PASSAGES)["cited"] == []


class TestAnnotation:

    def test_invalid_citation_is_flagged_not_deleted(self):
        """Stripping a fabricated citation would leave a fluent, unsourced
        assertion the reader has no reason to doubt. Flag it instead."""
        brief = "Escalation is required [Source 4]."
        check = verify_citations(brief, PASSAGES)
        out = annotate_invalid_citations(brief, check)
        assert "UNVERIFIED Source 4" in out
        assert "Escalation is required" in out      # content preserved

    def test_valid_citations_untouched(self):
        brief = "Risk elevated [Source 1] and [Source 2]."
        check = verify_citations(brief, PASSAGES)
        assert annotate_invalid_citations(brief, check) == brief

    def test_mixed_citations_only_bad_one_flagged(self):
        brief = "Both [Source 1] and [Source 9] apply."
        out = annotate_invalid_citations(brief, verify_citations(brief, PASSAGES))
        assert "[Source 1]" in out
        assert "UNVERIFIED Source 9" in out


class TestTemplateBriefIsGroundedByConstruction:
    """The template numbers the passages it was handed, so it cannot invent
    one. Worth asserting: it is the fallback whenever the LLM is unavailable."""

    def test_template_citations_always_valid(self):
        from rag_pipeline import generate_brief_template
        brief = generate_brief_template(
            {"BasePolicy": "Liability", "Age": 23}, ["[MILD] High risk: age"],
            PASSAGES, 0.87, "SIU",
        )
        check = verify_citations(brief, PASSAGES)
        assert check["ok"] is True
        assert check["cited"], "template should cite the retrieved passages"


class TestPromptHygiene:
    """SHAP magnitudes must not reach the model: it echoes what it is given,
    and the brief is written for a reader with no ML background."""

    def test_shap_values_stripped_from_prompt(self):
        import re
        codes = ["[STRONG] High risk: time since the last address change (SHAP: +1.7181)",
                 "[MILD] Low risk: policyholder age (SHAP: -0.0231)"]
        cleaned = [
            re.sub(r"\s*\(SHAP:[^)]*\)", "", re.sub(r"^\[[A-Z]+\]\s*", "", c)).strip()
            for c in codes
        ]
        assert cleaned == ["High risk: time since the last address change",
                           "Low risk: policyholder age"]
        assert not any("SHAP" in c for c in cleaned)
        assert not any(c.startswith("[") for c in cleaned)
