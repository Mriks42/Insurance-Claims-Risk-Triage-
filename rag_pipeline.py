"""
RAG Pipeline for Insurance Fraud Triage Briefs
================================================
Phase 5 (Weeks 9-10) of the FSE 570 Capstone project.

This module:
1. Loads and chunks insurance fraud guideline documents
2. Builds a ChromaDB vector index with sentence-transformer embeddings
3. Retrieves relevant policy passages given a claim's risk factors
4. Generates a cited triage brief using an LLM (OpenAI) or template fallback

Usage:
    python rag_pipeline.py                  # standalone demo
    from rag_pipeline import generate_brief # import for Streamlit app

Requirements:
    pip install chromadb sentence-transformers openai
    (openai is optional — template fallback works without it)

Environment variables:
    OPENAI_API_KEY — set this to enable LLM-powered briefs (optional)
"""

import os
import re
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore")

# Load .env file so OPENAI_API_KEY is available without setting env vars manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import OUTPUTS_DIR

# ============================================================
# Configuration
# ============================================================
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "fraud_guidelines")
RAG_DIR = os.path.join(OUTPUTS_DIR, "rag")
CHROMA_DIR = os.path.join(RAG_DIR, "chroma_db")
os.makedirs(RAG_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks
TOP_K_PASSAGES = 5      # number of passages to retrieve
COLLECTION_NAME = "fraud_guidelines"


# ============================================================
# 1) Document loading and chunking
# ============================================================
def load_documents(docs_dir=DOCS_DIR):
    """Load all markdown documents from the guidelines directory."""
    docs = []
    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        docs.append({"filename": filename, "content": content})
        print(f"  Loaded: {filename} ({len(content)} chars)")
    return docs


def chunk_document(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split a document into overlapping text chunks.
    Tries to split on paragraph/section boundaries first.
    """
    content = doc["content"]
    filename = doc["filename"]
    chunks = []

    # Split by sections (##) first for more meaningful chunks
    sections = content.split("\n## ")
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section  # restore heading

        # If section is small enough, keep it as one chunk
        if len(section) <= chunk_size:
            chunks.append({
                "text": section.strip(),
                "source": filename,
                "chunk_id": len(chunks),
            })
        else:
            # Split by paragraphs within the section
            paragraphs = section.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": filename,
                            "chunk_id": len(chunks),
                        })
                    current_chunk = para + "\n\n"
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": filename,
                    "chunk_id": len(chunks),
                })

    return chunks


def load_and_chunk_all(docs_dir=DOCS_DIR):
    """Load all docs and chunk them. Returns list of chunk dicts."""
    docs = load_documents(docs_dir)
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


# ============================================================
# 2) Build vector index with ChromaDB
# ============================================================
def build_vector_index(chunks, persist_dir=CHROMA_DIR):
    """
    Build a ChromaDB collection from document chunks.
    Uses PersistentClient — if the collection already exists it is reused,
    so the Streamlit app loads instantly without re-embedding on every restart.
    Pass force_rebuild=True (or delete the chroma_db folder) to force a rebuild.
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("  Installing chromadb...")
        os.system(f"{sys.executable} -m pip install chromadb --quiet")
        import chromadb
        from chromadb.utils import embedding_functions

    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
    except Exception:
        print("  Installing sentence-transformers...")
        os.system(f"{sys.executable} -m pip install sentence-transformers --quiet")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Reuse existing collection if it already has the right number of chunks
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        if collection.count() == len(chunks):
            print(f"  Reusing existing index: {collection.count()} chunks (no rebuild needed)")
            return client, collection
        else:
            print(f"  Index size mismatch ({collection.count()} vs {len(chunks)}), rebuilding...")
            client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection doesn't exist yet — create it below

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Insurance fraud investigation guidelines"},
    )

    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"  Vector index built: {collection.count()} chunks indexed")
    print(f"  Persisted to: {persist_dir}")
    return client, collection


# ============================================================
# 3) Retrieve relevant passages
# ============================================================
def retrieve_passages(collection, query, top_k=TOP_K_PASSAGES):
    """Retrieve the most relevant passages for a given query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    passages = []
    for i in range(len(results["documents"][0])):
        passages.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return passages


def build_retrieval_query(claim_data, reason_codes):
    """
    Build a natural language query from claim data and SHAP reason codes
    to retrieve relevant guideline passages.
    """
    parts = []

    # Include key claim attributes
    if claim_data:
        parts.append(f"Auto insurance claim analysis.")
        if "BasePolicy" in claim_data:
            parts.append(f"Base policy: {claim_data['BasePolicy']}.")
        if "Fault" in claim_data:
            parts.append(f"Fault: {claim_data['Fault']}.")
        if "PoliceReportFiled" in claim_data:
            parts.append(f"Police report filed: {claim_data['PoliceReportFiled']}.")
        if "WitnessPresent" in claim_data:
            parts.append(f"Witness present: {claim_data['WitnessPresent']}.")
        if "VehicleCategory" in claim_data:
            parts.append(f"Vehicle category: {claim_data['VehicleCategory']}.")
        if "AgentType" in claim_data:
            parts.append(f"Agent type: {claim_data['AgentType']}.")
        if "AddressChange_Claim" in claim_data:
            parts.append(f"Address change: {claim_data['AddressChange_Claim']}.")
        if "Deductible" in claim_data:
            parts.append(f"Deductible: ${claim_data['Deductible']}.")

    # Include reason codes
    if reason_codes:
        parts.append("Risk factors identified:")
        for rc in reason_codes[:5]:
            parts.append(f"- {rc}")

    return " ".join(parts)


# ============================================================
# 4) Generate triage brief
# ============================================================
def generate_brief_llm(claim_data, reason_codes, passages, risk_score, triage_bucket):
    """Generate a triage brief using OpenAI GPT."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    # Build context from retrieved passages
    context_parts = []
    for i, p in enumerate(passages, 1):
        context_parts.append(f"[Source {i}: {p['source']}]\n{p['text']}")
    context = "\n\n".join(context_parts)

    # Build reason codes text.
    # SHAP magnitudes and [INTENSITY] tags are stripped before they reach the
    # model. The brief is written for a claims investigator, and the model
    # echoes whatever it is handed — previously producing prose like "a strong
    # indicator of potential fraud risk (SHAP: +1.7181)". The numbers belong on
    # the dashboard next to the waterfall chart; the prompt gets the driver in
    # plain English.
    cleaned_reasons = [
        re.sub(r"\s*\(SHAP:[^)]*\)", "",
               re.sub(r"^\[[A-Z]+\]\s*", "", rc)).strip()
        for rc in reason_codes
    ]
    reasons_text = "\n".join(f"- {rc}" for rc in cleaned_reasons)

    # Build claim summary
    claim_summary = ", ".join(f"{k}: {v}" for k, v in claim_data.items()) if claim_data else "No claim data available"

    prompt = f"""You are an insurance fraud analyst AI assistant. Generate a concise triage brief for the following claim.

## Claim Data
{claim_summary}

## Risk Score: {risk_score:.4f}
## Triage Bucket: {triage_bucket}

## Machine-Generated Risk Factors (SHAP Analysis)
{reasons_text}

## Relevant Policy Guidelines
{context}

## Instructions
Write a 2-3 paragraph triage brief that:
1. Summarizes what the listed factors say about this claim, in plain language.
   Each factor is labelled "High risk" or "Low risk" — respect that labelling.
   A factor marked Low risk is a reason the claim looks ROUTINE, not a concern.
2. References specific policy guidelines that apply, citing them as [Source N].
   Only cite source numbers listed in the Relevant Policy Guidelines section above.
   Never cite a source number that was not provided.
3. Recommends specific next steps, consistent with the triage bucket above.
4. Is suitable for a non-technical insurance claims reviewer.

The tone must match the triage bucket:
- SIU: this claim is being escalated for investigation. Lead with the concerns.
- Manual Review: a human should verify specific details before approval. Lead
  with what needs checking.
- Approve: this claim looks routine and is heading for automated processing.
  Lead with why it looks legitimate. Do not write it up as a fraud narrative,
  and do not recommend an investigation the bucket does not call for. If there
  is a residual concern worth noting, note it briefly and say it is not
  sufficient to hold the claim.

Keep the brief concise, professional, and actionable. Write for a claims investigator with no machine-learning background: do not mention SHAP, model scores, or numeric feature weights."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert insurance fraud analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  OpenAI API error: {e}")
        return None


def generate_brief_template(claim_data, reason_codes, passages, risk_score, triage_bucket):
    """Generate a triage brief using a template (no LLM required)."""

    # Determine risk level description
    if risk_score >= 0.5:
        risk_desc = "HIGH"
    elif risk_score >= 0.3:
        risk_desc = "ELEVATED"
    elif risk_score >= 0.1:
        risk_desc = "MODERATE"
    else:
        risk_desc = "LOW"

    # Build risk factor summary
    high_risk_factors = [rc for rc in reason_codes if "High risk" in rc]
    low_risk_factors = [rc for rc in reason_codes if "Low risk" in rc]

    # Build the brief
    lines = []
    lines.append(f"## Triage Brief")
    lines.append(f"**Risk Score:** {risk_score:.4f} ({risk_desc})")
    lines.append(f"**Triage Bucket:** {triage_bucket}")
    lines.append("")

    # Paragraph 1: Risk summary
    lines.append("### Risk Assessment")
    if high_risk_factors:
        lines.append(f"This claim has been flagged with {len(high_risk_factors)} risk indicator(s):")
        for rf in high_risk_factors:
            lines.append(f"  - {rf}")
    else:
        lines.append("No significant risk indicators were identified for this claim.")

    if low_risk_factors:
        lines.append(f"\nThe following factor(s) reduce the overall risk assessment:")
        for rf in low_risk_factors:
            lines.append(f"  - {rf}")
    lines.append("")

    # Paragraph 2: Key claim details
    lines.append("### Claim Details")
    if claim_data:
        key_fields = ["BasePolicy", "Fault", "VehicleCategory", "PoliceReportFiled",
                      "WitnessPresent", "AgentType", "AddressChange_Claim", "Deductible", "Age"]
        for field in key_fields:
            if field in claim_data:
                lines.append(f"  - **{field}:** {claim_data[field]}")
    lines.append("")

    # Paragraph 3: Relevant policy guidelines (cited)
    lines.append("### Applicable Guidelines")
    if passages:
        for i, p in enumerate(passages[:3], 1):
            # Extract first sentence or first 150 chars
            snippet = p["text"][:200].replace("\n", " ").strip()
            if len(p["text"]) > 200:
                snippet += "..."
            lines.append(f"  - [Source {i}: {p['source']}] {snippet}")
    lines.append("")

    # Paragraph 4: Recommended action
    lines.append("### Recommended Action")
    if triage_bucket == "SIU":
        lines.append("This claim is recommended for **SIU Escalation**. The Special Investigations Unit should:")
        lines.append("  1. Review the complete claim file and all supporting documentation")
        lines.append("  2. Conduct recorded interviews with the policyholder")
        lines.append("  3. Perform vehicle inspection or independent appraisal")
        lines.append("  4. Cross-reference against NICB fraud database")
        lines.append("  5. Complete investigation within 30 business days")
    elif triage_bucket == "Manual Review":
        lines.append("This claim requires **Manual Review**. The reviewer should:")
        lines.append("  1. Verify police report details against the claim")
        lines.append("  2. Confirm policyholder identity and coverage status")
        lines.append("  3. Validate repair estimates against industry benchmarks")
        lines.append("  4. Review and assess machine-generated reason codes")
        lines.append("  5. Complete review within 10 business days")
    else:
        lines.append("This claim is recommended for **Standard Approval**. Process through automated workflow.")
        lines.append("Note: This claim will be part of the quarterly random sampling audit.")

    return "\n".join(lines)


CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)", re.IGNORECASE)


def verify_citations(brief, passages):
    """
    Check that every [Source N] in a generated brief refers to a passage that
    was actually retrieved.

    The template builder cannot get this wrong — it numbers the passages it was
    handed. An LLM can: it may cite [Source 4] when four passages were never
    supplied, or attribute a rule to the wrong document. The brief is shown to
    an investigator as grounds for escalating a claim, so an invented citation
    is the failure that matters most on that page — it looks exactly like a real
    one.

    This does not verify that the cited passage SUPPORTS the sentence it is
    attached to; that would need a separate entailment check. It verifies the
    citation points at something real, which is the cheap and checkable half.

    Returns a dict:
        cited          - source numbers referenced in the brief
        valid          - those within range of the retrieved passages
        invalid        - those that are not (hallucinated)
        n_passages     - how many were actually supplied
        ok             - True when nothing was invented
    """
    cited = sorted({int(n) for n in CITATION_PATTERN.findall(brief or "")})
    n = len(passages or [])
    valid = [c for c in cited if 1 <= c <= n]
    invalid = [c for c in cited if c < 1 or c > n]
    return {
        "cited":      cited,
        "valid":      valid,
        "invalid":    invalid,
        "n_passages": n,
        "ok":         not invalid,
    }


def annotate_invalid_citations(brief, check):
    """Mark hallucinated citations inline rather than deleting them.

    Silently stripping them would hide that the model fabricated a reference —
    the reader would see a fluent, unsourced assertion and have no reason to
    doubt it. Flagging keeps the failure visible to the person who has to act.
    """
    if check["ok"]:
        return brief
    out = brief
    for bad in check["invalid"]:
        out = re.sub(rf"\[Source\s+{bad}\b", f"[UNVERIFIED Source {bad}", out,
                     flags=re.IGNORECASE)
    return out


def generate_brief(claim_data, reason_codes, passages, risk_score, triage_bucket):
    """
    Generate a triage brief. Tries LLM first, falls back to template.

    Returns (brief_text, method_used, citation_check).
    """
    llm_brief = generate_brief_llm(claim_data, reason_codes, passages, risk_score, triage_bucket)
    if llm_brief:
        check = verify_citations(llm_brief, passages)
        return annotate_invalid_citations(llm_brief, check), "llm", check

    # Fall back to template — its citations are generated from the passage list
    # itself, so they are correct by construction.
    template_brief = generate_brief_template(
        claim_data, reason_codes, passages, risk_score, triage_bucket
    )
    return template_brief, "template", verify_citations(template_brief, passages)


# ============================================================
# 5) Full RAG pipeline: end-to-end
# ============================================================
class RAGPipeline:
    """
    Complete RAG pipeline for generating fraud triage briefs.
    Initialize once, then call generate() for each claim.
    """

    def __init__(self, docs_dir=DOCS_DIR):
        print("Initializing RAG pipeline...")
        print("\nLoading and chunking documents...")
        self.chunks = load_and_chunk_all(docs_dir)

        print("\nBuilding vector index...")
        self.client, self.collection = build_vector_index(self.chunks)
        print("RAG pipeline ready.\n")

    def generate(self, claim_data, reason_codes, risk_score, triage_bucket):
        """
        Generate a triage brief for a single claim.

        Args:
            claim_data: dict of claim attributes (e.g., {"BasePolicy": "Liability", ...})
            reason_codes: list of human-readable reason code strings
            risk_score: float, the model's predicted fraud probability
            triage_bucket: str, one of "SIU", "Manual Review", "Approve"

        Returns:
            dict with keys: brief, method, passages, query
        """
        # Build retrieval query
        query = build_retrieval_query(claim_data, reason_codes)

        # Retrieve relevant passages
        passages = retrieve_passages(self.collection, query)

        # Generate brief
        brief, method, citation_check = generate_brief(
            claim_data, reason_codes, passages, risk_score, triage_bucket
        )

        return {
            "brief": brief,
            "method": method,
            "passages": passages,
            "query": query,
            "citation_check": citation_check,
        }


# ============================================================
# Main: demo the RAG pipeline
# ============================================================
def run_demo():
    """Run a demo of the RAG pipeline with sample claims."""
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    rag = RAGPipeline()

    # Sample claims to demo
    sample_claims = [
        {
            "claim_data": {
                "BasePolicy": "Liability",
                "Fault": "Policy Holder",
                "VehicleCategory": "Sport",
                "VehiclePrice": "more than 69000",
                "PoliceReportFiled": "No",
                "WitnessPresent": "No",
                "AgentType": "External",
                "AddressChange_Claim": "2 to 3 years",
                "Deductible": 400,
                "Age": 23,
                "PastNumberOfClaims": "1",
            },
            "reason_codes": [
                "[MILD] High risk: AddressChange_Claim = '2 to 3 years' (SHAP: +0.1221)",
                "[MILD] High risk: Deductible (SHAP: +0.1181)",
                "[MILD] High risk: BasePolicy = 'Liability' (SHAP: +0.0670)",
                "[MILD] Low risk: Fault = 'Policy Holder' (SHAP: -0.0468)",
                "[MILD] High risk: Age (SHAP: +0.0146)",
            ],
            "risk_score": 0.5703,
            "triage_bucket": "SIU",
        },
        {
            "claim_data": {
                "BasePolicy": "Collision",
                "Fault": "Third Party",
                "VehicleCategory": "Sedan",
                "VehiclePrice": "20000 to 29000",
                "PoliceReportFiled": "Yes",
                "WitnessPresent": "Yes",
                "AgentType": "Internal",
                "AddressChange_Claim": "no change",
                "Deductible": 300,
                "Age": 42,
                "PastNumberOfClaims": "none",
            },
            "reason_codes": [
                "[MILD] Low risk: BasePolicy = 'Collision' (SHAP: -0.0320)",
                "[MILD] Low risk: Fault = 'Third Party' (SHAP: -0.0280)",
                "[MILD] Low risk: Age (SHAP: -0.0150)",
            ],
            "risk_score": 0.05,
            "triage_bucket": "Approve",
        },
        {
            "claim_data": {
                "BasePolicy": "All Perils",
                "Fault": "Policy Holder",
                "VehicleCategory": "Utility",
                "VehiclePrice": "40000 to 59000",
                "PoliceReportFiled": "No",
                "WitnessPresent": "Yes",
                "AgentType": "External",
                "AddressChange_Claim": "under 6 months",
                "Deductible": 700,
                "Age": 31,
                "PastNumberOfClaims": "2 to 4",
            },
            "reason_codes": [
                "[MILD] High risk: Deductible (SHAP: +0.0890)",
                "[MILD] High risk: PoliceReportFiled = 'No' (SHAP: +0.0450)",
                "[MILD] High risk: AddressChange_Claim = 'under 6 months' (SHAP: +0.0380)",
                "[MILD] High risk: PastNumberOfClaims (SHAP: +0.0250)",
                "[MILD] Low risk: WitnessPresent = 'Yes' (SHAP: -0.0180)",
            ],
            "risk_score": 0.35,
            "triage_bucket": "Manual Review",
        },
    ]

    # Generate briefs
    all_briefs = []
    for i, sample in enumerate(sample_claims, 1):
        print(f"\n{'=' * 60}")
        print(f"CLAIM {i}: {sample['triage_bucket']} (Risk Score: {sample['risk_score']:.4f})")
        print("=" * 60)

        result = rag.generate(
            claim_data=sample["claim_data"],
            reason_codes=sample["reason_codes"],
            risk_score=sample["risk_score"],
            triage_bucket=sample["triage_bucket"],
        )

        print(f"\nMethod: {result['method']}")
        print(f"Retrieved {len(result['passages'])} passages")
        print(f"\n--- TRIAGE BRIEF ---\n")
        print(result["brief"])

        all_briefs.append({
            "claim_id": i,
            "triage_bucket": sample["triage_bucket"],
            "risk_score": sample["risk_score"],
            "method": result["method"],
            "brief": result["brief"],
            "passages_used": len(result["passages"]),
            "passage_sources": [p["source"] for p in result["passages"]],
            "citation_check": result["citation_check"],
        })

    # Save all briefs
    import json
    with open(os.path.join(RAG_DIR, "demo_briefs.json"), "w", encoding="utf-8") as f:
        json.dump(all_briefs, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved all briefs to: {os.path.join(RAG_DIR, 'demo_briefs.json')}")

    # Save briefs as readable text
    with open(os.path.join(RAG_DIR, "demo_briefs.txt"), "w", encoding="utf-8") as f:
        for b in all_briefs:
            f.write(f"{'=' * 60}\n")
            f.write(f"CLAIM {b['claim_id']}: {b['triage_bucket']} (Risk: {b['risk_score']:.4f})\n")
            f.write(f"Method: {b['method']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(b["brief"])
            f.write(f"\n\n")
    print(f"Saved readable briefs to: {os.path.join(RAG_DIR, 'demo_briefs.txt')}")

    print("\n" + "=" * 60)
    print("[OK] RAG Pipeline demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
