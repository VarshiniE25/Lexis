"""
joint_prompts.py — Joint multi-clause extraction prompts.

Design principles:
  1. ONE prompt extracts ALL 4 clauses simultaneously.
  2. ONE prompt extracts ALL 4 structured fields simultaneously.
  3. ONE joint validation prompt validates ALL extracted clauses at once.
  4. Context is shared — LLM sees the same merged chunks for all clauses.
  5. Each clause has its own search hints embedded in the prompt to guide focus.
  6. Strict null-return rules prevent hallucination.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from .config import CONTRACT_TYPES


# ─── Joint Clause Extraction ─────────────────────────────────────────────────

def joint_clauses_prompt(
    chunk_texts: List[str],
    page_numbers: List[int],
) -> str:
    """
    Single prompt that extracts ALL 4 clauses from the contract in one LLM call.

    Context is the merged global+filtered chunk window.
    Returns a JSON object with keys: governing_law, audit_rights,
    non_compete, non_solicitation.
    """
    context = _format_chunks_with_pages(chunk_texts, page_numbers)

    return f"""You are a senior legal AI analyst. Extract FOUR specific clauses from the contract excerpts below.

════════════════════════════════════════
CONTRACT EXCERPTS (with page numbers)
════════════════════════════════════════
{context}
════════════════════════════════════════

EXTRACT THE FOLLOWING 4 CLAUSES:

━━━ 1. GOVERNING LAW ━━━
Find the clause specifying which jurisdiction's laws govern this contract.
Keywords: "governed by", "construed in accordance with", "laws of", "subject to the laws of"

━━━ 2. AUDIT RIGHTS ━━━
Find the clause granting one party the right to audit the other's records/books.
Keywords: "audit", "right to audit", "inspect records", "books and records", "examine"

━━━ 3. NON-COMPETE ━━━
Find the clause restricting a party from engaging in competing business activities.
Keywords: "non-compete", "not compete", "competing business", "competitive activities", "shall not engage"

━━━ 4. NON-SOLICITATION OF EMPLOYEES ━━━
Find the clause preventing a party from poaching the other party's employees.
Keywords: "non-solicitation", "solicit employees", "recruit", "hire employees", "poach"

════════════════════════════════════════
STRICT EXTRACTION RULES:
════════════════════════════════════════
1. Use ONLY the provided context. DO NOT hallucinate or invent text.
2. If a clause is NOT found → set value, exact_text, page all to null, confidence to 0.0.
3. exact_text MUST be verbatim text copied directly from the excerpts above.
4. page MUST be the integer from the [PAGE X] marker nearest to the found text.
5. confidence: 0.9+ = explicitly stated | 0.7 = clearly implied | 0.5 = uncertain | 0.0 = not found.
6. value should be a concise, human-readable summary of what the clause says.
7. Return ONLY the JSON below — no preamble, no explanation, no markdown fences.

RETURN THIS EXACT JSON STRUCTURE:
{{
  "governing_law": {{
    "value": "<concise summary or null>",
    "page": <integer or null>,
    "exact_text": "<verbatim text or null>",
    "confidence": <0.0-1.0>
  }},
  "audit_rights": {{
    "value": "<concise summary or null>",
    "page": <integer or null>,
    "exact_text": "<verbatim text or null>",
    "confidence": <0.0-1.0>
  }},
  "non_compete": {{
    "value": "<concise summary or null>",
    "page": <integer or null>,
    "exact_text": "<verbatim text or null>",
    "confidence": <0.0-1.0>
  }},
  "non_solicitation": {{
    "value": "<concise summary or null>",
    "page": <integer or null>,
    "exact_text": "<verbatim text or null>",
    "confidence": <0.0-1.0>
  }}
}}"""


# ─── Joint Structured Fields Extraction ──────────────────────────────────────

def joint_fields_prompt(
    chunk_texts: List[str],
    page_numbers: List[int],
) -> str:
    """
    Single prompt that extracts ALL 4 structured fields in one LLM call.
    """
    context = _format_chunks_with_pages(chunk_texts, page_numbers)

    return f"""You are a senior legal AI analyst. Extract FOUR structured fields from the contract excerpts below.

════════════════════════════════════════
CONTRACT EXCERPTS (with page numbers)
════════════════════════════════════════
{context}
════════════════════════════════════════

EXTRACT THE FOLLOWING 4 FIELDS:

━━━ 1. JURISDICTION ━━━
The legal jurisdiction (state/country/court) where disputes are resolved.
Keywords: "exclusive jurisdiction", "courts of", "venue", "submit to jurisdiction"
Example output: "Courts of the State of Delaware"

━━━ 2. PAYMENT TERMS ━━━
How and when payments must be made.
Keywords: "Net 30", "due within", "invoiced", "payment schedule", "monthly", "upon receipt"
Example output: "Net 30 days from invoice date"

━━━ 3. NOTICE PERIOD ━━━
How much advance notice is required (typically for termination or changes).
Keywords: "days prior written notice", "advance notice", "written notice to terminate"
Example output: "30 days written notice"

━━━ 4. LIABILITY CAP ━━━
The maximum financial liability either party can face under this contract.
Keywords: "aggregate liability", "shall not exceed", "liability limited to", "maximum liability"
Example output: "$500,000" or "total fees paid in the preceding 12 months"

════════════════════════════════════════
STRICT EXTRACTION RULES:
════════════════════════════════════════
1. Use ONLY the provided context. DO NOT hallucinate.
2. If a field is NOT found in the text → return null.
3. Values should be concise but complete. Do not truncate important details.
4. Return ONLY the JSON below — no preamble, no explanation, no markdown.

RETURN THIS EXACT JSON STRUCTURE:
{{
  "jurisdiction": "<string or null>",
  "payment_terms": "<string or null>",
  "notice_period": "<string or null>",
  "liability_cap": "<string or null>"
}}"""


# ─── Contract Type Classification ─────────────────────────────────────────────

def joint_contract_type_prompt(chunk_texts: List[str]) -> str:
    """
    Classify contract type. Kept separate from joint clause extraction
    because contract type uses different/broader chunks.
    """
    context = "\n\n---\n\n".join(
        f"EXCERPT {i+1}:\n{text}" for i, text in enumerate(chunk_texts)
    )
    types_list = "\n".join(f"  - {t}" for t in CONTRACT_TYPES)

    return f"""You are a senior contract analyst. Classify the contract type from the excerpts below.

CONTRACT EXCERPTS:
{context}

CLASSIFY INTO EXACTLY ONE OF:
{types_list}

STRICT RULES:
- value MUST be an exact string from the list above.
- confidence: 0.9+ = clearly identifiable | 0.6 = probable | 0.4 = ambiguous.
- Do NOT hallucinate. Base classification on provided text only.
- Return ONLY valid JSON — no preamble, no markdown.

RETURN THIS EXACT JSON:
{{
  "value": "<exact contract type from list>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explanation>"
}}"""


# ─── Joint Validation Prompt ──────────────────────────────────────────────────

def joint_validation_prompt(
    extracted_clauses: Dict[str, Dict],
    original_context: str,
) -> str:
    """
    Single validation prompt that validates ALL extracted clauses at once.
    Replaces 4 separate validation calls with 1.

    Args:
        extracted_clauses: Dict of clause_name → {value, exact_text, page, confidence}
        original_context: The full merged context used for extraction.
    """
    # Build a concise listing of what was extracted
    extraction_summary = ""
    for clause_name, data in extracted_clauses.items():
        label = clause_name.replace("_", " ").upper()
        if data.get("value") is None:
            extraction_summary += f"\n{label}: (not found — skip validation)\n"
        else:
            extraction_summary += f"""
{label}:
  value:      {data.get('value', 'null')}
  page:       {data.get('page', 'null')}
  exact_text: {data.get('exact_text', 'null')}
"""

    return f"""You are a senior legal QA reviewer validating contract clause extraction results.

════════════════════════════════════════
ORIGINAL CONTRACT CONTEXT
════════════════════════════════════════
{original_context[:6000]}
════════════════════════════════════════

════════════════════════════════════════
EXTRACTED CLAUSES TO VALIDATE
════════════════════════════════════════
{extraction_summary}
════════════════════════════════════════

VALIDATION TASK:
For each clause where value is not null, verify:
1. Does the exact_text appear VERBATIM in the contract context above?
2. Does the value accurately summarize what the exact_text says?
3. Is the clause type correctly identified?

STRICT VALIDATION RULES:
- validated = true ONLY if exact_text is found verbatim AND value is accurate.
- validated = false if exact_text is not found verbatim in the context.
- corrected_value = null unless you are highly confident in a correction.
- For clauses where value was null (not found), always set validated = false.
- Return ONLY valid JSON — no preamble, no markdown, no explanation.

RETURN THIS EXACT JSON STRUCTURE:
{{
  "governing_law": {{
    "validated": <true or false>,
    "corrected_value": "<corrected string or null>",
    "reasoning": "<one sentence>"
  }},
  "audit_rights": {{
    "validated": <true or false>,
    "corrected_value": "<corrected string or null>",
    "reasoning": "<one sentence>"
  }},
  "non_compete": {{
    "validated": <true or false>,
    "corrected_value": "<corrected string or null>",
    "reasoning": "<one sentence>"
  }},
  "non_solicitation": {{
    "validated": <true or false>,
    "corrected_value": "<corrected string or null>",
    "reasoning": "<one sentence>"
  }}
}}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_chunks_with_pages(
    chunk_texts: List[str],
    page_numbers: List[int],
    max_chunks: int = 15,
) -> str:
    """
    Format chunks with clear page markers and dividers.
    Limits to max_chunks to stay within token budget.
    """
    parts = []
    for i, (text, page) in enumerate(zip(chunk_texts[:max_chunks], page_numbers[:max_chunks])):
        parts.append(f"[PAGE {page}] — EXCERPT {i + 1} of {min(len(chunk_texts), max_chunks)}\n{text}")
    return "\n\n" + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(parts) + "\n"
