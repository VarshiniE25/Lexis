"""
prompts.py — All LLM prompts for the Contract Intelligence Engine.
Each extraction task has its own carefully engineered prompt.
"""

from __future__ import annotations
from typing import List
from .config import CONTRACT_TYPES


# ─── Contract Type Classification ─────────────────────────────────────────────

def contract_type_prompt(context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    types_list = "\n".join(f"- {t}" for t in CONTRACT_TYPES)
    return f"""You are a senior contract analyst. Classify the contract type based on the provided excerpts.

CONTRACT EXCERPTS:
{context}

TASK: Classify this contract into exactly ONE of the following types:
{types_list}

Return ONLY valid JSON. No preamble, no explanation, no markdown.
If you cannot determine the type with reasonable confidence, still pick the closest match.

Required JSON format:
{{
  "value": "<contract type from the list above>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}}

RULES:
- value MUST be one of the exact strings from the list above.
- confidence reflects how certain you are (0.9+ = very clear, 0.5 = ambiguous).
- Do NOT hallucinate. Base answer on provided text only.
"""


# ─── Clause Extraction Prompts ────────────────────────────────────────────────

def governing_law_prompt(context_chunks: List[str], page_hints: List[int]) -> str:
    context = _format_chunks(context_chunks, page_hints)
    return f"""You are a legal contract analyst. Extract the GOVERNING LAW clause from these contract excerpts.

{context}

TASK: Find the clause that specifies which jurisdiction's laws govern this contract.
Look for phrases like: "governed by", "construed in accordance with", "laws of", "subject to the laws".

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "value": "<concise description of the governing law, e.g. 'Laws of the State of New York'>",
  "exact_text": "<verbatim sentence(s) from the contract that contain this clause>",
  "page": <page number where this clause appears, integer>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

CRITICAL RULES:
- If NOT found in the provided text → return {{"value": null, "exact_text": null, "page": null, "confidence": 0.0, "reasoning": "Not found in retrieved context"}}
- Do NOT hallucinate. Only extract what is explicitly stated.
- exact_text must be VERBATIM from the contract text above.
- page must come from the [PAGE X] markers in the context.
"""


def audit_rights_prompt(context_chunks: List[str], page_hints: List[int]) -> str:
    context = _format_chunks(context_chunks, page_hints)
    return f"""You are a legal contract analyst. Extract the AUDIT RIGHTS clause from these contract excerpts.

{context}

TASK: Find the clause granting one party the right to audit the other party's records, books, or operations.
Look for: "audit", "inspect records", "right to examine", "books and records", "audit rights".

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "value": "<concise description of audit rights, e.g. 'Client may audit vendor books once per year with 30 days notice'>",
  "exact_text": "<verbatim sentence(s) from the contract>",
  "page": <page number, integer>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

CRITICAL RULES:
- If NOT found → return {{"value": null, "exact_text": null, "page": null, "confidence": 0.0, "reasoning": "Not found in retrieved context"}}
- Do NOT hallucinate.
- exact_text must be VERBATIM from the contract text above.
- page must come from the [PAGE X] markers in the context.
"""


def non_compete_prompt(context_chunks: List[str], page_hints: List[int]) -> str:
    context = _format_chunks(context_chunks, page_hints)
    return f"""You are a legal contract analyst. Extract the NON-COMPETE clause from these contract excerpts.

{context}

TASK: Find the clause restricting a party from engaging in competing business activities.
Look for: "non-compete", "non compete", "not compete", "competitive activities", "competing business",
"shall not engage", "restrict from competing".

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "value": "<concise description, e.g. 'Party A may not engage in competing services for 2 years within the United States'>",
  "exact_text": "<verbatim sentence(s) from the contract>",
  "page": <page number, integer>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

CRITICAL RULES:
- If NOT found → return {{"value": null, "exact_text": null, "page": null, "confidence": 0.0, "reasoning": "Not found in retrieved context"}}
- Do NOT hallucinate.
- exact_text must be VERBATIM from the contract text above.
- page must come from the [PAGE X] markers in the context.
"""


def non_solicitation_prompt(context_chunks: List[str], page_hints: List[int]) -> str:
    context = _format_chunks(context_chunks, page_hints)
    return f"""You are a legal contract analyst. Extract the NON-SOLICITATION OF EMPLOYEES clause from these contract excerpts.

{context}

TASK: Find the clause restricting a party from soliciting or hiring the other party's employees.
Look for: "non-solicitation", "solicit employees", "hire employees", "poach", "recruit employees",
"shall not solicit", "not directly or indirectly recruit".

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "value": "<concise description, e.g. 'Neither party may solicit or hire employees of the other for 12 months after termination'>",
  "exact_text": "<verbatim sentence(s) from the contract>",
  "page": <page number, integer>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

CRITICAL RULES:
- If NOT found → return {{"value": null, "exact_text": null, "page": null, "confidence": 0.0, "reasoning": "Not found in retrieved context"}}
- Do NOT hallucinate.
- exact_text must be VERBATIM from the contract text above.
- page must come from the [PAGE X] markers in the context.
"""


# ─── Structured Fields Extraction ─────────────────────────────────────────────

def structured_fields_prompt(context_chunks: List[str], page_hints: List[int]) -> str:
    context = _format_chunks(context_chunks, page_hints)
    return f"""You are a legal contract analyst. Extract structured fields from these contract excerpts.

{context}

TASK: Extract the following four fields:
1. jurisdiction: The legal jurisdiction (state/country) where disputes are resolved.
2. payment_terms: How and when payments are to be made (e.g., "Net 30", "monthly in arrears").
3. notice_period: How much advance notice is required for termination or other actions.
4. liability_cap: The maximum liability amount or limitation clause (e.g., "$500,000" or "total fees paid in prior 12 months").

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "jurisdiction": "<string or null>",
  "payment_terms": "<string or null>",
  "notice_period": "<string or null>",
  "liability_cap": "<string or null>"
}}

CRITICAL RULES:
- Return null for any field not explicitly found in the provided text.
- Do NOT hallucinate values.
- Values should be concise but complete.
"""


# ─── Validation Prompts ───────────────────────────────────────────────────────

def validation_prompt(
    clause_name: str,
    extracted_value: str,
    extracted_exact_text: str,
    original_context: str,
) -> str:
    return f"""You are a senior legal reviewer performing quality control on contract clause extraction.

CLAUSE TYPE: {clause_name}

EXTRACTED VALUE:
{extracted_value}

CLAIMED EXACT TEXT:
{extracted_exact_text}

ORIGINAL CONTRACT CONTEXT:
{original_context}

TASK: Verify whether:
1. The exact_text actually appears verbatim in the original context.
2. The extracted value accurately represents what the exact_text says.
3. The clause type is correctly identified.

Return ONLY valid JSON. No preamble, no markdown.

Required JSON format:
{{
  "validated": <true or false>,
  "corrected_value": "<corrected value if needed, or null if no correction>",
  "reasoning": "<one sentence explanation of your decision>"
}}

RULES:
- validated = true only if both the exact_text is found verbatim AND the value is accurate.
- If exact_text is not found verbatim, validated = false.
- corrected_value is null unless you are confident in a correction.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_chunks(chunks: List[str], page_hints: List[int]) -> str:
    """Format chunks with page markers for context."""
    parts = []
    for i, (chunk, page) in enumerate(zip(chunks, page_hints)):
        parts.append(f"[PAGE {page}] EXCERPT {i + 1}:\n{chunk}")
    return "\n\n" + "\n\n---\n\n".join(parts) + "\n"
