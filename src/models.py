"""
models.py — Pydantic data models for the Contract Intelligence Engine.
These define the exact schema of the final output JSON.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Sub-models ───────────────────────────────────────────────────────────────

class ContractTypeResult(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ClauseResult(BaseModel):
    value: Optional[str] = None
    page: Optional[int] = None
    exact_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    validated: bool = False

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))


class StructuredFields(BaseModel):
    jurisdiction: Optional[str] = None
    payment_terms: Optional[str] = None
    notice_period: Optional[str] = None
    liability_cap: Optional[str] = None


class ClausesResult(BaseModel):
    governing_law: ClauseResult = Field(default_factory=ClauseResult)
    audit_rights: ClauseResult = Field(default_factory=ClauseResult)
    non_compete: ClauseResult = Field(default_factory=ClauseResult)
    non_solicitation: ClauseResult = Field(default_factory=ClauseResult)


class ContractExtractionResult(BaseModel):
    """
    Final output model. Matches the required JSON schema exactly.
    """
    contract_type: ContractTypeResult = Field(default_factory=ContractTypeResult)
    clauses: ClausesResult = Field(default_factory=ClausesResult)
    fields: StructuredFields = Field(default_factory=StructuredFields)

    def to_output_dict(self) -> dict:
        """Serialize to the exact required output format."""
        return {
            "contract_type": {
                "value": self.contract_type.value,
                "confidence": round(self.contract_type.confidence, 4),
            },
            "clauses": {
                "governing_law": self._clause_dict(self.clauses.governing_law),
                "audit_rights": self._clause_dict(self.clauses.audit_rights),
                "non_compete": self._clause_dict(self.clauses.non_compete),
                "non_solicitation": self._clause_dict(self.clauses.non_solicitation),
            },
            "fields": {
                "jurisdiction": self.fields.jurisdiction,
                "payment_terms": self.fields.payment_terms,
                "notice_period": self.fields.notice_period,
                "liability_cap": self.fields.liability_cap,
            },
        }

    @staticmethod
    def _clause_dict(clause: ClauseResult) -> dict:
        return {
            "value": clause.value,
            "page": clause.page,
            "exact_text": clause.exact_text,
            "confidence": round(clause.confidence, 4),
            "validated": clause.validated,
        }


# ─── Internal Pipeline Models ─────────────────────────────────────────────────

class PageChunk(BaseModel):
    """A text chunk with its source page number(s)."""
    text: str
    start_page: int
    end_page: int
    chunk_index: int
    word_count: int


class RetrievedChunk(BaseModel):
    """A chunk retrieved from FAISS with its similarity score."""
    chunk: PageChunk
    score: float


class LLMExtractionResponse(BaseModel):
    """Raw LLM response for a clause or field extraction."""
    value: Optional[str] = None
    exact_text: Optional[str] = None
    page: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class ValidationResponse(BaseModel):
    """LLM validation response."""
    validated: bool = False
    corrected_value: Optional[str] = None
    reasoning: Optional[str] = None
