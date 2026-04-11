"""
joint_extractor.py — Joint Multi-Clause Extraction Pipeline.

LLM Call Budget Comparison:
─────────────────────────────────────────────────────────
 Old (extractor.py)          │ New (joint_extractor.py)
─────────────────────────────────────────────────────────
 1 call → contract type      │ 1 call → contract type
 1 call → governing_law      │                         ┐
 1 call → audit_rights       │ 1 call → ALL 4 clauses  │ joint
 1 call → non_compete        │                         │
 1 call → non_solicitation   │                         ┘
 1 call → structured fields  │ 1 call → ALL 4 fields
 4 calls → 4x validation     │ 1 call → ALL validation  joint
─────────────────────────────────────────────────────────
 TOTAL: 9+ LLM calls         │ TOTAL: 4 LLM calls  ✅
 For 20 clauses: 40+ calls   │ For 20 clauses: ~4 calls ✅
─────────────────────────────────────────────────────────

Architecture:
  PDF → Parse → Chunk → Embed → FAISS
  → JointRetriever (2-pass: global + filtered)
  → asyncio.gather([
        joint_clause_extraction (4 clauses, 1 call),
        joint_fields_extraction  (4 fields,  1 call),
        contract_type_extraction (1 call),
    ])
  → joint_validation (4 clauses, 1 call)
  → Structured JSON
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pdf_parser import PDFParser, ParsedDocument
from .chunker import TextChunker
from .embeddings import FAISSIndex
from .joint_retriever import JointRetriever, JointContext
from .llm_client import llm_call
from .models import (
    ContractExtractionResult,
    ContractTypeResult,
    ClauseResult,
    ClausesResult,
    StructuredFields,
)
from .joint_prompts import (
    joint_clauses_prompt,
    joint_fields_prompt,
    joint_contract_type_prompt,
    joint_validation_prompt,
    _format_chunks_with_pages,
)
from .logger import get_logger

logger = get_logger(__name__)


# ─── Call Counter (for demonstration / benchmarking) ──────────────────────────

@dataclass
class LLMCallStats:
    """Tracks LLM call counts for comparison reporting."""
    total_calls: int = 0
    call_log: List[str] = field(default_factory=list)

    def record(self, label: str):
        self.total_calls += 1
        self.call_log.append(label)
        logger.info(f"[LLM Call #{self.total_calls}] {label}")

    def summary(self) -> str:
        lines = [f"Total LLM calls: {self.total_calls}"]
        for i, label in enumerate(self.call_log, 1):
            lines.append(f"  Call {i}: {label}")
        return "\n".join(lines)


# ─── Joint Extractor ─────────────────────────────────────────────────────────

class JointContractExtractor:
    """
    Optimized contract extractor using joint multi-clause extraction.

    Key optimization: replaces N individual LLM calls with 1 joint call
    by combining all clause queries into a single richly-prompted LLM request.

    Usage:
        extractor = JointContractExtractor()
        result = extractor.process_file("contract.pdf")
        print(f"LLM calls used: {extractor.call_stats.total_calls}")  # → 4
    """

    def __init__(
        self,
        global_top_k: int = 20,
        filtered_top_k: int = 10,
    ):
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.index = FAISSIndex()
        self.joint_retriever: Optional[JointRetriever] = None

        self.global_top_k = global_top_k
        self.filtered_top_k = filtered_top_k

        # Observability
        self.call_stats = LLMCallStats()
        self.retrieved_chunks: Dict = {}
        self.joint_context: Optional[JointContext] = None
        self.parsed_doc: Optional[ParsedDocument] = None

    # ─── Public Entry Points ──────────────────────────────────────────────────

    def process_file(self, pdf_path: str | Path) -> ContractExtractionResult:
        """Process a PDF file from disk."""
        return asyncio.run(self._async_process_file(pdf_path))

    def process_bytes(
        self,
        pdf_bytes: bytes,
        file_name: str = "upload.pdf",
    ) -> ContractExtractionResult:
        """Process a PDF from raw bytes (Streamlit uploads)."""
        return asyncio.run(self._async_process_bytes(pdf_bytes, file_name))

    # ─── Async Pipeline ───────────────────────────────────────────────────────

    async def _async_process_file(self, pdf_path: str | Path) -> ContractExtractionResult:
        doc = self.parser.parse(pdf_path)
        return await self._run_pipeline(doc)

    async def _async_process_bytes(
        self,
        pdf_bytes: bytes,
        file_name: str,
    ) -> ContractExtractionResult:
        doc = self.parser.parse_from_bytes(pdf_bytes, file_name)
        return await self._run_pipeline(doc)

    async def _run_pipeline(self, doc: ParsedDocument) -> ContractExtractionResult:
        start_time = time.perf_counter()
        self.parsed_doc = doc
        self.call_stats = LLMCallStats()  # Reset for each run

        logger.info(
            f"[JointExtractor] Processing: {doc.file_name} "
            f"({doc.total_pages} pages, {doc.total_words} words)"
        )

        # ── Step 1: Chunk ──────────────────────────────────────────────────
        logger.info("Step 1/5: Chunking document...")
        chunks = self.chunker.chunk(doc)
        logger.info(f"  → {len(chunks)} chunks created")

        # ── Step 2: Embed + FAISS Index ────────────────────────────────────
        logger.info("Step 2/5: Building FAISS index...")
        self.index.build(chunks)

        # ── Step 3: Two-Pass Joint Retrieval ───────────────────────────────
        logger.info("Step 3/5: Running 2-pass joint retrieval...")
        self.joint_retriever = JointRetriever(
            self.index,
            global_top_k=self.global_top_k,
            filtered_top_k=self.filtered_top_k,
        )
        joint_ctx = self.joint_retriever.build_joint_context()
        self.joint_context = joint_ctx
        logger.info(
            f"  → Joint context: {len(joint_ctx.all_chunks)} unique chunks, "
            f"pages {joint_ctx.all_chunks[0].start_page if joint_ctx.all_chunks else '?'}"
            f"–{joint_ctx.all_chunks[-1].end_page if joint_ctx.all_chunks else '?'}"
        )

        # Prepare context arrays for prompts
        all_texts = [c.text for c in joint_ctx.all_chunks]
        all_pages = [c.start_page for c in joint_ctx.all_chunks]

        # For explainability UI (compatible with existing UI code)
        self.retrieved_chunks = self._build_explainability_map(joint_ctx)

        # Contract type uses its own focused context
        ct_texts, ct_pages = self.joint_retriever.build_contract_type_context()

        # ── Step 4: Parallel Joint LLM Extraction ─────────────────────────
        # 3 concurrent LLM calls replace what was 6+ sequential/parallel calls
        logger.info("Step 4/5: Running joint LLM extraction (3 parallel calls)...")
        logger.info("  → Call A: Contract type classification")
        logger.info("  → Call B: ALL 4 clauses joint extraction")
        logger.info("  → Call C: ALL 4 structured fields joint extraction")

        contract_type_result, raw_clauses, fields = await asyncio.gather(
            self._extract_contract_type_joint(ct_texts),
            self._extract_all_clauses_joint(all_texts, all_pages),
            self._extract_all_fields_joint(all_texts, all_pages),
        )

        # ── Step 5: Joint Validation ───────────────────────────────────────
        # 1 validation call replaces what was 4 separate validation calls
        logger.info("Step 5/5: Running joint validation (1 call for all clauses)...")
        validated_clauses = await self._validate_all_clauses_joint(
            raw_clauses,
            all_texts,
            all_pages,
        )

        # ── Assemble Result ────────────────────────────────────────────────
        result = ContractExtractionResult(
            contract_type=contract_type_result,
            clauses=ClausesResult(
                governing_law=validated_clauses["governing_law"],
                audit_rights=validated_clauses["audit_rights"],
                non_compete=validated_clauses["non_compete"],
                non_solicitation=validated_clauses["non_solicitation"],
            ),
            fields=fields,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[JointExtractor] ✅ Done in {elapsed:.2f}s | "
            f"{self.call_stats.summary()}"
        )

        return result

    # ─── Joint LLM Extractors ─────────────────────────────────────────────────

    async def _extract_contract_type_joint(
        self,
        chunk_texts: List[str],
    ) -> ContractTypeResult:
        """Call A: Contract type classification (1 LLM call)."""
        if not chunk_texts:
            logger.warning("No context for contract type")
            return ContractTypeResult()

        prompt = joint_contract_type_prompt(chunk_texts[:6])
        self.call_stats.record("Contract Type Classification")

        response = await llm_call(prompt)
        if response is None:
            return ContractTypeResult()

        try:
            return ContractTypeResult(
                value=response.get("value"),
                confidence=float(response.get("confidence", 0.0)),
            )
        except Exception as e:
            logger.error(f"Contract type parsing error: {e}")
            return ContractTypeResult()

    async def _extract_all_clauses_joint(
        self,
        chunk_texts: List[str],
        page_numbers: List[int],
    ) -> Dict[str, ClauseResult]:
        """
        Call B: Extract ALL 4 clauses in ONE LLM call.

        This is the core optimization — what was 4 calls is now 1.
        The LLM sees the full merged context and extracts all clauses at once.
        """
        if not chunk_texts:
            logger.warning("No context for joint clause extraction")
            return self._empty_clauses()

        prompt = joint_clauses_prompt(chunk_texts, page_numbers)
        self.call_stats.record("Joint Clause Extraction (governing_law + audit_rights + non_compete + non_solicitation)")

        response = await llm_call(prompt)
        if response is None:
            logger.error("Joint clause extraction returned None")
            return self._empty_clauses()

        return self._parse_joint_clause_response(response)

    async def _extract_all_fields_joint(
        self,
        chunk_texts: List[str],
        page_numbers: List[int],
    ) -> StructuredFields:
        """
        Call C: Extract ALL 4 structured fields in ONE LLM call.
        """
        if not chunk_texts:
            logger.warning("No context for joint fields extraction")
            return StructuredFields()

        prompt = joint_fields_prompt(chunk_texts, page_numbers)
        self.call_stats.record("Joint Fields Extraction (jurisdiction + payment_terms + notice_period + liability_cap)")

        response = await llm_call(prompt)
        if response is None:
            return StructuredFields()

        try:
            return StructuredFields(
                jurisdiction=response.get("jurisdiction"),
                payment_terms=response.get("payment_terms"),
                notice_period=response.get("notice_period"),
                liability_cap=response.get("liability_cap"),
            )
        except Exception as e:
            logger.error(f"Joint fields parsing error: {e}")
            return StructuredFields()

    async def _validate_all_clauses_joint(
        self,
        raw_clauses: Dict[str, ClauseResult],
        chunk_texts: List[str],
        page_numbers: List[int],
    ) -> Dict[str, ClauseResult]:
        """
        Call D: Validate ALL 4 clauses in ONE validation LLM call.

        Replaces 4 separate validation calls with 1 joint call.
        """
        # Skip validation if all clauses are null
        has_any_value = any(
            c.value is not None for c in raw_clauses.values()
        )
        if not has_any_value:
            logger.info("All clauses null — skipping validation")
            return raw_clauses

        # Build context string for validator (truncated to stay in token budget)
        context = _format_chunks_with_pages(
            chunk_texts[:12],
            page_numbers[:12],
        )

        # Build extracted_clauses dict for the prompt
        extracted_for_prompt = {
            name: {
                "value": c.value,
                "exact_text": c.exact_text,
                "page": c.page,
                "confidence": c.confidence,
            }
            for name, c in raw_clauses.items()
        }

        prompt = joint_validation_prompt(
            extracted_clauses=extracted_for_prompt,
            original_context=context,
        )
        self.call_stats.record("Joint Validation (all 4 clauses)")

        response = await llm_call(prompt)

        if response is None:
            logger.warning("Joint validation returned None — marking all as unvalidated")
            for clause in raw_clauses.values():
                clause.validated = False
            return raw_clauses

        return self._apply_validation_response(raw_clauses, response)

    # ─── Parsing Helpers ─────────────────────────────────────────────────────

    def _parse_joint_clause_response(
        self,
        response: dict,
    ) -> Dict[str, ClauseResult]:
        """
        Parse the joint clause extraction response into ClauseResult objects.
        Handles partial responses gracefully — missing keys default to null.
        """
        clause_names = ["governing_law", "audit_rights", "non_compete", "non_solicitation"]
        results = {}

        for name in clause_names:
            data = response.get(name, {})

            if not isinstance(data, dict):
                logger.warning(f"Unexpected type for clause '{name}': {type(data)}")
                results[name] = ClauseResult()
                continue

            try:
                clause = ClauseResult(
                    value=data.get("value"),
                    exact_text=data.get("exact_text"),
                    page=data.get("page"),
                    confidence=float(data.get("confidence", 0.0)),
                    validated=False,  # Set in validation step
                )
                logger.info(
                    f"  [{name}] extracted: "
                    f"value={'✓ ' + str(clause.value)[:50] if clause.value else '✗ null'}, "
                    f"confidence={clause.confidence:.2f}"
                )
                results[name] = clause
            except Exception as e:
                logger.error(f"Parsing error for clause '{name}': {e}")
                results[name] = ClauseResult()

        return results

    def _apply_validation_response(
        self,
        raw_clauses: Dict[str, ClauseResult],
        validation_response: dict,
    ) -> Dict[str, ClauseResult]:
        """
        Apply joint validation results to all clause objects.
        """
        clause_names = ["governing_law", "audit_rights", "non_compete", "non_solicitation"]

        for name in clause_names:
            clause = raw_clauses.get(name, ClauseResult())
            val_data = validation_response.get(name, {})

            if not isinstance(val_data, dict):
                clause.validated = False
                continue

            validated = bool(val_data.get("validated", False))
            corrected = val_data.get("corrected_value")
            reasoning = val_data.get("reasoning", "")

            clause.validated = validated

            # Apply correction if validator provides one and extraction failed
            if corrected and not validated and clause.value is not None:
                logger.info(f"  [{name}] Correction applied: {corrected[:60]}")
                clause.value = corrected

            logger.info(
                f"  [{name}] Validation: {'✅' if validated else '❌'} — {reasoning}"
            )
            raw_clauses[name] = clause

        return raw_clauses

    @staticmethod
    def _empty_clauses() -> Dict[str, ClauseResult]:
        return {
            "governing_law": ClauseResult(),
            "audit_rights": ClauseResult(),
            "non_compete": ClauseResult(),
            "non_solicitation": ClauseResult(),
        }

    def _build_explainability_map(
        self,
        joint_ctx: JointContext,
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Build the retrieved_chunks dict in the same format as the original
        extractor so the Streamlit UI remains compatible.
        """
        from .joint_retriever import CLAUSE_GROUPS

        explainability: Dict[str, Tuple[List[str], List[int]]] = {}

        # All chunks for global view
        all_texts = [c.text for c in joint_ctx.all_chunks]
        all_pages = [c.start_page for c in joint_ctx.all_chunks]

        # Per-group chunks for clause-level explainability
        for group_name, clause_names in CLAUSE_GROUPS.items():
            group_chunks = joint_ctx.filtered_chunks.get(group_name, [])
            group_texts = [c.text for c, _ in group_chunks]
            group_pages = [c.start_page for c, _ in group_chunks]
            for clause_name in clause_names:
                explainability[clause_name] = (group_texts, group_pages)

        # Add contract_type with its own context
        ct_texts, ct_pages = self.joint_retriever.build_contract_type_context()
        explainability["contract_type"] = (ct_texts, ct_pages)

        return explainability
