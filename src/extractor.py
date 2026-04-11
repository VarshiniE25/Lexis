"""
extractor.py — Core extraction pipeline for the Contract Intelligence Engine.

Pipeline:
  PDF → Parse → Chunk → Embed → FAISS Index
  → Parallel RAG-based Clause Extraction (asyncio.gather)
  → Parallel Validation
  → Structured JSON Output
"""

from __future__ import annotations
import asyncio
import time
from pathlib import Path
from typing import Optional

from .pdf_parser import PDFParser, ParsedDocument
from .chunker import TextChunker
from .embeddings import FAISSIndex
from .retriever import ContractRetriever
from .llm_client import llm_call
from .validator import ClauseValidator
from .models import (
    ContractExtractionResult,
    ContractTypeResult,
    ClauseResult,
    ClausesResult,
    StructuredFields,
    PageChunk,
)
from .prompts import (
    contract_type_prompt,
    governing_law_prompt,
    audit_rights_prompt,
    non_compete_prompt,
    non_solicitation_prompt,
    structured_fields_prompt,
)
from .config import CLAUSE_LABELS
from .logger import get_logger

logger = get_logger(__name__)


class ContractExtractor:
    """
    Main pipeline class for contract intelligence extraction.
    Orchestrates: parsing → chunking → indexing → retrieval → LLM → validation.
    """

    def __init__(self):
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.index = FAISSIndex()
        self.retriever: Optional[ContractRetriever] = None
        self.validator = ClauseValidator()

        # Store for explainability
        self.retrieved_chunks: dict = {}
        self.parsed_doc: Optional[ParsedDocument] = None

    # ─── Public Entry Points ──────────────────────────────────────────────────

    def process_file(self, pdf_path: str | Path) -> ContractExtractionResult:
        """Process a PDF file from disk."""
        return asyncio.run(self._async_process_file(pdf_path))

    def process_bytes(self, pdf_bytes: bytes, file_name: str = "upload.pdf") -> ContractExtractionResult:
        """Process a PDF from raw bytes (Streamlit uploads)."""
        return asyncio.run(self._async_process_bytes(pdf_bytes, file_name))

    # ─── Async Pipeline ───────────────────────────────────────────────────────

    async def _async_process_file(self, pdf_path: str | Path) -> ContractExtractionResult:
        doc = self.parser.parse(pdf_path)
        return await self._run_pipeline(doc)

    async def _async_process_bytes(self, pdf_bytes: bytes, file_name: str) -> ContractExtractionResult:
        doc = self.parser.parse_from_bytes(pdf_bytes, file_name)
        return await self._run_pipeline(doc)

    async def _run_pipeline(self, doc: ParsedDocument) -> ContractExtractionResult:
        start_time = time.perf_counter()
        self.parsed_doc = doc

        logger.info(f"Processing: {doc.file_name} ({doc.total_pages} pages, {doc.total_words} words)")

        # ── Step 1: Chunk ──────────────────────────────────────────────────────
        logger.info("Step 1/5: Chunking document...")
        chunks = self.chunker.chunk(doc)
        logger.info(f"Created {len(chunks)} chunks")

        # ── Step 2: Embed + Index ──────────────────────────────────────────────
        logger.info("Step 2/5: Building FAISS index...")
        self.index.build(chunks)

        # ── Step 3: Retrieval (all clauses) ───────────────────────────────────
        logger.info("Step 3/5: Retrieving relevant chunks per clause...")
        self.retriever = ContractRetriever(self.index)
        all_contexts = self.retriever.retrieve_all()

        # Store for explainability UI
        self.retrieved_chunks = all_contexts

        # ── Step 4: Parallel LLM Extraction ───────────────────────────────────
        logger.info("Step 4/5: Running parallel LLM extraction...")
        (
            contract_type_result,
            governing_law,
            audit_rights,
            non_compete,
            non_solicitation,
            fields,
        ) = await asyncio.gather(
            self._extract_contract_type(all_contexts),
            self._extract_governing_law(all_contexts),
            self._extract_audit_rights(all_contexts),
            self._extract_non_compete(all_contexts),
            self._extract_non_solicitation(all_contexts),
            self._extract_structured_fields(all_contexts),
        )

        # ── Step 5: Validation Pass ────────────────────────────────────────────
        logger.info("Step 5/5: Running validation pass...")
        raw_clauses = {
            "governing_law": governing_law,
            "audit_rights": audit_rights,
            "non_compete": non_compete,
            "non_solicitation": non_solicitation,
        }

        validated_clauses = await self.validator.validate_all_clauses(
            clauses=raw_clauses,
            contexts=all_contexts,
            clause_labels=CLAUSE_LABELS,
        )

        # ── Assemble Final Result ──────────────────────────────────────────────
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
        logger.info(f"Pipeline complete in {elapsed:.2f}s")

        return result

    # ─── Individual Extractors ────────────────────────────────────────────────

    async def _extract_contract_type(self, contexts: dict) -> ContractTypeResult:
        chunk_texts, _ = contexts.get("contract_type", ([], []))
        if not chunk_texts:
            logger.warning("No context retrieved for contract type")
            return ContractTypeResult()

        prompt = contract_type_prompt(chunk_texts[:5])
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

    async def _extract_governing_law(self, contexts: dict) -> ClauseResult:
        return await self._extract_clause(
            "governing_law",
            contexts,
            governing_law_prompt,
        )

    async def _extract_audit_rights(self, contexts: dict) -> ClauseResult:
        return await self._extract_clause(
            "audit_rights",
            contexts,
            audit_rights_prompt,
        )

    async def _extract_non_compete(self, contexts: dict) -> ClauseResult:
        return await self._extract_clause(
            "non_compete",
            contexts,
            non_compete_prompt,
        )

    async def _extract_non_solicitation(self, contexts: dict) -> ClauseResult:
        return await self._extract_clause(
            "non_solicitation",
            contexts,
            non_solicitation_prompt,
        )

    async def _extract_clause(
        self,
        clause_name: str,
        contexts: dict,
        prompt_fn,
    ) -> ClauseResult:
        chunk_texts, page_numbers = contexts.get(clause_name, ([], []))

        if not chunk_texts:
            logger.warning(f"No context retrieved for: {clause_name}")
            return ClauseResult()

        prompt = prompt_fn(chunk_texts[:5], page_numbers[:5])
        response = await llm_call(prompt)

        if response is None:
            return ClauseResult()

        try:
            result = ClauseResult(
                value=response.get("value"),
                exact_text=response.get("exact_text"),
                page=response.get("page"),
                confidence=float(response.get("confidence", 0.0)),
                validated=False,  # Will be set in validation pass
            )
            logger.info(
                f"[{clause_name}] Extracted: "
                f"value={str(result.value)[:60] if result.value else 'null'}, "
                f"confidence={result.confidence:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Clause parsing error for {clause_name}: {e}")
            return ClauseResult()

    async def _extract_structured_fields(self, contexts: dict) -> StructuredFields:
        # Combine contexts from multiple relevant clause types
        all_chunks = []
        all_pages = []

        for key in ["jurisdiction", "payment_terms", "notice_period", "liability_cap"]:
            chunks, pages = contexts.get(key, ([], []))
            all_chunks.extend(chunks[:2])  # Top 2 from each
            all_pages.extend(pages[:2])

        # Deduplicate
        seen = set()
        deduped_chunks = []
        deduped_pages = []
        for chunk, page in zip(all_chunks, all_pages):
            if chunk not in seen:
                seen.add(chunk)
                deduped_chunks.append(chunk)
                deduped_pages.append(page)

        if not deduped_chunks:
            logger.warning("No context for structured fields")
            return StructuredFields()

        prompt = structured_fields_prompt(deduped_chunks[:6], deduped_pages[:6])
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
            logger.error(f"Structured fields parsing error: {e}")
            return StructuredFields()
