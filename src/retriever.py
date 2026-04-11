"""
retriever.py — Per-clause retrieval queries and context assembly.
Maps each clause/field type to optimal search queries.
"""

from __future__ import annotations
from typing import List, Dict, Tuple

from .embeddings import FAISSIndex
from .models import PageChunk
from .config import TOP_K_CHUNKS
from .logger import get_logger

logger = get_logger(__name__)


# ─── Query Templates per Extraction Target ────────────────────────────────────

CLAUSE_QUERIES: Dict[str, List[str]] = {
    "contract_type": [
        "this agreement is a service agreement",
        "this lease agreement",
        "intellectual property agreement license",
        "supply agreement purchase order vendor",
        "nature and purpose of this agreement",
        "this agreement between the parties",
    ],
    "governing_law": [
        "governing law jurisdiction",
        "this agreement shall be governed by the laws",
        "construed in accordance with the laws",
        "subject to the laws of the state",
        "applicable law governing this contract",
    ],
    "audit_rights": [
        "audit rights books records",
        "right to audit inspect examine",
        "audit the books and records",
        "inspection of records audit",
        "party may audit financial records",
    ],
    "non_compete": [
        "non-compete non compete clause",
        "shall not engage in competing business",
        "competitive activities restriction",
        "not compete directly indirectly",
        "compete with similar services products",
    ],
    "non_solicitation": [
        "non-solicitation employees staff",
        "shall not solicit or hire employees",
        "solicitation of employees prohibition",
        "not recruit poach employees",
        "solicit employees personnel workers",
    ],
    "jurisdiction": [
        "jurisdiction venue courts",
        "exclusive jurisdiction courts of",
        "submit to jurisdiction",
        "disputes resolved in courts of",
    ],
    "payment_terms": [
        "payment terms net 30 days invoice",
        "shall pay within days of invoice",
        "payment due date schedule",
        "fees payment schedule monthly",
        "invoicing and payment",
    ],
    "notice_period": [
        "notice period days written notice",
        "termination notice required",
        "days prior written notice",
        "notice of termination",
        "advance notice required",
    ],
    "liability_cap": [
        "liability cap limitation damages",
        "maximum liability shall not exceed",
        "aggregate liability limited to",
        "cap on liability total fees",
        "limitation of liability clause",
    ],
}


class ContractRetriever:
    """
    Retrieves relevant context chunks for each clause/field type.
    Uses multi-query search to maximize recall.
    """

    def __init__(self, index: FAISSIndex, top_k: int = TOP_K_CHUNKS):
        self.index = index
        self.top_k = top_k

    def retrieve_for_clause(
        self,
        clause_name: str,
    ) -> List[Tuple[PageChunk, float]]:
        """
        Retrieve top-k chunks for a specific clause.

        Args:
            clause_name: One of the keys in CLAUSE_QUERIES.

        Returns:
            List of (PageChunk, score) sorted by relevance.
        """
        queries = CLAUSE_QUERIES.get(clause_name, [clause_name])
        logger.debug(f"Retrieving for '{clause_name}' with {len(queries)} queries")

        results = self.index.search_multi(
            queries=queries,
            top_k=self.top_k,
            deduplicate=True,
        )

        logger.info(
            f"[{clause_name}] Retrieved {len(results)} chunks, "
            f"top score: {results[0][1]:.3f}" if results else f"[{clause_name}] No chunks found"
        )

        return results

    def get_context_for_clause(
        self,
        clause_name: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Get formatted context for a clause.

        Returns:
            Tuple of (list of chunk texts, list of start pages)
        """
        results = self.retrieve_for_clause(clause_name)

        chunk_texts = []
        page_numbers = []

        for chunk, score in results:
            chunk_texts.append(chunk.text)
            page_numbers.append(chunk.start_page)

        return chunk_texts, page_numbers

    def retrieve_all(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Retrieve context for all clause types.

        Returns:
            Dict mapping clause_name → (chunk_texts, page_numbers)
        """
        all_contexts = {}
        for clause_name in CLAUSE_QUERIES:
            all_contexts[clause_name] = self.get_context_for_clause(clause_name)
        return all_contexts
