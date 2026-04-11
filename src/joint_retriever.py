"""
joint_retriever.py — Smart dual-pass retrieval for Joint Multi-Clause Extraction.

Strategy:
  Pass 1 → Global retrieval: fetch top-N chunks that cover the whole contract,
            using broad legal queries. These anchor the LLM in the document.
  Pass 2 → Filtered retrieval: for each clause group, pull top-K targeted chunks
            using clause-specific queries. These sharpen extraction accuracy.

The two passes are merged, deduplicated, and sorted by page number so the LLM
sees a coherent, document-order context window — not a random bag of snippets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .embeddings import FAISSIndex
from .models import PageChunk
from .config import TOP_K_CHUNKS
from .logger import get_logger

logger = get_logger(__name__)


# ─── Retrieval Config ─────────────────────────────────────────────────────────

# Broad queries that surface general contract structure
GLOBAL_QUERIES: List[str] = [
    "agreement between parties terms and conditions",
    "governing law jurisdiction applicable laws",
    "payment terms invoicing fees compensation",
    "termination notice period cancellation",
    "limitation of liability damages cap",
    "non-compete non-solicitation restrictive covenants",
    "audit rights books records inspection",
    "intellectual property confidentiality obligations",
    "representations warranties indemnification",
    "dispute resolution arbitration courts",
]

# Per-clause targeted queries (same as before, but used in focused pass)
CLAUSE_QUERIES: Dict[str, List[str]] = {
    "governing_law": [
        "this agreement shall be governed by the laws",
        "construed in accordance with the laws of",
        "governing law applicable jurisdiction",
    ],
    "audit_rights": [
        "right to audit books and records",
        "audit inspection examination of records",
        "party may audit financial records annually",
    ],
    "non_compete": [
        "shall not engage in competing business activities",
        "non-compete restriction competitive activities",
        "not compete directly or indirectly",
    ],
    "non_solicitation": [
        "shall not solicit or hire employees of the other party",
        "non-solicitation of employees prohibition",
        "not recruit poach employees personnel",
    ],
    "jurisdiction": [
        "exclusive jurisdiction courts venue",
        "submit to the jurisdiction of",
        "disputes resolved in the courts of",
    ],
    "payment_terms": [
        "payment due within days of invoice net 30",
        "invoicing payment schedule fees compensation",
        "shall pay monthly quarterly annually",
    ],
    "notice_period": [
        "days prior written notice of termination",
        "notice period advance written notice required",
        "written notice to terminate this agreement",
    ],
    "liability_cap": [
        "aggregate liability shall not exceed total fees",
        "maximum liability cap limitation of damages",
        "in no event shall liability exceed",
    ],
}

# Clause groups — which clauses share enough semantic overlap to share context
# (used for further optimization: 1 retrieval pass covers N clauses)
CLAUSE_GROUPS: Dict[str, List[str]] = {
    "restrictive_covenants": ["non_compete", "non_solicitation"],
    "legal_framework":       ["governing_law", "jurisdiction"],
    "financial":             ["payment_terms", "liability_cap"],
    "operational":           ["audit_rights", "notice_period"],
}

# Reverse map: clause → group
CLAUSE_TO_GROUP: Dict[str, str] = {
    clause: group
    for group, clauses in CLAUSE_GROUPS.items()
    for clause in clauses
}


@dataclass
class JointContext:
    """
    Assembled context ready for a joint LLM extraction call.

    Attributes:
        global_chunks:   Top-N broad chunks (document-wide coverage).
        filtered_chunks: Per-clause-group targeted chunks (deep focus).
        all_chunks:      Merged, deduplicated, page-sorted view for prompting.
        chunk_page_map:  chunk_text → page_number for grounding.
    """
    global_chunks: List[Tuple[PageChunk, float]]
    filtered_chunks: Dict[str, List[Tuple[PageChunk, float]]]  # group → chunks
    all_chunks: List[PageChunk]
    chunk_page_map: Dict[int, int]  # chunk_index → page_number


class JointRetriever:
    """
    Two-pass retriever that assembles a rich, deduplicated context window
    for joint multi-clause extraction in a single LLM call.

    LLM Call Reduction:
        Old approach: 1 retrieval + 1 LLM call per clause = N calls total
        New approach: 2-pass retrieval + 1 joint LLM call = 1 call total
                      (+ 1 joint validation call = 2 calls total for ALL clauses)
    """

    def __init__(
        self,
        index: FAISSIndex,
        global_top_k: int = 20,
        filtered_top_k: int = 10,
    ):
        self.index = index
        self.global_top_k = global_top_k
        self.filtered_top_k = filtered_top_k

    # ─── Public Interface ─────────────────────────────────────────────────────

    def build_joint_context(self) -> JointContext:
        """
        Execute both retrieval passes and return a JointContext.

        Returns a fully assembled context with:
        - global_chunks: broad document coverage
        - filtered_chunks: targeted per-group chunks
        - all_chunks: merged, deduplicated, page-ordered
        """
        logger.info("JointRetriever: Pass 1 — Global retrieval...")
        global_chunks = self._global_pass()
        logger.info(f"  → {len(global_chunks)} global chunks retrieved")

        logger.info("JointRetriever: Pass 2 — Filtered retrieval per clause group...")
        filtered_chunks = self._filtered_pass()
        for group, chunks in filtered_chunks.items():
            logger.info(f"  → [{group}] {len(chunks)} chunks")

        logger.info("JointRetriever: Merging + deduplicating...")
        all_chunks = self._merge_and_sort(global_chunks, filtered_chunks)
        logger.info(f"  → {len(all_chunks)} unique chunks after merge")

        chunk_page_map = {c.chunk_index: c.start_page for c in all_chunks}

        return JointContext(
            global_chunks=global_chunks,
            filtered_chunks=filtered_chunks,
            all_chunks=all_chunks,
            chunk_page_map=chunk_page_map,
        )

    def build_contract_type_context(self) -> Tuple[List[str], List[int]]:
        """
        Separate, lightweight retrieval for contract type classification.
        Contract type rarely needs the full joint context.
        """
        queries = [
            "this agreement is entered into between",
            "nature and purpose of this agreement",
            "service agreement master services",
            "lease agreement landlord tenant property",
            "intellectual property license assignment",
            "supply agreement purchase vendor",
        ]
        results = self.index.search_multi(queries, top_k=5, deduplicate=True)
        texts = [r[0].text for r in results]
        pages = [r[0].start_page for r in results]
        return texts, pages

    # ─── Private Passes ───────────────────────────────────────────────────────

    def _global_pass(self) -> List[Tuple[PageChunk, float]]:
        """
        Pass 1: Broad multi-query search across the whole document.
        Retrieves top-N chunks that provide document-wide context.
        """
        return self.index.search_multi(
            queries=GLOBAL_QUERIES,
            top_k=self.global_top_k,
            deduplicate=True,
        )

    def _filtered_pass(self) -> Dict[str, List[Tuple[PageChunk, float]]]:
        """
        Pass 2: Targeted per-clause-group search.
        Each group gets its own top-K chunks from clause-specific queries.
        """
        group_results: Dict[str, List[Tuple[PageChunk, float]]] = {}

        for group_name, clause_names in CLAUSE_GROUPS.items():
            # Aggregate queries for all clauses in this group
            group_queries: List[str] = []
            for clause in clause_names:
                group_queries.extend(CLAUSE_QUERIES.get(clause, []))

            results = self.index.search_multi(
                queries=group_queries,
                top_k=self.filtered_top_k,
                deduplicate=True,
            )
            group_results[group_name] = results

        return group_results

    def _merge_and_sort(
        self,
        global_chunks: List[Tuple[PageChunk, float]],
        filtered_chunks: Dict[str, List[Tuple[PageChunk, float]]],
    ) -> List[PageChunk]:
        """
        Merge global + all filtered chunks, deduplicate by chunk_index,
        and sort by page number for coherent document-order context.
        """
        seen_indices: set[int] = set()
        merged: List[PageChunk] = []

        # Add global chunks first (highest priority for coverage)
        for chunk, _ in global_chunks:
            if chunk.chunk_index not in seen_indices:
                seen_indices.add(chunk.chunk_index)
                merged.append(chunk)

        # Add filtered chunks (targeted signal)
        for group_results in filtered_chunks.values():
            for chunk, _ in group_results:
                if chunk.chunk_index not in seen_indices:
                    seen_indices.add(chunk.chunk_index)
                    merged.append(chunk)

        # Sort by page number → coherent reading order for LLM
        merged.sort(key=lambda c: (c.start_page, c.chunk_index))

        return merged

    def get_chunks_for_group(
        self,
        joint_context: JointContext,
        group_name: str,
        max_chunks: int = 8,
    ) -> Tuple[List[str], List[int]]:
        """
        Get the best chunks for a specific clause group from a built JointContext.
        Combines filtered (targeted) + global (broad) chunks for the group.
        """
        targeted = joint_context.filtered_chunks.get(group_name, [])
        targeted_indices = {c.chunk_index for c, _ in targeted}

        # Top targeted first
        chunks_out: List[PageChunk] = [c for c, _ in targeted]

        # Supplement with global chunks not already included
        remaining = max_chunks - len(chunks_out)
        if remaining > 0:
            for c in joint_context.all_chunks:
                if c.chunk_index not in targeted_indices and remaining > 0:
                    chunks_out.append(c)
                    remaining -= 1

        # Sort by page
        chunks_out.sort(key=lambda c: c.start_page)
        chunks_out = chunks_out[:max_chunks]

        texts = [c.text for c in chunks_out]
        pages = [c.start_page for c in chunks_out]
        return texts, pages
