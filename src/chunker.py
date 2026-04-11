"""
Clause-aware chunking for legal documents.
Splits document based on headings/clauses instead of word count.
"""

import re
from typing import List
from .models import PageChunk
from .pdf_parser import ParsedDocument
from .logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Clause-aware chunker:
    - Detects clause headings
    - Groups text under each clause
    - Preserves page numbers
    """

    CLAUSE_PATTERN = re.compile(
        r"(?i)(\n|^)\s*(\d+(\.\d+)*\s+)?(governing law|termination|liability|confidentiality|payment|notice|audit|non[- ]?compete|non[- ]?solicitation)[^\n]*"
    )

    def chunk(self, document: ParsedDocument) -> List[PageChunk]:
        chunks: List[PageChunk] = []
        chunk_index = 0

        current_text = []
        current_pages = []

        def flush_chunk():
            nonlocal chunk_index
            if current_text:
                text = " ".join(current_text).strip()
                chunk = PageChunk(
                    text=text,
                    start_page=min(current_pages),
                    end_page=max(current_pages),
                    chunk_index=chunk_index,
                    word_count=len(text.split()),
                )
                chunks.append(chunk)
                chunk_index += 1

        for page in document.pages:
            lines = page.text.split("\n")

            for line in lines:
                # Check if this line is a clause heading
                if self.CLAUSE_PATTERN.search(line):
                    # Save previous chunk
                    flush_chunk()
                    current_text = [line]
                    current_pages = [page.page_number]
                else:
                    current_text.append(line)
                    current_pages.append(page.page_number)

        # Flush last chunk
        flush_chunk()

        logger.info(f"Clause-aware chunking created {len(chunks)} chunks")
        return chunks