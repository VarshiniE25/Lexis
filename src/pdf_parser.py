"""
pdf_parser.py — PDF text extraction with page numbers using PyMuPDF (fitz).
Handles multi-column, poor formatting, and large PDFs efficiently.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    page_number: int          # 1-indexed
    text: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class ParsedDocument:
    pages: List[PageText]
    total_pages: int
    total_words: int
    file_name: str

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def get_text_with_pages(self) -> List[Tuple[int, str]]:
        """Returns list of (page_number, text) tuples."""
        return [(p.page_number, p.text) for p in self.pages]


class PDFParser:
    """
    Extracts structured text from PDF files using PyMuPDF.
    Handles:
    - Multi-column layouts (via blocks sorting)
    - Headers/footers (basic heuristic removal)
    - Large PDFs (streaming page-by-page)
    - Poor OCR quality (whitespace normalization)
    """

    # Minimum characters per page to consider it non-empty
    MIN_PAGE_CHARS = 50

    def __init__(self):
        self._whitespace_re = re.compile(r"\s+")
        self._page_num_re = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file and return structured text by page.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with per-page text.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        pages: List[PageText] = []

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            logger.info(f"Total pages: {total_pages}")

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed

                text = self._extract_page_text(page)

                # FIX: Always append the page to keep the list index aligned with the PDF
                # Even if text is short, we keep the page_number entry.
                if len(text) < self.MIN_PAGE_CHARS:
                    logger.debug(f"Page {page_number} is near-empty. Storing as empty string.")
                    text = "" 

                pages.append(PageText(
                    page_number=page_number,
                    text=text,
                ))

        total_words = sum(p.word_count for p in pages)
        logger.info(f"Parsed {len(pages)} pages, {total_words} total words")

        return ParsedDocument(
            pages=pages,
            total_pages=total_pages,
            total_words=total_words,
            file_name=pdf_path.name,
        )

    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract clean text from a single page.
        Uses block-based extraction for better multi-column handling.
        """
        # Use "blocks" for spatial ordering (handles 2-col layouts)
        blocks = page.get_text("blocks", sort=True)

        text_parts = []
        for block in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            if block[6] == 0:  # text block (not image)
                block_text = block[4].strip()
                if block_text:
                    text_parts.append(block_text)

        raw_text = "\n".join(text_parts)
        return self._clean_text(raw_text)

    def _clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove common PDF artifacts.
        """
        # Remove standalone page numbers
        text = self._page_num_re.sub("", text)

        # Normalize Unicode hyphens/dashes
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Normalize smart quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')

        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize multiple blank lines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize multiple spaces (but preserve newlines)
        lines = text.split("\n")
        lines = [self._whitespace_re.sub(" ", line).strip() for line in lines]
        text = "\n".join(line for line in lines if line)

        return text.strip()

    def parse_from_bytes(self, pdf_bytes: bytes, file_name: str = "upload.pdf") -> ParsedDocument:
        """
        Parse a PDF from raw bytes (e.g., Streamlit upload).
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            result = self.parse(tmp_path)
            result.file_name = file_name
            return result
        finally:
            os.unlink(tmp_path)
