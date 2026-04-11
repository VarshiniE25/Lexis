"""
config.py — Centralized configuration for the Contract Intelligence Engine.
All tuneable parameters live here. Override via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ─── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = 0.0          # Deterministic extraction
LLM_MAX_TOKENS: int = 1024
LLM_MAX_RETRIES: int = 3
LLM_RETRY_WAIT: float = 2.0           # seconds between retries

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384               # all-MiniLM-L6-v2 output dimension

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_MIN_WORDS: int = 500
CHUNK_MAX_WORDS: int = 800
CHUNK_OVERLAP_WORDS: int = 100

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))

# ─── Caching ──────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))   # seconds
CACHE_ENABLED: bool = True

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = LOG_DIR / "contract_engine.log"

# ─── Contract Types ───────────────────────────────────────────────────────────
CONTRACT_TYPES = [
    "Service Agreement",
    "Lease Agreement",
    "IP Agreement",
    "Supply Agreement",
]

# ─── Clause Definitions ───────────────────────────────────────────────────────
CLAUSE_NAMES = [
    "governing_law",
    "audit_rights",
    "non_compete",
    "non_solicitation",
]

# Human-readable labels for prompts
CLAUSE_LABELS = {
    "governing_law": "Governing Law",
    "audit_rights": "Audit Rights",
    "non_compete": "Non-Compete",
    "non_solicitation": "Non-Solicitation of Employees",
}

# ─── Structured Fields ────────────────────────────────────────────────────────
FIELD_NAMES = [
    "jurisdiction",
    "payment_terms",
    "notice_period",
    "liability_cap",
]

FIELD_LABELS = {
    "jurisdiction": "Jurisdiction",
    "payment_terms": "Payment Terms",
    "notice_period": "Notice Period",
    "liability_cap": "Liability Cap",
}
