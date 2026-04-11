"""
metrics.py — Batch metrics computation for contract extraction results.
Computes extraction success rate, clause coverage, and field coverage.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

CLAUSE_NAMES = ["governing_law", "audit_rights", "non_compete", "non_solicitation"]
FIELD_NAMES = ["jurisdiction", "payment_terms", "notice_period", "liability_cap"]


@dataclass
class BatchMetrics:
    total: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0
    avg_clause_coverage: float = 0.0
    avg_field_coverage: float = 0.0
    clause_coverage: Dict[str, float] = field(default_factory=dict)
    field_coverage: Dict[str, float] = field(default_factory=dict)
    avg_confidence: Dict[str, float] = field(default_factory=dict)
    validation_rates: Dict[str, float] = field(default_factory=dict)
    contract_type_distribution: Dict[str, int] = field(default_factory=dict)

    def compute(self, output_dir: Path) -> "BatchMetrics":
        """Compute metrics from all JSON output files in output_dir."""
        json_files = sorted(output_dir.glob("*_extracted.json"))

        if not json_files:
            logger.warning(f"No extracted JSON files found in {output_dir}")
            return self

        records = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(data)
            except Exception as e:
                logger.error(f"Failed to read {jf}: {e}")

        self.total = len(records)

        # Separate successful and failed
        successful = [r for r in records if r.get("_metadata", {}).get("status") == "success"]
        failed = [r for r in records if r.get("_metadata", {}).get("status") != "success"]

        self.successful = len(successful)
        self.failed = len(failed)
        self.success_rate = self.successful / self.total if self.total > 0 else 0.0

        if not successful:
            return self

        # ── Clause Coverage ──────────────────────────────────────────────────
        clause_found_counts = {name: 0 for name in CLAUSE_NAMES}
        clause_confidences = {name: [] for name in CLAUSE_NAMES}
        clause_validated_counts = {name: 0 for name in CLAUSE_NAMES}

        for record in successful:
            clauses = record.get("clauses", {})
            for name in CLAUSE_NAMES:
                clause = clauses.get(name, {})
                if clause.get("value") is not None:
                    clause_found_counts[name] += 1
                    conf = clause.get("confidence", 0.0)
                    if conf:
                        clause_confidences[name].append(conf)
                    if clause.get("validated"):
                        clause_validated_counts[name] += 1

        self.clause_coverage = {
            name: clause_found_counts[name] / self.successful
            for name in CLAUSE_NAMES
        }
        self.avg_clause_coverage = (
            sum(self.clause_coverage.values()) / len(CLAUSE_NAMES)
        )
        self.avg_confidence = {
            name: (sum(confs) / len(confs)) if confs else 0.0
            for name, confs in clause_confidences.items()
        }
        self.validation_rates = {
            name: clause_validated_counts[name] / max(clause_found_counts[name], 1)
            for name in CLAUSE_NAMES
        }

        # ── Field Coverage ───────────────────────────────────────────────────
        field_found_counts = {name: 0 for name in FIELD_NAMES}
        for record in successful:
            fields = record.get("fields", {})
            for name in FIELD_NAMES:
                if fields.get(name) is not None:
                    field_found_counts[name] += 1

        self.field_coverage = {
            name: field_found_counts[name] / self.successful
            for name in FIELD_NAMES
        }
        self.avg_field_coverage = (
            sum(self.field_coverage.values()) / len(FIELD_NAMES)
        )

        # ── Contract Type Distribution ───────────────────────────────────────
        type_dist: Dict[str, int] = {}
        for record in successful:
            ct = record.get("contract_type", {}).get("value")
            if ct:
                type_dist[ct] = type_dist.get(ct, 0) + 1
        self.contract_type_distribution = type_dist

        logger.info(
            f"Metrics computed: {self.successful}/{self.total} successful, "
            f"avg clause coverage={self.avg_clause_coverage:.1%}"
        )

        return self

    def save(self, path: Path) -> None:
        """Save metrics as JSON."""
        data = {
            "summary": {
                "total": self.total,
                "successful": self.successful,
                "failed": self.failed,
                "success_rate": round(self.success_rate, 4),
                "avg_clause_coverage": round(self.avg_clause_coverage, 4),
                "avg_field_coverage": round(self.avg_field_coverage, 4),
            },
            "clause_coverage": {k: round(v, 4) for k, v in self.clause_coverage.items()},
            "field_coverage": {k: round(v, 4) for k, v in self.field_coverage.items()},
            "avg_confidence_per_clause": {k: round(v, 4) for k, v in self.avg_confidence.items()},
            "validation_rates": {k: round(v, 4) for k, v in self.validation_rates.items()},
            "contract_type_distribution": self.contract_type_distribution,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Metrics saved to {path}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame for reporting."""
        rows = []
        for name in CLAUSE_NAMES:
            rows.append({
                "type": "clause",
                "name": name,
                "coverage": self.clause_coverage.get(name, 0.0),
                "avg_confidence": self.avg_confidence.get(name, 0.0),
                "validation_rate": self.validation_rates.get(name, 0.0),
            })
        for name in FIELD_NAMES:
            rows.append({
                "type": "field",
                "name": name,
                "coverage": self.field_coverage.get(name, 0.0),
                "avg_confidence": None,
                "validation_rate": None,
            })
        return pd.DataFrame(rows)
