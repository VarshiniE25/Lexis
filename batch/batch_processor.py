"""
batch_processor.py — Batch contract processing for 500+ contracts.

Usage:
    python -m batch.batch_processor --input_dir ./contracts --output_dir ./results
    python -m batch.batch_processor --input_dir ./contracts --output_dir ./results --workers 4
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import ContractExtractor
from src.logger import get_logger
from batch.metrics import BatchMetrics

logger = get_logger(__name__)
console = Console()


class BatchProcessor:
    """
    Processes multiple contract PDFs in parallel using a thread pool.
    Each thread runs its own extractor pipeline.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        workers: int = 2,
        resume: bool = True,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.resume = resume

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracker
        self.metrics = BatchMetrics()

    def find_pdfs(self) -> List[Path]:
        """Find all PDF files in input directory."""
        pdfs = sorted(self.input_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {self.input_dir}")
        return pdfs

    def get_output_path(self, pdf_path: Path) -> Path:
        """Determine output JSON path for a PDF."""
        return self.output_dir / f"{pdf_path.stem}_extracted.json"

    def already_processed(self, pdf_path: Path) -> bool:
        """Check if a PDF was already processed (for resume support)."""
        return self.get_output_path(pdf_path).exists()

    def process_single(self, pdf_path: Path) -> dict:
        """
        Process a single PDF. Runs in a thread.
        Returns a result dict with status + metadata.
        """
        start_time = time.perf_counter()
        output_path = self.get_output_path(pdf_path)

        try:
            extractor = ContractExtractor()  # Fresh extractor per thread
            result = extractor.process_file(pdf_path)
            output = result.to_output_dict()

            # Add metadata
            output["_metadata"] = {
                "source_file": str(pdf_path.name),
                "processing_time_s": round(time.perf_counter() - start_time, 2),
                "status": "success",
            }

            # Write output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ {pdf_path.name} → {output_path.name}")
            return {"file": pdf_path.name, "status": "success", "output": output}

        except Exception as e:
            logger.error(f"❌ {pdf_path.name}: {e}")
            error_output = {
                "_metadata": {
                    "source_file": str(pdf_path.name),
                    "processing_time_s": round(time.perf_counter() - start_time, 2),
                    "status": "error",
                    "error": str(e),
                }
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(error_output, f, indent=2)

            return {"file": pdf_path.name, "status": "error", "error": str(e)}

    def run(self) -> BatchMetrics:
        """Run batch processing on all PDFs."""
        pdfs = self.find_pdfs()

        if not pdfs:
            console.print("[yellow]No PDF files found in input directory.[/yellow]")
            return self.metrics

        # Filter already processed if resume=True
        if self.resume:
            pending = [p for p in pdfs if not self.already_processed(p)]
            skipped = len(pdfs) - len(pending)
            if skipped > 0:
                console.print(f"[cyan]Resuming: {skipped} already processed, {len(pending)} remaining[/cyan]")
        else:
            pending = pdfs

        if not pending:
            console.print("[green]All contracts already processed![/green]")
            return self._compute_metrics(pdfs)

        console.print(f"[bold]Processing {len(pending)} contracts with {self.workers} workers...[/bold]")

        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_pdf = {
                executor.submit(self.process_single, pdf): pdf
                for pdf in pending
            }

            with tqdm(total=len(pending), desc="Processing", unit="contract") as pbar:
                for future in as_completed(future_to_pdf):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        "success": sum(1 for r in results if r["status"] == "success"),
                        "error": sum(1 for r in results if r["status"] == "error"),
                    })

        # Compute and display metrics
        metrics = self._compute_metrics(pdfs)
        self._display_metrics(metrics)

        return metrics

    def _compute_metrics(self, all_pdfs: List[Path]) -> "BatchMetrics":
        """Compute batch metrics from output files."""
        return self.metrics.compute(self.output_dir)

    def _display_metrics(self, metrics: "BatchMetrics") -> None:
        """Display metrics in a rich table."""
        table = Table(title="Batch Processing Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Contracts", str(metrics.total))
        table.add_row("Successful", f"[green]{metrics.successful}[/green]")
        table.add_row("Failed", f"[red]{metrics.failed}[/red]")
        table.add_row("Success Rate", f"{metrics.success_rate:.1%}")
        table.add_row("Avg Clause Coverage", f"{metrics.avg_clause_coverage:.1%}")

        for clause, coverage in metrics.clause_coverage.items():
            table.add_row(f"  ↳ {clause}", f"{coverage:.1%}")

        console.print(table)

        # Save metrics JSON
        metrics_path = self.output_dir / "batch_metrics.json"
        metrics.save(metrics_path)
        console.print(f"\n[bold green]Metrics saved to {metrics_path}[/bold green]")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch process contract PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF contracts")
    parser.add_argument("--output_dir", required=True, help="Directory for JSON outputs")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess already-done files")

    args = parser.parse_args()

    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=not args.no_resume,
    )

    processor.run()


if __name__ == "__main__":
    main()
