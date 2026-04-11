"""
validator.py — Second-pass LLM validation for extracted clause values.
Verifies that extracted text appears verbatim and the value is accurate.
"""

from __future__ import annotations
from typing import Optional

from .llm_client import llm_call
from .prompts import validation_prompt
from .models import ClauseResult, ValidationResponse
from .logger import get_logger

logger = get_logger(__name__)


class ClauseValidator:
    """
    Validates extracted clause values via a second LLM pass.
    """

    async def validate(
        self,
        clause_name: str,
        clause_result: ClauseResult,
        original_context: str,
    ) -> ClauseResult:
        """
        Validate a clause extraction.

        Args:
            clause_name: Human-readable clause name.
            clause_result: The extracted clause.
            original_context: The raw context chunks used for extraction.

        Returns:
            Updated ClauseResult with validated flag set.
        """
        # Skip validation if nothing was extracted
        if clause_result.value is None or clause_result.exact_text is None:
            logger.debug(f"Skipping validation for null clause: {clause_name}")
            clause_result.validated = False
            return clause_result

        prompt = validation_prompt(
            clause_name=clause_name,
            extracted_value=clause_result.value,
            extracted_exact_text=clause_result.exact_text,
            original_context=original_context,
        )

        response = await llm_call(prompt)

        if response is None:
            logger.warning(f"Validation LLM call failed for {clause_name}")
            # Default to false if validation fails
            clause_result.validated = False
            return clause_result

        try:
            validation = ValidationResponse(
                validated=bool(response.get("validated", False)),
                corrected_value=response.get("corrected_value"),
                reasoning=response.get("reasoning", ""),
            )

            clause_result.validated = validation.validated
            logger.info(
                f"[{clause_name}] Validation: {validation.validated} — "
                f"{validation.reasoning}"
            )

            # Apply correction if validator provides one
            if validation.corrected_value and not validation.validated:
                logger.info(f"[{clause_name}] Applying correction: {validation.corrected_value[:80]}")
                clause_result.value = validation.corrected_value

        except Exception as e:
            logger.error(f"Validation parsing error for {clause_name}: {e}")
            clause_result.validated = False

        return clause_result

    async def validate_all_clauses(
        self,
        clauses: dict,
        contexts: dict,
        clause_labels: dict,
    ) -> dict:
        """
        Validate all clauses in parallel.

        Args:
            clauses: Dict of clause_name → ClauseResult
            contexts: Dict of clause_name → (chunk_texts, page_numbers)
            clause_labels: Dict of clause_name → human readable label

        Returns:
            Updated clauses dict with validation flags set.
        """
        import asyncio

        async def _validate_one(clause_name: str, result: ClauseResult) -> tuple:
            chunk_texts, _ = contexts.get(clause_name, ([], []))
            context_str = "\n\n".join(chunk_texts)
            label = clause_labels.get(clause_name, clause_name)
            updated = await self.validate(label, result, context_str)
            return clause_name, updated

        tasks = [
            _validate_one(name, result)
            for name, result in clauses.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        validated_clauses = dict(clauses)  # copy
        for item in results:
            if isinstance(item, Exception):
                logger.error(f"Validation task failed: {item}")
            else:
                name, updated = item
                validated_clauses[name] = updated

        return validated_clauses
