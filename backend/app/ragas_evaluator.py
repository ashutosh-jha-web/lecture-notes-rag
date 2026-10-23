
import asyncio
import logging
import math
import time
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)
from ragas.metrics.collections.faithfulness.metric import (
    NLIStatementInput,
    NLIStatementOutput,
)

from app.config import CHAT_MODEL, GOOGLE_API_KEY, EMBEDDING_MODEL


logger = logging.getLogger(__name__)

EVALUATION_CONCURRENCY = 3
FAITHFULNESS_BATCH_SIZE = 5
EVALUATION_TIMEOUT_SECONDS = 90.0

_evaluation_semaphore = asyncio.Semaphore(EVALUATION_CONCURRENCY)


def normalize_score(value: Any) -> float | None:
    """Extract a finite numeric score and constrain it to [0, 1]."""
    if value is None:
        return None

    candidates = [value]

    if hasattr(value, "value"):
        candidates.insert(0, value.value)

    for candidate in candidates:
        try:
            score = float(candidate)
        except (AttributeError, TypeError, ValueError):
            continue

        if math.isfinite(score):
            return max(0.0, min(1.0, score))

    return None


@lru_cache(maxsize=1)
def get_evaluation_embeddings() -> HuggingFaceEmbeddings:
    """Load and reuse the local evaluation embedding model."""
    logger.info(
        "[RAGAS] Loading evaluation embedding model: %s",
        EMBEDDING_MODEL,
    )
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


class BatchedFaithfulness(Faithfulness):
    """Evaluate faithfulness verdicts in small batches."""

    async def _create_verdicts(
        self,
        statements: list[str],
        context: str,
    ) -> NLIStatementOutput:
        all_verdicts = []

        for start in range(0, len(statements), FAITHFULNESS_BATCH_SIZE):
            batch = statements[
                start : start + FAITHFULNESS_BATCH_SIZE
            ]

            logger.info(
                "[RAGAS] Faithfulness batch %d: statements %d-%d of %d",
                start // FAITHFULNESS_BATCH_SIZE + 1,
                start + 1,
                start + len(batch),
                len(statements),
            )

            input_data = NLIStatementInput(
                context=context,
                statements=batch,
            )
            prompt_str = self.nli_statement_prompt.to_string(input_data)

            result = await self.llm.agenerate(
                prompt_str,
                NLIStatementOutput,
            )

            batch_verdicts = result.statements

            if len(batch_verdicts) != len(batch):
                raise ValueError(
                    "Faithfulness returned an unexpected number of "
                    f"verdicts: expected {len(batch)}, "
                    f"got {len(batch_verdicts)}."
                )

            all_verdicts.extend(batch_verdicts)

        return NLIStatementOutput(statements=all_verdicts)


def log_failed_completion(name: str, exc: Exception) -> None:
    """Log structured-output details when the exception exposes them."""
    completion = getattr(exc, "last_completion", None)

    if completion is None:
        logger.debug(
            "[RAGAS] %s exception has no last_completion attribute.",
            name,
        )
        return

    try:
        if hasattr(completion, "model_dump_json"):
            details = completion.model_dump_json(exclude_none=True)
        else:
            details = repr(completion)

        logger.error(
            "[RAGAS] %s failed completion details: %s",
            name,
            details[:4000],
        )
    except Exception:
        logger.exception(
            "[RAGAS] Could not log failed completion for %s.",
            name,
        )


async def aevaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    include_errors: bool = False,
) -> dict[str, float | None] | tuple[
    dict[str, float | None], dict[str, str]
]:
    """Evaluate faithfulness, answer relevancy, and context precision."""

    evaluation_start = time.perf_counter()

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    if not answer or not answer.strip():
        raise ValueError("The answer cannot be empty.")

    contexts = [
        context.strip()
        for context in contexts
        if isinstance(context, str) and context.strip()
    ]

    if not contexts:
        raise ValueError("At least one non-empty context is required.")

    results: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
    }
    errors: dict[str, str] = {}

    client = AsyncOpenAI(
        api_key=GOOGLE_API_KEY,
        base_url=(
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        timeout=EVALUATION_TIMEOUT_SECONDS,
        max_retries=0,
    )

    try:
        model_name = (
            CHAT_MODEL.removeprefix("models/").removeprefix("gemini/")
        )

        evaluator_llm = llm_factory(
            model_name,
            client=client,
        )

        embedding_start = time.perf_counter()
        evaluator_embeddings = get_evaluation_embeddings()

        logger.info(
            "[RAGAS] Evaluation embeddings ready in %.2fs",
            time.perf_counter() - embedding_start,
        )

        metrics = [
            (
                "faithfulness",
                BatchedFaithfulness(llm=evaluator_llm),
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                },
            ),
            (
                "answer_relevancy",
                AnswerRelevancy(
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                ),
                {
                    "user_input": question,
                    "response": answer,
                },
            ),
            (
                "context_precision",
                ContextPrecisionWithoutReference(
                    llm=evaluator_llm,
                ),
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                },
            ),
        ]

        async def evaluate_one(
            name: str,
            metric: Any,
            kwargs: dict[str, Any],
        ) -> tuple[str, float | None, str | None]:
            metric_start = time.perf_counter()
            logger.info("[RAGAS] Starting %s", name)

            try:
                score = await metric.ascore(**kwargs)
                normalized = normalize_score(score)
                elapsed = time.perf_counter() - metric_start

                if normalized is None:
                    error_message = (
                        "RAGAS returned no valid numeric score."
                    )
                    logger.error(
                        "[RAGAS] %s returned no score after %.2fs",
                        name,
                        elapsed,
                    )
                    return name, None, error_message

                logger.info(
                    "[RAGAS] %s = %.4f (%.2fs)",
                    name,
                    normalized,
                    elapsed,
                )

                return name, normalized, None

            except Exception as exc:
                elapsed = time.perf_counter() - metric_start
                error_message = f"{type(exc).__name__}: {exc}"

                logger.error(
                    "[RAGAS] %s failed after %.2fs: %s",
                    name,
                    elapsed,
                    error_message,
                    exc_info=True,
                )

                log_failed_completion(name, exc)

                return name, None, error_message

        async with _evaluation_semaphore:
            metric_results = await asyncio.gather(
                *(
                    evaluate_one(name, metric, kwargs)
                    for name, metric, kwargs in metrics
                )
            )

        for name, score, error in metric_results:
            results[name] = score

            if error:
                errors[name] = error

        total_elapsed = time.perf_counter() - evaluation_start

        if errors:
            logger.warning(
                "[RAGAS] Partial evaluation after %.2fs. "
                "Failed metrics: %s",
                total_elapsed,
                errors,
            )
        else:
            logger.info(
                "[RAGAS] All three metrics completed in %.2fs",
                total_elapsed,
            )

        if include_errors:
            return results, errors

        return results

    finally:
        await client.close()


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float | None]:
    """Synchronous wrapper for scripts outside an active event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            aevaluate_rag_response(
                question=question,
                answer=answer,
                contexts=contexts,
            )
        )

    raise RuntimeError(
        "An event loop is already running. "
        "Use `await aevaluate_rag_response(...)` instead."
    )
