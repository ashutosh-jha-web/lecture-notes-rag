import asyncio
from unittest.mock import AsyncMock

import pytest

from app.ragas_evaluator import run_metric_with_retries


@pytest.mark.parametrize(
    "error_message",
    [
        "429 RESOURCE_EXHAUSTED: rate limit exceeded",
        "503 Service Unavailable",
        "Connection error",
        "Request timeout",
    ],
)
def test_retries_transient_errors_then_succeeds(
    monkeypatch, error_message
):
    async def run():
        metric = AsyncMock()
        metric.ascore.side_effect = [
            Exception(error_message),
            0.85,
        ]

        async def no_sleep(_delay):
            pass

        monkeypatch.setattr(
            "app.ragas_evaluator.asyncio.sleep", no_sleep
        )

        result = await run_metric_with_retries(
            metric, {"question": "Test question"}
        )

        assert result == 0.85
        assert metric.ascore.await_count == 2

    asyncio.run(run())


def test_does_not_retry_permanent_errors(monkeypatch):
    async def run():
        metric = AsyncMock()
        metric.ascore.side_effect = ValueError("Invalid input")

        async def no_sleep(_delay):
            pytest.fail("Sleep should not happen for permanent errors")

        monkeypatch.setattr(
            "app.ragas_evaluator.asyncio.sleep", no_sleep
        )

        with pytest.raises(ValueError, match="Invalid input"):
            await run_metric_with_retries(
                metric, {"question": "Test question"}
            )

        assert metric.ascore.await_count == 1

    asyncio.run(run())


def test_raises_after_maximum_retry_attempts(monkeypatch):
    async def run():
        metric = AsyncMock()
        metric.ascore.side_effect = Exception("429 rate limit exceeded")

        async def no_sleep(_delay):
            pass

        monkeypatch.setattr(
            "app.ragas_evaluator.asyncio.sleep", no_sleep
        )

        with pytest.raises(Exception, match="429 rate limit"):
            await run_metric_with_retries(
                metric,
                {"question": "Test question"},
                max_attempts=3,
            )

        assert metric.ascore.await_count == 3

    asyncio.run(run())


def test_succeeds_without_retry():
    async def run():
        metric = AsyncMock()
        metric.ascore.return_value = 0.95

        result = await run_metric_with_retries(
            metric, {"question": "Test question"}
        )

        assert result == 0.95
        assert metric.ascore.await_count == 1

    asyncio.run(run())


from app.ragas_evaluator import normalize_score


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.85, 0.85),
        (0, 0.0),
        (1, 1.0),
        (-0.2, 0.0),
        (1.5, 1.0),
        (float("nan"), None),
        (float("inf"), None),
        (None, None),
        ("invalid", None),
    ],
)
def test_normalize_score(value, expected):
    assert normalize_score(value) == expected
