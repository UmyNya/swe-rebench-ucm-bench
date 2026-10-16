"""Latency distributions and closed-loop throughput accounting."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Sequence, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dispatch import ClosedLoopReport

def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(percentile(values, 0.50), 6),
        "p95": round(percentile(values, 0.95), 6),
        "p99": round(percentile(values, 0.99), 6),
        "maximum": round(max(values), 6),
    }


def summarize(report: ClosedLoopReport) -> dict[str, Any]:
    """Summarize one closed-loop run without inventing offered-rate metrics."""
    successful = [
        record
        for record in report.records
        if record.error is None
        and record.actual_submit_s is not None
        and record.completion_s is not None
    ]
    ttfts = [
        float(record.response["ttft_ms"])
        for record in successful
        if record.response.get("ttft_ms") is not None
    ]
    submitted_records = [
        record for record in report.records if record.submission_sequence is not None
    ]
    submit_order = [
        record.ordinal
        for record in sorted(
            submitted_records,
            key=lambda item: int(item.submission_sequence or 0),
        )
    ]
    submit_hash = hashlib.sha256(
        json.dumps(submit_order, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "requests": len(report.records),
        "successful_requests": len(successful),
        "errors": len(report.records) - len(successful),
        "wall_seconds": round(report.wall_seconds, 6),
        "completion_throughput_rps": (
            len(successful) / report.wall_seconds
            if report.wall_seconds > 0
            else math.nan
        ),
        "input_tps": (
            sum(int(record.response.get("prompt_tokens", 0)) for record in successful)
            / report.wall_seconds if report.wall_seconds > 0 else math.nan
        ),
        "output_tps": (
            sum(int(record.response.get("output_tokens", 0)) for record in successful)
            / report.wall_seconds if report.wall_seconds > 0 else math.nan
        ),
        "mean_e2e_ms": statistics.fmean(
            float(record.total_ms) for record in successful if record.total_ms is not None
        ) if successful else math.nan,
        "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else math.nan,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "prompt_tokens": sum(
            int(record.response.get("prompt_tokens", 0)) for record in successful
        ),
        "output_tokens": sum(
            int(record.response.get("output_tokens", 0)) for record in successful
        ),
        "client_concurrency": report.client_concurrency,
        "observed_max_inflight": report.observed_max_inflight,
        "dependency_wait_ms": distribution(
            [float(record.dependency_wait_ms or 0.0) for record in report.records]
        ),
        "client_admission_wait_ms": distribution(
            [
                float(record.client_admission_wait_ms or 0.0)
                for record in report.records
            ]
        ),
        "actual_submit_order_sha256": submit_hash,
        "arrival_schedule": None,
        "offered_rate_rps": None,
    }
