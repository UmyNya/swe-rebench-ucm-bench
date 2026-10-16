"""Compare successful client runs, reporting request-setting differences explicitly."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .metrics import percentile


def load_result(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "client-result.json"
    result = json.loads(path.read_text())
    summary = result["summary"]
    if result["status"] != "completed" or summary["errors"] != 0:
        raise ValueError(f"comparison requires a successful run: {path}")
    if summary["successful_requests"] != summary["requests"] or summary["requests"] < 1:
        raise ValueError(f"request accounting is incomplete: {path}")
    if not result["execution_order"]["client_send_order_matches_manifest"]:
        raise ValueError(f"request order check failed: {path}")
    if len(result["records"]) != summary["requests"]:
        raise ValueError(f"missing request records: {path}")
    if any(record.get("error") or record["output_tokens"] != result["max_output_tokens"] for record in result["records"]):
        raise ValueError(f"request output accounting failed: {path}")
    if summary["observed_max_inflight"] > summary["client_concurrency"]:
        raise ValueError(f"concurrency bound exceeded: {path}")
    return result


def aggregate(results: list[dict]) -> dict:
    """Use total tokens / total time and pool request latencies across rounds."""
    summaries = [result["summary"] for result in results]
    wall = sum(summary["wall_seconds"] for summary in summaries)
    if wall <= 0:
        raise ValueError("measured wall time must be positive")
    ttfts = [record["ttft_ms"] for result in results for record in result["records"]]
    return {
        "rounds": len(results),
        "successful_requests": sum(summary["successful_requests"] for summary in summaries),
        "wall_seconds": wall,
        "input_tps": sum(summary["prompt_tokens"] for summary in summaries) / wall,
        "output_tps": sum(summary["output_tokens"] for summary in summaries) / wall,
        "ttft_mean_ms": statistics.fmean(ttfts),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
    }


def request_identity(result: dict) -> dict:
    return {
        "order_sha256": result["order_sha256"],
        "input_fingerprint": result.get("prompt_snapshot", {}).get("input_fingerprint"),
        "requests": result["summary"]["requests"],
        "client_concurrency": result["summary"]["client_concurrency"],
        "max_output_tokens": result["max_output_tokens"],
        "max_model_len": result["max_model_len"],
        "model": result["model"],
        "seed": result["seed"],
        "protocol": result["protocol"]["name"],
    }


def compare_runs(off_paths: list[Path], on_paths: list[Path], *, allow_parameter_differences: bool = False) -> dict:
    if not off_paths or len(off_paths) != len(on_paths):
        raise ValueError("provide the same nonzero number of OFF and ON runs")
    off = [load_result(path) for path in off_paths]
    on = [load_result(path) for path in on_paths]
    reference = request_identity(off[0])
    differences = []
    for arm, results in [("OFF", off), ("ON", on)]:
        for index, result in enumerate(results):
            identity = request_identity(result)
            changed = {key: {"reference": value, "actual": identity[key]} for key, value in reference.items() if identity[key] != value}
            if not identity["input_fingerprint"]:
                changed["input_fingerprint"] = {"error": "exact input fingerprint is missing"}
            if changed:
                differences.append({"arm": arm, "round": index + 1, "differences": changed})
    if differences and not allow_parameter_differences:
        raise ValueError("request settings or exact inputs differ; use --allow-parameter-differences only for an intentional parameter comparison: " + json.dumps(differences))
    metrics = {"OFF": aggregate(off), "ON": aggregate(on)}
    changes = {
        metric: (metrics["ON"][metric] / metrics["OFF"][metric] - 1) * 100
        for metric in ["input_tps", "output_tps", "ttft_mean_ms", "ttft_p95_ms", "ttft_p99_ms", "wall_seconds"]
    }
    paired = {}
    for metric in ["input_tps", "ttft_mean_ms"]:
        values = [
            (aggregate([on_run])[metric] / aggregate([off_run])[metric] - 1) * 100
            for off_run, on_run in zip(off, on, strict=True)
        ]
        paired[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "private_artifact": True,
        "request_settings_match": not differences,
        "parameter_differences": differences,
        "metrics": metrics,
        "on_vs_off_percent": changes,
        "paired_change_statistics": paired,
        "runs": {"OFF": [str(path.resolve()) for path in off_paths], "ON": [str(path.resolve()) for path in on_paths]},
        "limitations": [
            "OFF/ON are user-provided labels; the client does not verify or modify the external service configuration.",
            "Start each formal arm with an independent empty KV cache and the documented server configuration.",
            "One pair cannot establish multi-round stability; sample_sd is null for one pair.",
        ],
    }
