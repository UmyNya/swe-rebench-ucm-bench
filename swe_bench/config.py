"""Resolve request settings; server installation and lifecycle are external."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .dispatch import validate_concurrency
from .workload import validate_closed_loop_workload

ROOT = Path(__file__).resolve().parents[1]


def make_args(
    workload: Path,
    output: Path,
    *,
    base_url: str = "http://127.0.0.1:18147",
    model: str = "glm52-mixed",
    concurrency: int = 8,
    output_tokens: int = 1,
    max_model_len: int = 131072,
    request_timeout: float = 3600,
    request_limit: int | None = None,
    api_key_env: str = "SWE_BENCH_API_KEY",
    label: str = "",
) -> argparse.Namespace:
    """Build the replay inputs explicitly, deriving seed and W from the workload."""
    validate_concurrency(concurrency)
    if output_tokens < 1 or max_model_len < 1:
        raise ValueError("output_tokens and max_model_len must be positive")
    if output_tokens >= max_model_len:
        raise ValueError("output_tokens must leave room for input tokens")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("base_url must be an HTTP(S) service root")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("base_url cannot contain credentials, a query or a fragment; use api_key_env")
    if endpoint.path.rstrip("/").endswith("/v1"):
        raise ValueError("base_url is the service root, without the /v1 suffix")
    if not model.strip():
        raise ValueError("model must be the server's registered model name")

    workload = workload.resolve()
    manifest = json.loads((workload / "manifest.json").read_text())
    interleaving = manifest["interleaving"]
    events, _ = validate_closed_loop_workload(
        workload / "order.jsonl", workload / "manifest.json", workload / "profiles.jsonl",
        seed=int(interleaving["seed"]),
        session_pool=int(interleaving["active_sessions"]),
        prompt_salt_prefix=interleaving["prompt_salt_label"],
    )
    if request_limit is not None and not 1 <= request_limit <= len(events):
        raise ValueError(f"request_limit must be between 1 and {len(events)}")
    output = output.resolve()
    return argparse.Namespace(
        order=workload / "order.jsonl",
        manifest=workload / "manifest.json",
        profile=workload / "profiles.jsonl",
        parquet=ROOT / "inputs/source-metadata/trajectories.selected100.sparse-index.parquet",
        tools=ROOT / "inputs/source-metadata/tools.json",
        tokenizer=ROOT / "inputs/tokenizer",
        base_url=base_url.rstrip("/"),
        model=model,
        run_id=output.name,
        output=output / "client-result.json",
        trace_output=output / "request-records.jsonl",
        snapshot_dir=output / "prompt-snapshots",
        max_concurrency=concurrency,
        max_model_len=max_model_len,
        max_output_tokens=output_tokens,
        request_timeout=request_timeout,
        seed=int(interleaving["seed"]),
        session_pool=int(interleaving["active_sessions"]),
        prompt_salt_prefix=interleaving["prompt_salt_label"],
        request_limit=request_limit,
        api_key_env=api_key_env,
        label=label,
    )


def run_settings(args: argparse.Namespace) -> dict:
    """Describe parameters for the local run record; never read credential values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "measurement_hook"
    }
