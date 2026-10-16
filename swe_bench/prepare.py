"""Create a causal workload by changing request-side sampling and interleaving."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .interleave import MergeSettings, causal_merge
from .io import load_jsonl, sha256_file
from .workload import validate_order


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def prepare_workload(
    output: Path,
    *,
    session_pool: int = 24,
    seed: int = 20260911,
    trajectory_limit: int | None = None,
    request_limit: int | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    """Keep complete historical prompts and causal dependencies for each selection."""
    if session_pool < 1:
        raise ValueError("session_pool must be positive")
    original = ROOT / "workloads/W24-N3379-seed20260911"
    profiles = copy.deepcopy(load_jsonl(original / "profiles.jsonl"))
    if trajectory_limit is not None:
        if not 1 <= trajectory_limit <= len(profiles):
            raise ValueError(f"trajectory_limit must be between 1 and {len(profiles)}")
        profiles = profiles[:trajectory_limit]
    if session_pool > len(profiles):
        raise ValueError("session_pool cannot exceed the selected trajectory count")
    available_requests = sum(len(profile["steps"]) for profile in profiles)
    if request_limit is not None and not 1 <= request_limit <= available_requests:
        raise ValueError(f"request_limit must be between 1 and {available_requests}")
    storage = None
    output_created = False
    if tokenizer is not None:
        from .token_cache import prepare_token_cache

        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        output_created = True
        storage = prepare_token_cache(
            output, profiles, tokenizer_path=tokenizer,
            parquet_path=ROOT / "inputs/source-metadata/trajectories.selected100.sparse-index.parquet",
            tools_path=ROOT / "inputs/source-metadata/tools.json",
            salt_prefix="swe-real-20260911",
        )
    events = causal_merge(profiles, MergeSettings(active_sessions=session_pool, seed=seed))
    if request_limit is not None:
        events = events[:request_limit]
    validate_order(events)

    selected = {event["request_id"] for event in events}
    selected_profiles = []
    for profile in profiles:
        steps = [
            step for step in profile["steps"]
            if f"{profile['trajectory_id']}:request-{step['request_index']}" in selected
        ]
        if steps:
            selected_profiles.append({**profile, "steps": steps, "measured_requests": len(steps)})

    if not output_created:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_jsonl(output / "order.jsonl", events)
    write_jsonl(output / "profiles.jsonl", selected_profiles)
    manifest = {
        "schema_version": 1,
        "generated_workload": True,
        "outputs": {
            name: {"path": f"{name}.jsonl", "sha256": sha256_file(output / f"{name}.jsonl")}
            for name in ["order", "profiles"]
        },
        "interleaving": {
            "seed": seed,
            "active_sessions": session_pool,
            "prompt_salt_label": "swe-real-20260911",
            "primer_included": False,
            "method": "original causal_merge stochastic renewal",
            "virtual_min_delay": 0.5,
            "virtual_median_delay": 1.0,
            "virtual_delay_sigma": 0.6,
        },
        "profile_summary": {
            "trajectories": len(selected_profiles),
            "measured_requests": len(events),
        },
        "selection": {
            "trajectory_policy": "prefix of the frozen profile list before seeded interleaving",
            "trajectory_limit": trajectory_limit,
            "request_limit": request_limit,
            "full_history_preserved": True,
        },
        "source_profiles_sha256": sha256_file(original / "profiles.jsonl"),
        "tokenizer": "bundled GLM tokenizer; input token IDs must match the tested model",
        "max_model_len": 131072,
    }
    if storage is not None:
        manifest["token_storage"] = storage
        manifest["tokenizer"] = "Prepared with the supplied local tokenizer; a new tokenization, not the frozen baseline."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
