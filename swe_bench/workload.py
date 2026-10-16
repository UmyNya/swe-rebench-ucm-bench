"""Input order and manifest validation, without server interaction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import load_jsonl, sha256_file

def validate_order(events: Sequence[Mapping[str, Any]]) -> None:
    if not events:
        raise ValueError("request order is empty")
    if [event.get("ordinal") for event in events] != list(range(len(events))):
        raise ValueError("request ordinals must be contiguous and ordered")
    seen: set[str] = set()
    previous: dict[str, str] = {}
    for event in events:
        request_id = str(event["request_id"])
        trajectory_id = str(event["trajectory_id"])
        if request_id in seen:
            raise ValueError(f"duplicate request ID: {request_id}")
        expected = previous.get(trajectory_id)
        if event.get("depends_on") != expected:
            raise ValueError(
                f"dependency mismatch at ordinal {event['ordinal']}: "
                f"expected {expected!r}, found {event.get('depends_on')!r}"
            )
        seen.add(request_id)
        previous[trajectory_id] = request_id


def validate_closed_loop_workload(
    order_path: Path,
    manifest_path: Path,
    profile_path: Path,
    *,
    seed: int,
    session_pool: int,
    prompt_salt_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a timestamp-free causal workload and its recorded hashes."""
    events = load_jsonl(order_path)
    validate_order(events)
    if any("arrival_time_s" in event for event in events):
        raise ValueError("closed-loop order must not contain arrival timestamps")
    if any(str(event.get("phase")) != "measured" for event in events):
        raise ValueError("the no-primer campaign accepts measured requests only")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    expected_order_hash = outputs.get("order", {}).get("sha256")
    expected_profile_hash = outputs.get("profiles", {}).get("sha256")
    if expected_order_hash != sha256_file(order_path):
        raise ValueError("closed-loop order SHA-256 does not match manifest")
    if expected_profile_hash != sha256_file(profile_path):
        raise ValueError("closed-loop profile SHA-256 does not match manifest")

    interleaving = manifest.get("interleaving", {})
    if int(interleaving.get("seed", -1)) != seed:
        raise ValueError("runner seed does not match workload manifest")
    if int(interleaving.get("active_sessions", -1)) != session_pool:
        raise ValueError("session pool does not match workload manifest")
    if str(interleaving.get("prompt_salt_label")) != prompt_salt_prefix:
        raise ValueError("prompt salt does not match workload manifest")
    if bool(interleaving.get("primer_included", True)):
        raise ValueError("closed-loop campaign requires no-primer workload")

    profile_summary = manifest.get("profile_summary", {})
    if int(profile_summary.get("measured_requests", -1)) != len(events):
        raise ValueError("manifest request count does not match order")
    if int(profile_summary.get("trajectories", -1)) != len(
        {str(event["trajectory_id"]) for event in events}
    ):
        raise ValueError("manifest trajectory count does not match order")
    return events, manifest
