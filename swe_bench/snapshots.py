"""Save exact request inputs locally before the measured interval.

Trajectories share long prefixes. Store each token sequence once and describe
the request slices, instead of writing the same history thousands of times.
The snapshot contains all bytes needed to reconstruct each sent token vector.
"""
from __future__ import annotations

import array
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prompts import PromptSource


def little_endian_bytes(tokens: Sequence[int]) -> bytes:
    values = array.array("I", tokens)
    if values.itemsize != 4:
        raise RuntimeError("prompt snapshots require 32-bit unsigned integers")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def save_prompt_snapshot(
    directory: Path,
    order: Sequence[Mapping[str, Any]],
    sources: Mapping[str, PromptSource],
    *,
    max_model_len: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Validate lengths and persist a lossless snapshot before any requests."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    trajectories: dict[str, Any] = {}
    for index, (trajectory, source) in enumerate(sorted(sources.items())):
        filename = f"trajectory-{index:04d}.u32le.gz"
        raw = little_endian_bytes(source.token_ids)
        path = directory / filename
        with path.open("xb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
                compressed.write(raw)
        path.chmod(0o600)
        trajectories[trajectory] = {
            "file": filename,
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "salt_ids": source.salt_ids,
            "ends": source.ends,
            "starts": getattr(source, "starts", None),
        }

    requests = []
    input_fingerprint = hashlib.sha256()
    for event in order:
        trajectory = str(event["trajectory_id"])
        tokens = sources[trajectory].prompt_for(int(event["request_index"]))
        if len(tokens) + output_tokens > max_model_len:
            raise ValueError(f"input plus output exceeds model context: {event['request_id']}")
        digest = hashlib.sha256(little_endian_bytes(tokens)).hexdigest()
        input_fingerprint.update(bytes.fromhex(digest))
        requests.append({
            "ordinal": event["ordinal"],
            "request_id": event["request_id"],
            "trajectory_id": trajectory,
            "request_index": event["request_index"],
            "prompt_tokens": len(tokens),
            "sha256_u32le": digest,
        })

    manifest = {
        "schema_version": 1,
        "private_artifact": True,
        "format": "uint32-little-endian; salt_ids + trajectory[start:end]",
        "input_fingerprint": input_fingerprint.hexdigest(),
        "trajectories": trajectories,
        "requests": requests,
    }
    path = directory / "manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    path.chmod(0o600)
    return {
        "directory": str(directory.resolve()),
        "requests": len(requests),
        "input_fingerprint": manifest["input_fingerprint"],
        "created_outside_measurement": True,
        "private_artifact": True,
    }


def read_snapshot_prompt(directory: Path, ordinal: int) -> list[int]:
    """Reconstruct and verify one request without the dataset or tokenizer."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if not 0 <= ordinal < len(manifest["requests"]):
        raise ValueError("snapshot request ordinal is out of range")
    request = manifest["requests"][ordinal]
    trajectory = manifest["trajectories"][request["trajectory_id"]]
    filename = Path(trajectory["file"])
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("snapshot token file must be a relative filename")
    with gzip.open(directory / filename, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != trajectory["uncompressed_sha256"]:
        raise ValueError("snapshot trajectory checksum mismatch")
    values = array.array("I")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    index = str(request["request_index"])
    start = (trajectory.get("starts") or {}).get(index, 0)
    tokens = [*trajectory["salt_ids"], *values[start:trajectory["ends"][index]]]
    if hashlib.sha256(little_endian_bytes(tokens)).hexdigest() != request["sha256_u32le"]:
        raise ValueError("snapshot request checksum mismatch")
    return tokens
