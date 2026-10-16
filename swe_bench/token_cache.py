"""Prepare exact request token IDs for another model's local chat template.

The frozen baseline uses its original GLM tokenization path. Other tokenizers
render each historical request independently, before any timed replay.
"""
from __future__ import annotations

import array
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .prompts import PromptSource, _load_selected_parquet_rows, normalize_message
from .snapshots import little_endian_bytes


def render_request(tokenizer: Any, messages: list[dict], assistant_index: int, tools: list) -> list[int]:
    """Render the history immediately before the selected assistant turn."""
    boundaries = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not 0 <= assistant_index < len(boundaries):
        raise ValueError("request index exceeds the trajectory's assistant turns")
    result = tokenizer.apply_chat_template(
        messages[:boundaries[assistant_index]],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if not result or not all(isinstance(token, int) and token >= 0 for token in result):
        raise ValueError("chat template must produce a nonempty flat list of token IDs")
    return result


def prepare_token_cache(
    output: Path,
    profiles: list[dict],
    *,
    tokenizer_path: Path,
    parquet_path: Path,
    tools_path: Path,
    salt_prefix: str,
) -> dict[str, Any]:
    """Render full histories, update profile lengths and store a local token cache."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("the supplied tokenizer must provide a chat template")
    tools = json.loads(tools_path.read_text())
    by_id = {str(profile["trajectory_id"]): profile for profile in profiles}
    rows = _load_selected_parquet_rows(
        pq.ParquetFile(parquet_path), {int(profile["source_row"]) for profile in profiles}
    )
    cache = output / "token-cache"
    cache.mkdir(mode=0o700)
    trajectories = {}
    for position, row in enumerate(rows):
        trajectory = str(row["trajectory_id"])
        profile = by_id[trajectory]
        messages = [normalize_message(message) for message in row["trajectory"]]
        salt = f"<|workload_namespace|>{salt_prefix}|trajectory={trajectory}\n"
        salt_ids = tokenizer.encode(salt, add_special_tokens=False)
        values = array.array("I")
        starts = {}
        ends = {}
        for step in profile["steps"]:
            index = int(step["request_index"])
            tokens = render_request(tokenizer, messages, index, tools)
            starts[index] = len(values)
            values.extend(tokens)
            ends[index] = len(values)
            step.update(
                prompt_tokens=len(tokens),
                body_prompt_tokens=len(tokens),
                salt_tokens=len(salt_ids),
                salted_prompt_tokens=len(tokens) + len(salt_ids),
                chunk_aligned_prompt_tokens=(len(tokens) + len(salt_ids)) // 64 * 64,
            )
        raw = little_endian_bytes(values)
        filename = f"trajectory-{position:04d}.u32le.gz"
        with (cache / filename).open("xb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
                compressed.write(raw)
        trajectories[trajectory] = {
            "file": filename,
            "sha256_u32le": hashlib.sha256(raw).hexdigest(),
            "starts": starts,
            "ends": ends,
            "salt_ids": salt_ids,
        }
    return {
        "format": "prepared-uint32le-v1",
        "directory": "token-cache",
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "trajectories": trajectories,
    }


def load_token_cache(workload: Path, storage: dict, order: list[dict]) -> tuple[dict[str, PromptSource], dict]:
    """Verify cached vectors before running; no tokenizer is needed during measurement."""
    if storage.get("format") != "prepared-uint32le-v1" or storage.get("directory") != "token-cache":
        raise ValueError("unsupported prepared token cache")
    sources = {}
    required = {str(event["trajectory_id"]) for event in order}
    for trajectory in sorted(required):
        metadata = storage["trajectories"][trajectory]
        filename = Path(metadata["file"])
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError("token cache file must be a relative filename")
        with gzip.open(workload / "token-cache" / filename, "rb") as handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != metadata["sha256_u32le"]:
            raise ValueError(f"token cache checksum mismatch for {trajectory}")
        values = array.array("I")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        sources[trajectory] = PromptSource(
            values,
            {int(key): value for key, value in metadata["ends"].items()},
            metadata["salt_ids"],
            {int(key): value for key, value in metadata["starts"].items()},
        )
    for event in order:
        source = sources[str(event["trajectory_id"])]
        index = int(event["request_index"])
        start = source.starts[index]
        end = source.ends[index]
        if not 0 <= start < end <= len(source.token_ids):
            raise ValueError("prepared token slice is outside its trajectory")
        if end - start + len(source.salt_ids) != event["salted_prompt_tokens"]:
            raise ValueError("prepared token count does not match the workload")
    return sources, {
        "trajectory_count": len(sources),
        "mode": "prepared_token_cache",
        "chat_template_sha256": storage["chat_template_sha256"],
        "prompt_salt_scope": "trajectory",
    }
