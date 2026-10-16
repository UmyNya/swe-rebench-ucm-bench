"""Rebuild the exact frozen GLM input token sequences from public trajectories."""
from __future__ import annotations

import array
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import distribution

def normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = str(message["role"])
    normalized: dict[str, Any] = {
        "role": role,
        "content": message.get("content") or "",
    }
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(
                {
                    "id": call.get("id"),
                    "type": call.get("type") or "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                }
            )
        normalized["tool_calls"] = calls
    if role == "tool":
        normalized["name"] = message.get("name")
        normalized["tool_call_id"] = message.get("tool_call_id")
    return normalized


def subsequence_ends(values: Sequence[int], marker: Sequence[int]) -> list[int]:
    return [
        start + len(marker)
        for start in range(len(values) - len(marker) + 1)
        if values[start : start + len(marker)] == marker
    ]


@dataclass
class PromptSource:
    token_ids: array.array
    ends: dict[int, int]
    salt_ids: list[int]
    starts: dict[int, int] | None = None

    def prompt_for(self, request_index: int) -> list[int]:
        start = 0 if self.starts is None else self.starts[request_index]
        return [*self.salt_ids, *self.token_ids[start:self.ends[request_index]]]


def _load_selected_parquet_rows(
    parquet: Any, selected_source_rows: set[int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    global_start = 0
    for row_group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        group_end = global_start + count
        wanted = {
            value for value in selected_source_rows if global_start <= value < group_end
        }
        if wanted:
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=32,
                columns=["trajectory_id", "trajectory"],
                row_groups=[row_group],
            ):
                for row in batch.to_pylist():
                    source_row = global_start + offset
                    offset += 1
                    if source_row in wanted:
                        selected.append({**row, "source_row": source_row})
        global_start = group_end
    if len(selected) != len(selected_source_rows):
        raise RuntimeError(
            f"loaded {len(selected)} source rows, expected {len(selected_source_rows)}"
        )
    return selected


def load_prompt_sources(
    order: Sequence[Mapping[str, Any]],
    *,
    parquet_path: Path,
    tools_path: Path,
    tokenizer_path: Path,
    prompt_salt_prefix: str,
) -> tuple[dict[str, PromptSource], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("runner requires pyarrow and transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    marker = tokenizer.encode("<|assistant|><think></think>", add_special_tokens=False)
    by_trajectory: dict[str, list[Mapping[str, Any]]] = {}
    for event in order:
        by_trajectory.setdefault(str(event["trajectory_id"]), []).append(event)
    rows = _load_selected_parquet_rows(
        pq.ParquetFile(parquet_path), {int(event["source_row"]) for event in order}
    )
    sources: dict[str, PromptSource] = {}
    body_lengths: list[int] = []
    salted_lengths: list[int] = []
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        messages = [normalize_message(message) for message in row["trajectory"]]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids: Any = (
            tokenized["input_ids"]
            if hasattr(tokenized, "keys") and "input_ids" in tokenized
            else tokenized
        )
        if hasattr(full_ids, "tolist"):
            full_ids = full_ids.tolist()
        ends = subsequence_ends(full_ids, marker)
        selected_ends: dict[int, int] = {}
        salt = (
            f"<|workload_namespace|>{prompt_salt_prefix}|"
            f"trajectory={trajectory_id}\n"
        )
        salt_ids = tokenizer.encode(salt, add_special_tokens=False)
        for event in by_trajectory[trajectory_id]:
            request_index = int(event["request_index"])
            if request_index >= len(ends):
                raise ValueError(
                    f"assistant boundary missing for {event['request_id']}"
                )
            body = int(ends[request_index])
            expected = int(event.get("body_prompt_tokens", event["prompt_tokens"]))
            if body != expected:
                raise ValueError(
                    f"body token mismatch for {event['request_id']}: {body} != {expected}"
                )
            expected_salted = int(
                event.get("salted_prompt_tokens", body + len(salt_ids))
            )
            if body + len(salt_ids) != expected_salted:
                raise ValueError(f"salted token mismatch for {event['request_id']}")
            selected_ends[request_index] = body
            body_lengths.append(body)
            salted_lengths.append(expected_salted)
        sources[trajectory_id] = PromptSource(
            token_ids=array.array("I", full_ids),
            ends=selected_ends,
            salt_ids=salt_ids,
        )
    if set(sources) != set(by_trajectory):
        raise ValueError("tokenized trajectory set does not match order")
    return sources, {
        "trajectory_count": len(sources),
        "body_prompt_tokens": distribution(body_lengths),
        "salted_prompt_tokens": distribution(salted_lengths),
        "tokenizer": str(tokenizer_path.resolve()),
        "prompt_salt_scope": "trajectory",
    }
