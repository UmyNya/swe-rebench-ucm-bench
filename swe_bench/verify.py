"""Offline verification of shipped files and all frozen baseline token vectors."""
from __future__ import annotations

import hashlib
import json

from .config import ROOT, make_args
from .io import load_jsonl, sha256_file
from .prompts import load_prompt_sources
from .snapshots import little_endian_bytes


def verify_inputs(*, tokens: bool = False) -> dict:
    checksums = json.loads((ROOT / "reference/data-checksums.json").read_text())
    for name, expected in checksums.items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"shipped data checksum mismatch: {name}")
    for w in (24, 32):
        make_args(ROOT / f"workloads/W{w}-N3379-seed20260911", ROOT / "results/unused-verification")
    report = {"passed": True, "files_verified": len(checksums), "token_vectors_checked": False}
    if tokens:
        workload = ROOT / "workloads/W24-N3379-seed20260911"
        args = make_args(workload, ROOT / "results/unused-verification")
        order = load_jsonl(args.order)
        sources, _ = load_prompt_sources(
            order, parquet_path=args.parquet, tools_path=args.tools,
            tokenizer_path=args.tokenizer, prompt_salt_prefix=args.prompt_salt_prefix,
        )
        expected = json.loads((ROOT / "reference/payload-hashes.json").read_text())
        if len(order) != expected["requests"]:
            raise ValueError("baseline request count changed")
        total = 0
        for event, row in zip(order, expected["rows"], strict=True):
            vector = sources[event["trajectory_id"]].prompt_for(event["request_index"])
            digest = hashlib.sha256(little_endian_bytes(vector)).hexdigest()
            if event["request_id"] != row["request_id"] or len(vector) != row["tokens"] or digest != row["token_ids_sha256"]:
                raise ValueError(f"baseline input token vector changed: {event['request_id']}")
            total += len(vector)
        if total != expected["input_tokens"]:
            raise ValueError("baseline total input token count changed")
        report.update(token_vectors_checked=True, requests=len(order), input_tokens=total)
    return report
