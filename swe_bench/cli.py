"""Explicit request-side commands for an independently managed vLLM service."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import ROOT, make_args, run_settings
from .replay import replay
from .snapshots import read_snapshot_prompt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Replay requests against an existing service")
    workload = run.add_mutually_exclusive_group()
    workload.add_argument("--workload", choices=["W24", "W32"], default="W24")
    workload.add_argument("--workload-dir", type=Path, help="A workload made by the prepare command")
    run.add_argument("--base-url", default="http://127.0.0.1:18147", help="Service root, without /v1")
    run.add_argument("--model", default="glm52-mixed", help="Served model name")
    run.add_argument("--output-dir", type=Path, required=True, help="New local results directory")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--output-tokens", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=131072, help="Client context limit; does not configure the server")
    run.add_argument("--request-timeout", type=float, default=3600)
    run.add_argument("--request-limit", type=int, help="Replay a causal prefix for a smaller test")
    run.add_argument("--label", default="", help="Local annotation, e.g. OFF; does not configure the server")
    run.add_argument("--api-key-env", default="SWE_BENCH_API_KEY", help="Environment variable containing the API key")
    run.add_argument("--dry-run", action="store_true", help="Check paths, hashes and parameters without contacting the server")

    prepare = commands.add_parser("prepare", help="Create a workload with a different W, seed or causal subset")
    prepare.add_argument("--session-pool", type=int, default=24, help="W: interleaved active trajectories, not HTTP concurrency")
    prepare.add_argument("--seed", type=int, default=20260911)
    prepare.add_argument("--trajectory-limit", type=int)
    prepare.add_argument("--request-limit", type=int)
    prepare.add_argument("--tokenizer", type=Path, help="Prepare new token IDs using another local model's chat template")
    prepare.add_argument("--output-dir", type=Path, required=True)

    snapshot = commands.add_parser("snapshot", help="Read one prompt from a local run snapshot")
    snapshot.add_argument("directory", type=Path, help="The run's prompt-snapshots directory")
    snapshot.add_argument("--ordinal", type=int, required=True)
    snapshot.add_argument("--output", type=Path, required=True, help="Write exact token IDs to a new local JSON file")

    compare = commands.add_parser("compare", help="Compare one or more completed OFF/ON pairs")
    compare.add_argument("--off", type=Path, nargs="+", required=True)
    compare.add_argument("--on", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--allow-parameter-differences", action="store_true", help="Explicitly compare different request settings instead of a controlled OFF/ON pair")

    verify = commands.add_parser("verify", help="Verify bundled data without sending requests")
    verify.add_argument("--tokens", action="store_true", help="Also reconstruct and verify all 3379 original token vectors")
    return result


def run_command(options: argparse.Namespace) -> int:
    workload = options.workload_dir or ROOT / f"workloads/{options.workload}-N3379-seed20260911"
    args = make_args(
        workload,
        options.output_dir,
        base_url=options.base_url,
        model=options.model,
        concurrency=options.concurrency,
        output_tokens=options.output_tokens,
        max_model_len=options.max_model_len,
        request_timeout=options.request_timeout,
        request_limit=options.request_limit,
        api_key_env=options.api_key_env,
        label=options.label,
    )
    settings = run_settings(args)
    if options.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        return 0

    output = args.output.parent
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "client-config.json").write_text(
        json.dumps({"private_artifact": True, **settings}, indent=2, ensure_ascii=False) + "\n"
    )
    result, code = asyncio.run(replay(args))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return code


def main() -> int:
    command_parser = parser()
    options = command_parser.parse_args()
    previous_umask = os.umask(0o077)
    try:
        if options.command == "run":
            return run_command(options)
        if options.command == "verify":
            from .verify import verify_inputs

            print(json.dumps(verify_inputs(tokens=options.tokens), indent=2))
            return 0
        if options.command == "prepare":
            from .prepare import prepare_workload

            manifest = prepare_workload(
                options.output_dir,
                session_pool=options.session_pool,
                seed=options.seed,
                trajectory_limit=options.trajectory_limit,
                request_limit=options.request_limit,
                tokenizer=options.tokenizer,
            )
            print(json.dumps(manifest, indent=2))
            return 0
        if options.command == "snapshot":
            tokens = read_snapshot_prompt(options.directory, options.ordinal)
            with options.output.open("x", encoding="utf-8") as handle:
                json.dump({"private_artifact": True, "prompt_token_ids": tokens}, handle)
                handle.write("\n")
            return 0
        if options.command == "compare":
            from .compare import compare_runs

            report = compare_runs(options.off, options.on, allow_parameter_differences=options.allow_parameter_differences)
            options.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with options.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            print(json.dumps(report["on_vs_off_percent"], indent=2))
            return 0
        raise ValueError(f"unknown command: {options.command}")
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        command_parser.error(str(error))
    finally:
        os.umask(previous_umask)
    return 2
