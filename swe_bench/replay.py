"""Prepare inputs, warm up once, then measure a closed-loop request replay."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Callable, Mapping

from .dispatch import dispatch_closed_loop, serialize_record, validate_concurrency
from .io import sha256_file
from .metrics import distribution, summarize
from .prompts import load_prompt_sources
from .snapshots import save_prompt_snapshot
from .trace import AtomicClosedLoopTrace
from .transport import consume_completion_sse, create_http_session
from .workload import validate_closed_loop_workload

async def replay(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Validate inputs, run one excluded warmup, then replay once."""
    validate_concurrency(args.max_concurrency)
    if args.max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if args.max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    if args.output.exists() or args.trace_output.exists():
        raise FileExistsError("refusing to overwrite result or trace output")
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("--request-limit must be positive")

    full_order, manifest = validate_closed_loop_workload(
        args.order,
        args.manifest,
        args.profile,
        seed=args.seed,
        session_pool=args.session_pool,
        prompt_salt_prefix=args.prompt_salt_prefix,
    )
    limit = len(full_order) if args.request_limit is None else args.request_limit
    if limit > len(full_order):
        raise ValueError("--request-limit exceeds workload request count")
    order = full_order[:limit]
    if "token_storage" in manifest:
        from .token_cache import load_token_cache

        sources, prompt_summary = load_token_cache(args.order.parent, manifest["token_storage"], order)
    else:
        sources, prompt_summary = load_prompt_sources(
            order,
            parquet_path=args.parquet,
            tools_path=args.tools,
            tokenizer_path=args.tokenizer,
            prompt_salt_prefix=args.prompt_salt_prefix,
        )
    snapshot = save_prompt_snapshot(
        args.snapshot_dir,
        order,
        sources,
        max_model_len=args.max_model_len,
        output_tokens=args.max_output_tokens,
    )

    try:
        import aiohttp
    except ModuleNotFoundError as error:
        raise RuntimeError("runner requires aiohttp") from error

    loop = asyncio.get_running_loop()
    trace_config = aiohttp.TraceConfig()

    async def on_chunk_sent(_session: Any, trace_context: Any, _params: Any) -> None:
        context = trace_context.trace_request_ctx
        if not context["submit_marked"]:
            context["submit_marked"] = True
            context["submitted_at"] = loop.time()
            context["mark_submitted"](context["submitted_at"])

    trace_config.on_request_chunk_sent.append(on_chunk_sent)
    base_url = args.base_url.rstrip("/")
    namespace = sha256_file(args.order)[:16]

    async with create_http_session(
        aiohttp,
        timeout_seconds=args.request_timeout,
        trace_configs=[trace_config],
    ) as session:
        api_key = os.environ.get(getattr(args, "api_key_env", "SWE_BENCH_API_KEY"))
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        async with session.get(f"{base_url}/health") as response:
            if response.status != 200:
                raise RuntimeError(f"health endpoint returned HTTP {response.status}")

        async def completion_request(
            *,
            prompt: str | list[int],
            request_id: str,
            mark_submitted: Callable[[float | None], None],
        ) -> dict[str, Any]:
            payload = await asyncio.to_thread(
                json.dumps,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_output_tokens,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": args.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "return_token_ids": True,
                    "request_id": request_id,
                },
                separators=(",", ":"),
            )
            trace_context: dict[str, Any] = {
                "mark_submitted": mark_submitted,
                "submit_marked": False,
                "submitted_at": None,
            }
            async with session.post(
                f"{base_url}/v1/completions",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                trace_request_ctx=trace_context,
            ) as response:
                if response.status >= 400:
                    body = (await response.text())[:2000]
                    raise RuntimeError(f"HTTP {response.status}: {body}")
                first, prompt_tokens, output_tokens = await consume_completion_sse(
                    response.content,
                    clock=loop.time,
                )
            completed = loop.time()
            submitted = trace_context["submitted_at"]
            if submitted is None:
                raise RuntimeError("aiohttp did not expose HTTP send boundary")
            if output_tokens != args.max_output_tokens:
                raise RuntimeError(
                    f"expected {args.max_output_tokens} output tokens for {request_id}, got {output_tokens}"
                )
            if first is None:
                if output_tokens > 0:
                    first = completed
                else:
                    raise RuntimeError(f"no token returned for {request_id}")
            return {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": max((first - submitted) * 1000, 0.0),
            }

        warmup_submit: float | None = None

        def mark_warmup_submitted(at: float | None = None) -> None:
            nonlocal warmup_submit
            warmup_submit = loop.time() if at is None else at

        warmup_started = loop.time()
        warmup_response = await completion_request(
            prompt=(
                "Warm up this inference service and answer briefly. "
                "This request is excluded from closed-loop metrics."
            ),
            request_id=f"swe-closed-warmup-{args.run_id}",
            mark_submitted=mark_warmup_submitted,
        )
        warmup_completed = loop.time()
        warmup = {
            **warmup_response,
            "included_in_metrics": False,
            "total_ms": max(
                (warmup_completed - (warmup_submit or warmup_started)) * 1000,
                0.0,
            ),
        }

        async def send_event(
            event: Mapping[str, Any],
            mark_submitted: Callable[[float | None], None],
        ) -> Mapping[str, Any]:
            prompt_ids = sources[str(event["trajectory_id"])].prompt_for(
                int(event["request_index"])
            )
            if len(prompt_ids) + args.max_output_tokens > args.max_model_len:
                raise ValueError(
                    f"salted prompt exceeds max model length: {event['request_id']}"
                )
            response = await completion_request(
                prompt=prompt_ids,
                request_id=(
                    f"swe-closed-{namespace}-{args.max_concurrency}-"
                    f"{int(event['ordinal'])}"
                ),
                mark_submitted=mark_submitted,
            )
            if int(response["prompt_tokens"]) != len(prompt_ids):
                raise ValueError(
                    f"server prompt token mismatch for {event['request_id']}: "
                    f"{response['prompt_tokens']} != {len(prompt_ids)}"
                )
            return {
                **response,
                "expected_body_prompt_tokens": int(event["prompt_tokens"]),
                "expected_salted_prompt_tokens": len(prompt_ids),
            }

        if getattr(args, "measurement_hook", None):
            args.measurement_hook("before")
        measurement_errors: list[dict[str, str]] = []
        measurement_start_ns = time.monotonic_ns()
        measurement_start_wall_ns = time.time_ns()
        trace_writer = AtomicClosedLoopTrace(args.trace_output)
        try:
            report = await dispatch_closed_loop(
                order,
                send_event,
                client_concurrency=args.max_concurrency,
                clock=loop.time,
                record_callback=trace_writer.append,
            )
        finally:
            measurement_end_ns = time.monotonic_ns()
            trace_writer.close()
            if getattr(args, "measurement_hook", None):
                try:
                    args.measurement_hook("after")
                except Exception as error:
                    # Preserve client evidence when the engine has died and
                    # the final worker snapshot can no longer be obtained.
                    measurement_errors.append(
                        {"stage": "after", "error": f"{type(error).__name__}: {error}"}
                    )

    summary = summarize(report)
    expected_ordinals = list(range(len(order)))
    submitted_ordinals = [
        record.ordinal
        for record in sorted(
            (record for record in report.records if record.submission_sequence is not None),
            key=lambda item: int(item.submission_sequence or 0),
        )
    ]
    previous_submission: dict[str, int] = {}
    same_trajectory_gaps: list[float] = []
    for record in sorted(
        (record for record in report.records if record.submission_sequence is not None),
        key=lambda item: int(item.submission_sequence or 0),
    ):
        assert record.submission_sequence is not None
        previous = previous_submission.get(record.trajectory_id)
        if previous is not None:
            same_trajectory_gaps.append(record.submission_sequence - previous - 1)
        previous_submission[record.trajectory_id] = record.submission_sequence
    result = {
        "measurement_window": {"start_ns": measurement_start_ns, "end_ns": measurement_end_ns, "start_wall_ns": measurement_start_wall_ns},
        "schema_version": 1,
        "private_artifact": True,
        "label": getattr(args, "label", ""),
        "service_management": "external; server configuration is not inferred from the label",
        "status": (
            "invalid_measurement"
            if measurement_errors
            else "completed" if summary["errors"] == 0 else "completed_with_errors"
        ),
        "measurement_errors": measurement_errors,
        "run_id": args.run_id,
        "model": args.model,
        "protocol": {
            "name": "closed_loop_completion_driven",
            "client_concurrency": args.max_concurrency,
            "active_session_pool": args.session_pool,
            "active_session_pool_is_client_concurrency": False,
            "arrival_schedule": None,
            "offered_rate_rps": None,
            "send_order": "strict_manifest_order",
            "dependency_policy": "wait_for_depends_on_completion",
            "global_hol_compatibility": "preserved_from_historical_runner",
        },
        "order": str(args.order.resolve()),
        "manifest": str(args.manifest.resolve()),
        "profile": str(args.profile.resolve()),
        "order_sha256": sha256_file(args.order),
        "manifest_sha256": sha256_file(args.manifest),
        "profile_sha256": sha256_file(args.profile),
        "trace_output": str(args.trace_output.resolve()),
        "trace_sha256": sha256_file(args.trace_output),
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "request_timeout_s": args.request_timeout,
        "prompt_salt_prefix": args.prompt_salt_prefix,
        "request_limit": args.request_limit,
        "workload_profile_summary": manifest.get("profile_summary"),
        "prompt_preparation": prompt_summary,
        "prompt_snapshot": snapshot,
        "warmup": warmup,
        "execution_order": {
            "client_send_order_matches_manifest": submitted_ordinals
            == expected_ordinals,
            "actual_submit_order_sha256": summary[
                "actual_submit_order_sha256"
            ],
            "same_trajectory_intervening_requests": distribution(
                same_trajectory_gaps
            ),
            "manifest_same_trajectory_intervening_requests": manifest.get(
                "interval_analysis", {}
            ).get("all_measured", {}).get("gap_requests"),
        },
        "summary": summary,
        "state_timeline": report.state_timeline,
        "records": [serialize_record(record) for record in report.records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return result, 0 if summary["errors"] == 0 and not measurement_errors else 1
