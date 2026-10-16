"""Completion-driven dispatch preserving strict order and trajectory dependencies."""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .workload import validate_order

SendEvent = Callable[[Mapping[str, Any], Callable[[float | None], None]], Awaitable[Mapping[str, Any] | None]]

def validate_concurrency(value: int) -> None:
    """Accept any positive integer concurrency for request-side experiments."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("client concurrency must be a positive integer")


@dataclass
class ClosedLoopRecord:
    """Auditable timing state for one closed-loop request."""

    ordinal: int
    request_id: str
    trajectory_id: str
    depends_on: str | None
    dispatch_considered_s: float | None = None
    dependency_ready_s: float | None = None
    actual_submit_s: float | None = None
    completion_s: float | None = None
    dependency_wait_ms: float | None = None
    client_admission_wait_ms: float | None = None
    total_ms: float | None = None
    inflight_after_submit: int | None = None
    inflight_before_completion: int | None = None
    submission_sequence: int | None = None
    completion_sequence: int | None = None
    response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ClosedLoopReport:
    """Dispatcher result independent of the HTTP transport."""

    records: list[ClosedLoopRecord]
    state_timeline: list[dict[str, Any]]
    client_concurrency: int
    observed_max_inflight: int
    wall_seconds: float


async def dispatch_closed_loop(
    events: Sequence[Mapping[str, Any]],
    send_event: SendEvent,
    *,
    client_concurrency: int,
    clock: Callable[[], float] = time.perf_counter,
    record_callback: Callable[[ClosedLoopRecord], None] | None = None,
) -> ClosedLoopReport:
    """Replay strict manifest order with causal and closed-loop bounds.

    The behavior intentionally matches the historical runner: dispatch order
    is the order file, a request waits for its direct dependency, and a new
    request is created only when fewer than ``client_concurrency`` prior tasks
    remain incomplete.  Thus C16/C32 is a completion-driven outstanding limit,
    not W, ``max_num_seqs``, an offered rate, or a nominal schedule.

    Preserving exact send order also preserves the historical global
    head-of-line behavior when the next manifest entry depends on an unfinished
    request.  That property is recorded rather than hidden.
    """
    validate_order(events)
    validate_concurrency(client_concurrency)
    loop = asyncio.get_running_loop()
    origin = clock()
    completion = {
        str(event["request_id"]): asyncio.Event() for event in events
    }
    records = [
        ClosedLoopRecord(
            ordinal=int(event["ordinal"]),
            request_id=str(event["request_id"]),
            trajectory_id=str(event["trajectory_id"]),
            depends_on=(
                None if event.get("depends_on") is None else str(event["depends_on"])
            ),
        )
        for event in events
    ]
    in_flight_tasks: set[asyncio.Task[None]] = set()
    state_timeline: list[dict[str, Any]] = []
    submit_ordinals: list[int] = []
    completion_sequence = 0
    observed_max_inflight = 0
    network_inflight = 0

    def relative_now() -> float:
        return max(clock() - origin, 0.0)

    def snapshot(reason: str, record: ClosedLoopRecord) -> None:
        state_timeline.append(
            {
                "time_s": round(relative_now(), 9),
                "reason": reason,
                "ordinal": record.ordinal,
                "inflight": network_inflight,
                "active_tasks": len(in_flight_tasks),
                "completed": completion_sequence,
            }
        )

    async def reap_finished(wait_for_one: bool) -> None:
        nonlocal in_flight_tasks
        if not in_flight_tasks:
            return
        done = {task for task in in_flight_tasks if task.done()}
        if not done and wait_for_one:
            done, pending = await asyncio.wait(
                in_flight_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight_tasks = set(pending)
        else:
            in_flight_tasks -= done
        if done:
            await asyncio.gather(*done)

    async def run_one(
        event: Mapping[str, Any],
        record: ClosedLoopRecord,
        submitted_signal: asyncio.Event,
    ) -> None:
        nonlocal completion_sequence, observed_max_inflight, network_inflight
        submitted = False

        def mark_submitted(at: float | None = None) -> None:
            def register() -> None:
                nonlocal submitted, observed_max_inflight, network_inflight
                if submitted:
                    raise RuntimeError(
                        f"submit boundary repeated for {record.request_id}"
                    )
                submitted = True
                absolute = clock() if at is None else at
                record.actual_submit_s = max(absolute - origin, 0.0)
                record.submission_sequence = len(submit_ordinals)
                submit_ordinals.append(record.ordinal)
                network_inflight += 1
                observed_max_inflight = max(observed_max_inflight, network_inflight)
                record.inflight_after_submit = network_inflight
                snapshot("submit", record)
                submitted_signal.set()

            loop.call_soon_threadsafe(register)

        try:
            response = await send_event(event, mark_submitted)
            if not submitted_signal.is_set():
                await asyncio.sleep(0)
            if not submitted_signal.is_set():
                raise RuntimeError("send completed without an HTTP send boundary")
            if response is not None:
                record.response = dict(response)
        except asyncio.CancelledError:
            record.error = "CancelledError: closed-loop replay aborted"
            raise
        except Exception as error:  # noqa: BLE001 - preserve per-request failure
            record.error = f"{type(error).__name__}: {error}"
        finally:
            now = relative_now()
            record.completion_s = now
            if submitted and record.actual_submit_s is not None:
                record.inflight_before_completion = network_inflight
                network_inflight -= 1
                record.total_ms = max((now - record.actual_submit_s) * 1000, 0.0)
            record.completion_sequence = completion_sequence
            completion_sequence += 1
            completion[record.request_id].set()
            snapshot("complete", record)
            if record_callback is not None:
                record_callback(record)
            # A transport failure before the callback must not deadlock the
            # dispatcher.  It remains an explicit request error and invalidates
            # a formal run.
            submitted_signal.set()

    for event, record in zip(events, records, strict=True):
        record.dispatch_considered_s = relative_now()
        dependency_started = clock()
        if record.depends_on is not None:
            await completion[record.depends_on].wait()
        record.dependency_ready_s = relative_now()
        record.dependency_wait_ms = max((clock() - dependency_started) * 1000, 0.0)
        snapshot("dependency_ready", record)

        admission_started = clock()
        await reap_finished(wait_for_one=False)
        while len(in_flight_tasks) >= client_concurrency:
            await reap_finished(wait_for_one=True)
        record.client_admission_wait_ms = max(
            (clock() - admission_started) * 1000, 0.0
        )

        submitted_signal = asyncio.Event()
        task = asyncio.create_task(run_one(event, record, submitted_signal))
        in_flight_tasks.add(task)
        # The historical runner serialized the actual socket-send boundaries,
        # not response processing.  This keeps the client send order identical
        # to the manifest even with many outstanding requests.
        await submitted_signal.wait()

    if in_flight_tasks:
        await asyncio.gather(*in_flight_tasks)
    wall_seconds = relative_now()
    return ClosedLoopReport(
        records=records,
        state_timeline=state_timeline,
        client_concurrency=client_concurrency,
        observed_max_inflight=observed_max_inflight,
        wall_seconds=wall_seconds,
    )


def serialize_record(record: ClosedLoopRecord) -> dict[str, Any]:
    """Flatten transport response metrics into the request record."""
    value = asdict(record)
    response = value.pop("response")
    value.update(response)
    return value
