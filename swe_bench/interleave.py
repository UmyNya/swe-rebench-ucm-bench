"""The original seeded causal merge; virtual times choose order, not request timing."""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class MergeSettings:
    active_sessions: int = 24
    seed: int = 20260911
    virtual_min_delay: float = 0.5
    virtual_median_delay: float = 1.0
    virtual_delay_sigma: float = 0.6

def _sample_delay(rng: random.Random, config: MergeSettings) -> float:
    return config.virtual_min_delay + rng.lognormvariate(
        math.log(config.virtual_median_delay), config.virtual_delay_sigma
    )

def causal_merge(
    profiles: Sequence[dict[str, Any]], config: MergeSettings
) -> list[dict[str, Any]]:
    """Deterministically interleave sessions; W is not client concurrency."""
    rng = random.Random(config.seed)
    shuffled = list(profiles)
    rng.shuffle(shuffled)
    by_id = {str(item["trajectory_id"]): item for item in shuffled}
    queue: list[tuple[float, int, str, int]] = []
    activation = 0
    tie = 0
    active = 0

    def activate(base_time: float, initial: bool) -> None:
        nonlocal activation, tie, active
        if activation >= len(shuffled):
            return
        profile = shuffled[activation]
        activation += 1
        active += 1
        ready = rng.random() if initial else base_time + _sample_delay(rng, config)
        heapq.heappush(queue, (ready, tie, str(profile["trajectory_id"]), 0))
        tie += 1

    for _ in range(config.active_sessions):
        activate(0.0, True)

    events: list[dict[str, Any]] = []
    previous_request: dict[str, str] = {}
    while queue:
        ready, _, trajectory_id, step_index = heapq.heappop(queue)
        profile = by_id[trajectory_id]
        step = profile["steps"][step_index]
        request_id = f"{trajectory_id}:request-{step['request_index']}"
        events.append(
            {
                "ordinal": len(events),
                "request_id": request_id,
                "depends_on": previous_request.get(trajectory_id),
                "virtual_ready_time": round(ready, 9),
                "source_row": profile["source_row"],
                "trajectory_id": trajectory_id,
                "instance_id": profile["instance_id"],
                "repo": profile["repo"],
                "resolved": profile["resolved"],
                "active_sessions_at_emit": active,
                "unactivated_sessions_remaining": len(shuffled) - activation,
                **step,
            }
        )
        previous_request[trajectory_id] = request_id
        next_step = step_index + 1
        if next_step < len(profile["steps"]):
            heapq.heappush(
                queue,
                (ready + _sample_delay(rng, config), tie, trajectory_id, next_step),
            )
            tie += 1
        else:
            active -= 1
            activate(ready, False)
    return events
