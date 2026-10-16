"""vLLM-compatible streaming transport with the original HTTP send timing boundary."""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

Clock = Callable[[], float]
HTTP_READ_BUFSIZE = 8 * 1024 * 1024

def create_http_session(
    aiohttp_module: Any,
    *,
    timeout_seconds: float,
    trace_configs: Sequence[Any] = (),
) -> Any:
    """Create the unbounded client with room for 128k-token SSE lines."""
    timeout = aiohttp_module.ClientTimeout(total=timeout_seconds)
    connector = aiohttp_module.TCPConnector(
        limit=0,
        limit_per_host=0,
        force_close=True,
    )
    return aiohttp_module.ClientSession(
        timeout=timeout,
        connector=connector,
        trace_configs=list(trace_configs),
        # return_token_ids can put every prompt token ID in one SSE JSON line.
        read_bufsize=HTTP_READ_BUFSIZE,
    )


async def consume_completion_sse(
    content: Any,
    *,
    clock: Clock,
) -> tuple[float | None, int, int]:
    """Consume a vLLM completion SSE stream and return timing plus usage."""
    first_token_at: float | None = None
    prompt_tokens = 0
    output_tokens = 0
    while not content.at_eof():
        raw_line = await content.readline()
        if not raw_line:
            break
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        item = json.loads(line[6:])
        usage = item.get("usage")
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            output_tokens = int(usage.get("completion_tokens", output_tokens))
        choices = item.get("choices") or []
        token_ids = choices[0].get("token_ids") or [] if choices else []
        text_value = choices[0].get("text") if choices else None
        if first_token_at is None and (token_ids or text_value):
            first_token_at = clock()
    return first_token_at, prompt_tokens, output_tokens
