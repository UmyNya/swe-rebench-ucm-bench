"""Atomic, local-only per-request trace output."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .dispatch import ClosedLoopRecord, serialize_record

class AtomicClosedLoopTrace:
    """Publish a new JSONL request trace atomically."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.temporary = target.with_suffix(target.suffix + ".tmp")
        if self.target.exists() or self.temporary.exists():
            raise FileExistsError(f"refusing to overwrite trace: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("x", encoding="utf-8")
        self._closed = False

    def append(self, record: ClosedLoopRecord) -> None:
        """Append one completed record."""
        if self._closed:
            raise RuntimeError("trace is already closed")
        self._handle.write(
            json.dumps(
                serialize_record(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def close(self) -> None:
        """Flush, fsync, and atomically publish the trace."""
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.temporary.replace(self.target)
        self._closed = True
