"""Behavioral checks for input fidelity, causal dispatch and local snapshots."""
from __future__ import annotations

import array
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from swe_bench.config import ROOT, make_args
from swe_bench.dispatch import dispatch_closed_loop, validate_concurrency
from swe_bench.io import load_jsonl, sha256_file
from swe_bench.prepare import prepare_workload
from swe_bench.prompts import PromptSource
from swe_bench.snapshots import read_snapshot_prompt, save_prompt_snapshot
from swe_bench.workload import validate_order


def event(ordinal, trajectory, dependency=None):
    return {
        "ordinal": ordinal,
        "request_id": f"request-{ordinal}",
        "trajectory_id": trajectory,
        "depends_on": dependency,
        "request_index": ordinal,
    }


class InputTests(unittest.TestCase):
    def test_original_order_and_profiles_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            for w in (24, 32):
                output = Path(temporary) / str(w)
                prepare_workload(output, session_pool=w)
                baseline = ROOT / f"workloads/W{w}-N3379-seed20260911"
                for filename in ("order.jsonl", "profiles.jsonl"):
                    self.assertEqual(sha256_file(output / filename), sha256_file(baseline / filename))

    def test_custom_workload_remains_causal_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / name for name in ("first", "second")]
            for output in outputs:
                prepare_workload(output, session_pool=7, seed=123, trajectory_limit=10, request_limit=93)
            first = load_jsonl(outputs[0] / "order.jsonl")
            validate_order(first)
            self.assertEqual(len(first), 93)
            self.assertLessEqual(max(row["active_sessions_at_emit"] for row in first), 7)
            self.assertEqual(sha256_file(outputs[0] / "order.jsonl"), sha256_file(outputs[1] / "order.jsonl"))

    def test_request_configuration_rejects_invalid_or_secret_urls(self):
        workload = ROOT / "workloads/W24-N3379-seed20260911"
        for settings in [
            {"concurrency": 0}, {"output_tokens": 0}, {"request_limit": 0},
            {"request_timeout": float("nan")}, {"base_url": "http://" + "user:secret@localhost:8000"},
            {"base_url": "http://localhost:8000/v1"},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_args(workload, Path("results/invalid"), **settings)
        for concurrency in (1, 3, 7, 64):
            validate_concurrency(concurrency)

    def test_prompt_snapshot_is_lossless_and_detects_corruption(self):
        sources = {"a": PromptSource(array.array("I", [10, 20, 30, 40]), {0: 2, 1: 4}, [1, 2])}
        order = [event(0, "a"), event(1, "a", "request-0")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "prompt-snapshots"
            report = save_prompt_snapshot(directory, order, sources, max_model_len=16, output_tokens=3)
            self.assertTrue(report["created_outside_measurement"])
            self.assertEqual(read_snapshot_prompt(directory, 0), [1, 2, 10, 20])
            self.assertEqual(read_snapshot_prompt(directory, 1), [1, 2, 10, 20, 30, 40])
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual((directory / "manifest.json").stat().st_mode & 0o777, 0o600)
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["requests"][0]["sha256_u32le"] = "0" * 64
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_snapshot_prompt(directory, 0)


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_order_keeps_head_of_line_dependency(self):
        order = [event(0, "a"), event(1, "a", "request-0"), event(2, "b")]
        entered = asyncio.Event()
        release = asyncio.Event()
        submissions = []

        async def send(row, mark):
            mark()
            submissions.append(row["ordinal"])
            if row["ordinal"] == 0:
                entered.set()
                await release.wait()
            return {"ttft_ms": 1, "prompt_tokens": 10, "output_tokens": 1}

        task = asyncio.create_task(dispatch_closed_loop(order, send, client_concurrency=3))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(submissions, [0])
        release.set()
        report = await asyncio.wait_for(task, timeout=1)
        self.assertEqual(submissions, [0, 1, 2])
        self.assertLessEqual(report.observed_max_inflight, 3)
        self.assertGreaterEqual(report.records[1].actual_submit_s, report.records[0].completion_s)

    async def test_positive_nonstandard_concurrency_and_failure_accounting(self):
        order = [event(index, str(index)) for index in range(7)]

        async def send(row, mark):
            mark()
            await asyncio.sleep(0)
            if row["ordinal"] == 2:
                raise RuntimeError("deliberate transport failure")
            return {"ttft_ms": 1, "prompt_tokens": 10, "output_tokens": 5}

        report = await dispatch_closed_loop(order, send, client_concurrency=3)
        self.assertLessEqual(report.observed_max_inflight, 3)
        self.assertEqual(sum(record.error is not None for record in report.records), 1)
        self.assertEqual([record.submission_sequence for record in report.records], list(range(7)))


if __name__ == "__main__":
    unittest.main()
