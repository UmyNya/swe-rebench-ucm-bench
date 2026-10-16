"""Exercise real HTTP/SSE, local prompt snapshots and comparison without a GPU."""
from __future__ import annotations

import array
import copy
import importlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from swe_bench.compare import compare_runs
from swe_bench.config import make_args
from swe_bench.io import sha256_file
from swe_bench.prepare import write_jsonl
from swe_bench.prompts import PromptSource
from swe_bench.snapshots import read_snapshot_prompt
from swe_bench.token_cache import render_request

replay_module = importlib.import_module("swe_bench.replay")


def create_workload(path: Path):
    path.mkdir()
    events = []
    for ordinal, (trajectory, index, body, dependency) in enumerate([
        ("a", 0, 2, None), ("b", 0, 2, None), ("a", 1, 4, "request-0"),
    ]):
        events.append({
            "ordinal": ordinal, "request_id": f"request-{ordinal}",
            "trajectory_id": trajectory, "request_index": index, "depends_on": dependency,
            "phase": "measured", "prompt_tokens": body, "salted_prompt_tokens": body + 1,
        })
    write_jsonl(path / "order.jsonl", events)
    write_jsonl(path / "profiles.jsonl", [{"trajectory_id": "a"}, {"trajectory_id": "b"}])
    manifest = {
        "outputs": {name: {"sha256": sha256_file(path / f"{name}.jsonl")} for name in ["order", "profiles"]},
        "interleaving": {"seed": 1, "active_sessions": 2, "prompt_salt_label": "test", "primer_included": False},
        "profile_summary": {"measured_requests": 3, "trajectories": 2},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return {
        "a": PromptSource(array.array("I", [1, 2, 3, 4]), {0: 2, 1: 4}, [9]),
        "b": PromptSource(array.array("I", [5, 6]), {0: 2}, [9]),
    }


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = create_workload(self.root / "workload")
        self.secret = uuid.uuid4().hex
        self.fail_request = False
        self.payloads = []
        self.authorizations = []

        async def health(request):
            return web.json_response({"status": "ok"})

        async def completion(request):
            payload = await request.json()
            self.payloads.append(payload)
            self.authorizations.append(request.headers.get("Authorization"))
            if self.fail_request and isinstance(payload["prompt"], list) and payload["request_id"].endswith("-1"):
                return web.Response(status=500, text="deliberate test failure")
            count = payload["max_tokens"]
            stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await stream.prepare(request)
            await stream.write(b'data: {"choices":[{"token_ids":[1],"text":""}]}\n\n')
            usage = {"prompt_tokens": len(payload["prompt"]) if isinstance(payload["prompt"], list) else 18, "completion_tokens": count}
            await stream.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            return stream

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/v1/completions", completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temporary.cleanup()

    async def run_replay(self, name):
        output = self.root / name
        output.mkdir()
        args = make_args(self.root / "workload", output, base_url=self.base_url, concurrency=3, output_tokens=5, max_model_len=32, api_key_env="TEST_BENCH_KEY")
        stages = []

        def hook(stage):
            self.assertTrue((args.snapshot_dir / "manifest.json").exists())
            stages.append(stage)

        args.measurement_hook = hook
        with patch.object(replay_module, "load_prompt_sources", return_value=(self.sources, {})), patch.dict("os.environ", {"TEST_BENCH_KEY": self.secret}):
            result, code = await replay_module.replay(args)
        self.assertEqual(stages, ["before", "after"])
        self.assertNotIn(self.secret, json.dumps(result))
        return output, result, code

    async def test_multi_token_replay_snapshot_and_comparison(self):
        off, result, code = await self.run_replay("off")
        self.assertEqual(code, 0)
        self.assertEqual(result["summary"]["successful_requests"], 3)
        self.assertEqual(result["summary"]["output_tokens"], 15)
        self.assertEqual(len(self.payloads), 4)
        self.assertTrue(all(payload["max_tokens"] == 5 for payload in self.payloads))
        self.assertTrue(all(value == f"Bearer {self.secret}" for value in self.authorizations))
        self.assertEqual(read_snapshot_prompt(off / "prompt-snapshots", 2), [9, 1, 2, 3, 4])

        on = self.root / "on"
        on.mkdir()
        faster = copy.deepcopy(result)
        faster["summary"]["wall_seconds"] /= 2
        for record in faster["records"]:
            record["ttft_ms"] /= 2
        (on / "client-result.json").write_text(json.dumps(faster))
        comparison = compare_runs([off], [on])
        self.assertAlmostEqual(comparison["on_vs_off_percent"]["input_tps"], 100)
        self.assertAlmostEqual(comparison["on_vs_off_percent"]["ttft_mean_ms"], -50)
        self.assertIsNone(comparison["paired_change_statistics"]["input_tps"]["sample_sd"])
        faster["summary"]["client_concurrency"] = 4
        (on / "client-result.json").write_text(json.dumps(faster))
        with self.assertRaisesRegex(ValueError, "settings"):
            compare_runs([off], [on])
        self.assertFalse(compare_runs([off], [on], allow_parameter_differences=True)["request_settings_match"])

    async def test_request_error_invalidates_the_comparison(self):
        self.fail_request = True
        output, result, code = await self.run_replay("failed")
        self.assertEqual(code, 1)
        self.assertEqual(result["summary"]["errors"], 1)
        self.assertEqual(len((output / "request-records.jsonl").read_text().splitlines()), 3)
        with self.assertRaisesRegex(ValueError, "successful"):
            compare_runs([output], [output])


class OtherTokenizerTests(unittest.TestCase):
    def test_chat_template_receives_only_history_before_target_assistant(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "future answer"},
        ]
        received = []

        class Tokenizer:
            def apply_chat_template(self, history, **settings):
                received.append((history, settings))
                return [10, 20, 30]

        self.assertEqual(render_request(Tokenizer(), messages, 1, []), [10, 20, 30])
        self.assertEqual(received[0][0], messages[:3])
        self.assertTrue(received[0][1]["add_generation_prompt"])
        self.assertFalse(received[0][1]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
