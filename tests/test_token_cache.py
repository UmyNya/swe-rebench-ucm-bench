"""Exercise another real tokenizer, Parquet input, token cache and snapshot."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from swe_bench.snapshots import read_snapshot_prompt, save_prompt_snapshot
from swe_bench.token_cache import load_token_cache, prepare_token_cache, render_request


class TokenCacheTests(unittest.TestCase):
    def test_other_tokenizer_roundtrip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = {word: index for index, word in enumerate([
                "[UNK]", "user", "assistant", "first", "second", "past", "future",
            ])}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
            tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
            tokenizer.save_pretrained(root / "tokenizer")
            messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "past"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "future"},
            ]
            pq.write_table(pa.Table.from_pylist([{"trajectory_id": "example", "trajectory": messages}]), root / "data.parquet")
            (root / "tools.json").write_text("[]")
            profiles = [{"trajectory_id": "example", "source_row": 0, "steps": [{"request_index": 0}, {"request_index": 1}]}]
            storage = prepare_token_cache(
                root, profiles, tokenizer_path=root / "tokenizer", parquet_path=root / "data.parquet",
                tools_path=root / "tools.json", salt_prefix="test",
            )
            # JSON roundtrip exercises conversion of integer request keys to strings.
            storage = json.loads(json.dumps(storage))
            events = [
                {**step, "ordinal": index, "request_id": f"example:request-{index}", "trajectory_id": "example"}
                for index, step in enumerate(profiles[0]["steps"])
            ]
            sources, _ = load_token_cache(root, storage, events)
            source = sources["example"]
            for index in (0, 1):
                expected = source.salt_ids + render_request(tokenizer, messages, index, [])
                self.assertEqual(source.prompt_for(index), expected)
                self.assertNotIn(vocabulary["future"], expected)
            self.assertIn(vocabulary["past"], source.prompt_for(1))
            save_prompt_snapshot(root / "snapshot", events, sources, max_model_len=128, output_tokens=1)
            self.assertEqual(read_snapshot_prompt(root / "snapshot", 1), source.prompt_for(1))
            metadata = storage["trajectories"]["example"]
            metadata["sha256_u32le"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_token_cache(root, storage, events)


if __name__ == "__main__":
    unittest.main()
