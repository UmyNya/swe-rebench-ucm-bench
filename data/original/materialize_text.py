#!/usr/bin/env python3
"""Recreate all 3379 plain-text requests from losslessly deduplicated text sources."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (ROOT / 'TEXT_INDEX.jsonl').read_text().splitlines()]
assert len(rows) == 3379
assert [r['ordinal'] for r in rows] == list(range(3379))
sources = {}
(ROOT / 'prompts').mkdir(exist_ok=True)
for i, row in enumerate(rows):
    source = row['source_file']
    if source not in sources:
        sources[source] = (ROOT / source).read_bytes()
    data = sources[source][:row['text_bytes']]
    assert len(data) == row['text_bytes']
    assert hashlib.sha256(data).hexdigest() == row['text_sha256']
    path = ROOT / row['text_file']
    assert path.parent == ROOT / 'prompts'
    if path.exists():
        assert path.read_bytes() == data, f'Existing text differs: {path}'
    else:
        with path.open('xb') as f:
            f.write(data)
    if (i + 1) % 500 == 0:
        print(f'restored={i+1}/3379', flush=True)
print('restored=3379/3379; all text SHA256 values match', flush=True)
