#!/usr/bin/env python3
"""Export the fixed W32 replay requests as readable text; no HTTP or model inference."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TORCH='0', USE_TF='0', TOKENIZERS_PARALLELISM='false')
sys.path.insert(0, str(ROOT / 'vendor'))
from benchmark_swe_rebench_open_loop import load_prompt_sources, validate_order
from transformers import AutoTokenizer

def main():
    order_path = ROOT / 'inputs/baseline/causal_w32_noprimer.seed-20260725.order.jsonl'
    order = [json.loads(line) for line in order_path.read_text().splitlines()]
    assert len(order) == 3379
    validate_order(order)
    out = ROOT / 'prompts'
    out.mkdir(exist_ok=True)
    if any(out.iterdir()) or (ROOT / 'requests.jsonl').exists():
        raise FileExistsError('Refusing to overwrite existing export')
    print('Reconstructing 100 fixed trajectories with original replay loader', flush=True)
    sources, summary = load_prompt_sources(order,
        parquet_path=ROOT / 'inputs/source-metadata/trajectories.selected100.sparse-index.parquet',
        tools_path=ROOT / 'inputs/source-metadata/tools.json',
        tokenizer_path=ROOT / 'inputs/tokenizer',
        prompt_salt_prefix='swe-rebench-w32-noprimer-seed-20260725')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'inputs/tokenizer', local_files_only=True)
    fields = ['ordinal', 'text_file', 'trajectory_id', 'request_id', 'depends_on', 'repo', 'instance_id', 'request_index', 'body_tokens', 'sent_tokens']
    started = time.monotonic()
    total_bytes = 0
    with (ROOT / 'requests.jsonl').open('x', encoding='utf-8') as metadata, (ROOT / 'requests.csv').open('x', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i, event in enumerate(order):
            assert event['ordinal'] == i
            ids = sources[event['trajectory_id']].prompt_for(int(event['request_index']))
            text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            data = text.encode('utf-8')
            relative = f'prompts/{i:04d}.txt'
            with (ROOT / relative).open('xb') as handle:
                handle.write(data)
            row = {key: event.get(key) for key in fields}
            row.update(text_file=relative, body_tokens=event['prompt_tokens'], sent_tokens=len(ids))
            writer.writerow(row)
            metadata.write(json.dumps({**event, **row, 'text_bytes': len(data), 'text_sha256': hashlib.sha256(data).hexdigest()}, ensure_ascii=False) + '\n')
            total_bytes += len(data)
            if (i + 1) % 200 == 0 or i == 0 or i == len(order) - 1:
                print(f'exported={i+1}/3379 text_bytes={total_bytes} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    summary.update(request_count=len(order), text_bytes=total_bytes, order_sha256=hashlib.sha256(order_path.read_bytes()).hexdigest(), source_ucm_commit='d6a465446ae4676f357754f4a19e98bfa4e0e66e', text_role='decoded exact replay input token IDs; special tokens retained; use source loader for exact token replay')
    (ROOT / 'EXPORT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    assert len(list(out.glob('*.txt'))) == 3379

if __name__ == '__main__':
    main()
