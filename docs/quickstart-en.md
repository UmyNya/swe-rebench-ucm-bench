# English quickstart

This repository contains the SWE-rebench data and a configurable request client. Install and start vLLM/UCM separately, using the fixed settings in [server-settings.json](../reference/server-settings.json) and the [server guide](server.md). The client never starts, stops or reconfigures a server.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m swe_bench verify --tokens
python -m swe_bench run --workload W24 --request-limit 8 --output-dir results/smoke
```

The baseline has 100 trajectories, 3,379 full-history requests and 295,123,622 input tokens. W24/W32 describe the active trajectory pool; client concurrency defaults to 8. Replay is completion-driven and preserves strict manifest order and same-trajectory dependencies. Generated answers are not appended to subsequent histories.

For a formal pair, start a fresh OFF service with empty KV caches, run the complete workload, stop it and clean up its owned resources. Repeat with a separate ON service and empty caches. The client prepares local prompt snapshots and sends one excluded warmup before measurement.

```bash
python -m swe_bench run --workload W24 --label OFF --output-dir results/w24-off
python -m swe_bench run --workload W24 --label ON --output-dir results/w24-on
python -m swe_bench compare --off results/w24-off --on results/w24-on \
  --output results/comparison.json
```

Labels do not change or verify server settings. The full reference plan is W24 and W32, three OFF/ON pairs each. Perform a short calibration before the formal runs; keep compilation caches but reset KV caches between arms. A single pair is not evidence of multi-round stability.

Request-side parameters include `--concurrency`, `--output-tokens`, `--request-limit`, `--max-model-len`, `--request-timeout`, `--model` and `--base-url`. Use `prepare` for another W, seed, trajectory subset or local tokenizer. A different model needs compatible token IDs; changing only its served name does not retokenize the inputs.

```bash
python -m swe_bench prepare --session-pool 12 --seed 42 \
  --request-limit 200 --output-dir workloads/custom-example
python -m swe_bench run --workload-dir workloads/custom-example \
  --concurrency 4 --output-tokens 8 --output-dir results/custom-example
python -m unittest discover -s tests -v
```

Keep results, prompt snapshots, private instructions, credentials and machine paths local. These artifacts must not be committed. Clean up the GPU service, its shared memory and its own L2 data after use; the client closes only its own HTTP connections.
