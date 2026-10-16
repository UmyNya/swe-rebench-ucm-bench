# 原实验服务参考配置

使用者自行安装 vLLM、UCM、CUDA 依赖和模型，客户端连接已经就绪的服务。完整设置见 [server-settings.json](../reference/server-settings.json)。这里给出复现固定配置，不提供服务管理或参数搜索功能。

## 固定版本和模型

- UCM：[`codex/ucm-deferred-pin-wait-v0.8.0`](https://github.com/StorTecDev/ucm-prefetch/tree/codex/ucm-deferred-pin-wait-v0.8.0)，固定 commit `520428c4d2eaa23cb12e4d47d70a7bb820f8d4e0`，使用分支内已包含的 READY 和层间同步补丁。
- vLLM `0.28.0`，构建 commit `2cf0a6915`；PyTorch `2.13.0+cu130`、CUDA `13.0`、服务端 Transformers `5.17.0`、Python `3.12.3`。
- 模型：`zai-org/GLM-5.2-FP8`，revision `f33c6dc501ee5a2c7e35155653b1b1abbc320951`。原实验运行前 14 层、原 embedding、最终 norm 和输出 head，关闭 next-token prediction。使用者准备相应权重目录并设置 `MODEL_PATH`。
- 原硬件：单张 NVIDIA L20X，显存约 140 GiB，驱动 `580.105.08`；共享内存 32 GiB，SSD 支持直接 I/O 和 Linux AIO。不同硬件的绝对吞吐不应视为严格等价。

客户端依赖与服务端依赖分开安装。不要因为客户端使用另一个 Transformers 版本，就覆盖服务端环境。

## 手动启动示例

从仓库根目录执行。先设置 `MODEL_PATH` 为自己准备的模型目录，`SERVER_PYTHON` 为服务端环境的 Python。每一组选择不同的、空的 L2 路径；配置示例中的路径相对于服务工作目录。

```bash
export ENABLE_UCM_PATCH=1
export GLM51_LAYERWISE_SYNC_FIX=1
export CUDA_VISIBLE_DEVICES=0

# OFF 组；ON 组改为 configs/ucm-on.json。
export UCM_CONFIG_FILE=configs/ucm-off.json
export ENGINE_ID=swe-w24-r1-off

# 不复用已有目录；若 off 已存在，先确认上一组服务已停止并清理其所属数据。
mkdir -p local/l2
mkdir local/l2/off

KV_TRANSFER_CONFIG=$(python - <<'PY'
import json
import os

print(json.dumps({
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "engine_id": os.environ["ENGINE_ID"],
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": os.environ["UCM_CONFIG_FILE"]},
}))
PY
)

"$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 18147 \
  --model "$MODEL_PATH" --served-model-name glm52-mixed --seed 20260911 \
  --hf-overrides '{"num_hidden_layers":14,"num_nextn_predict_layers":0}' \
  --speculative-config null \
  --data-parallel-size 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --enable-expert-parallel --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 8192 --max-num-seqs 16 --max-model-len 131072 \
  --gpu-memory-utilization 0.92 --kv-cache-memory-bytes 5372160000 \
  --kv-cache-dtype auto --safetensors-load-strategy lazy --no-async-scheduling \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
```

`GLM51_LAYERWISE_SYNC_FIX=1` 是指定 UCM 提交提供的同步补丁开关。其名字包含 GLM51，但该提交明确包含本实验使用的 GLM52/UCM08 适配。默认保持自动 block size 和图执行；原环境得到 block size 64、5000 个 blocks、编译模式 3 和 `FULL_AND_PIECEWISE` CUDA Graph。

OFF/ON 配置都使用 `Cache|Posix`、19 GiB L1、128 GiB L2、layerwise 和 event sync。两组的处理差异是 `enable_early_prefetch`，L2 目录和 engine ID 则各自隔离。不要在一个已有热 KV 缓存的服务上直接改标签比较。

ON 组相应创建 `local/l2/on`。多轮实验也可以复制配置到被忽略的 `local/` 目录，为每一组填写新的 L2 目录和 engine ID。

## 校准、正式测量与清理

W24 可先用前 192 条请求校准，W32 使用前 256 条。保留编译缓存，随后重启服务并换用空 KV/L2 目录，正式发送完整 3379 条。若正式窗口出现新 GPU 编译、请求错误或其他任务争用，应保留异常记录，查明原因后重跑该组。

每轮 OFF/ON 完成后，由启动服务的人停止相应进程，确认 GPU 已释放，再清理这一组拥有的共享内存和 L2 目录。不要使用针对整台机器的进程杀除或缓存清理命令。客户端退出不会停止外部服务。

本仓库的结果比较覆盖请求、token 输入、因果顺序、并发和端到端性能。原实验的细粒度 KV 来源归因、内部调度观测等需要额外的服务侧观测；客户端不会声称已验证不可见的服务内部状态。
