# SWE-rebench UCM 性能测试客户端

向已经启动的 vLLM 服务回放 SWE-rebench 完整历史请求，测量吞吐和首 token 延迟。仓库包含原始数据包、W24/W32 固定输入及原实验的服务配置，也支持调整请求侧参数。vLLM、UCM 和模型由使用者自行安装、配置和启动。

原实验使用 100 条真实软件工程轨迹、每轮 3,379 条请求、295,123,622 个输入 token。每条请求包含当时已有的完整历史；本次生成的回答不会追加到后续请求。

## 安装客户端

使用 Python 3.12：

```bash
git clone https://github.com/UmyNya/swe-rebench-ucm-bench.git
cd swe-rebench-ucm-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m swe_bench verify --tokens
```

客户端只需要 CPU。`verify --tokens` 校验随附文件，并逐条重建、核对原实验全部输入 token 的 SHA-256；省略 `--tokens` 可只检查文件和工作负载元数据。

`requirements.lock.txt` 固定验收环境的全部客户端依赖；`requirements.txt` 列出直接依赖。服务端依赖由使用者另行管理。

## 运行固定负载

先按 [服务参考配置](docs/server.md) 启动兼容服务。以下地址是示例；`--base-url` 使用服务根地址，不包含 `/v1`。

```bash
# 检查参数和工作负载，不发送请求，也不创建结果目录。
python -m swe_bench run --workload W24 \
  --base-url http://127.0.0.1:18147 --model glm52-mixed \
  --output-dir results/w24-check --dry-run

# 连通性检查：只发送原顺序前 8 条请求。
python -m swe_bench run --workload W24 --request-limit 8 \
  --output-dir results/smoke

# 完整单组测量：服务需已配置为对应的 OFF 状态。
python -m swe_bench run --workload W24 --label OFF \
  --output-dir results/w24-r1-off
```

每次使用新的输出目录。客户端先准备精确输入和本地提示词快照，再发送一次不计入测量的短预热，最后进行正式回放。`--label OFF` 只是结果注释，不会切换服务器的预取状态。

W24/W32 的 **W 是交错的活跃轨迹池大小**；客户端并发 **C 默认是 8**。客户端按固定顺序发送请求，等待同轨迹前一条请求完成，并在并发槽位释放后发送下一条。W、C 和服务端 `max_num_seqs` 是三个不同参数。

## 原实验的 OFF/ON 对比

[复现计划](configs/reproduction-plan.json) 包含 W24、W32 各三轮 OFF/ON。每一组都由使用者重启服务并准备独立、空的 KV 缓存；小规模检查用过的服务也应重启后再做正式测量。

1. 使用对应的短负载预编译、预热运行环境，保留编译缓存；正式测量窗口内不应发生新 GPU 编译。
2. 启动全新的 OFF 服务和空 L2 目录，运行完整 OFF 负载，然后停止该服务并清理其所属缓存。
3. 启动全新的 ON 服务和另一空 L2 目录，运行相同负载，然后停止该服务并清理其所属缓存。
4. 比较结果；按相同方法完成其他轮次和 W32。

```bash
# 以下命令各自在相应服务就绪后执行。
python -m swe_bench run --workload W24 --label OFF --output-dir results/w24-r1-off
python -m swe_bench run --workload W24 --label ON  --output-dir results/w24-r1-on
python -m swe_bench compare \
  --off results/w24-r1-off --on results/w24-r1-on \
  --output results/w24-r1-comparison.json
```

`compare` 默认要求请求设置和精确输入一致，并拒绝含请求错误的结果。多轮汇总可给 `--off` 和 `--on` 分别传入按轮次排列的多个目录。吞吐使用总 token 数除以总测量时间，延迟按逐请求数据汇总。服务配置、缓存冷启动和资源独占情况需由服务使用者核验；客户端不会根据标签推断这些条件已经满足。

## 调整请求参数

```bash
# 调整客户端并发和每条请求的输出长度。
python -m swe_bench run --workload W24 --concurrency 4 --output-tokens 16 \
  --output-dir results/c4-output16

# 生成新的 W/seed 和因果前缀子集，不改动固定基线。
python -m swe_bench prepare --session-pool 12 --seed 42 \
  --trajectory-limit 20 --request-limit 200 --output-dir workloads/custom-w12
python -m swe_bench run --workload-dir workloads/custom-w12 \
  --concurrency 4 --output-dir results/custom-w12

# 使用其他模型时，先用它的本地 tokenizer 重新生成输入。
python -m swe_bench prepare --tokenizer "$TOKENIZER_PATH" \
  --session-pool 12 --trajectory-limit 20 --output-dir workloads/custom-model
python -m swe_bench run --workload-dir workloads/custom-model \
  --model "$SERVED_MODEL_NAME" --max-model-len 131072 \
  --output-dir results/custom-model
```

其他 tokenizer 需要可用的聊天模板，并支持这些历史消息及工具定义。准备过程可能较长，但位于测量之外。生成 token IDs 后，客户端检查服务返回的输入 token 数；只修改 `--model` 名称并不能让不同词表的模型使用原 GLM token IDs。

还可调整 `--base-url`、`--request-timeout`、`--request-limit` 和客户端上下文限制 `--max-model-len`。这些选项不修改服务端配置。需要认证时，在 `SWE_BENCH_API_KEY` 环境变量中提供凭据，或用 `--api-key-env` 指定变量名。故意比较不同参数时，`compare --allow-parameter-differences` 会保留并列出差异。

## 文件与隐私

| 目录 | 用途 |
| --- | --- |
| `swe_bench/` | 请求准备、调度、传输、快照和比较模块 |
| `workloads/W24-*`、`workloads/W32-*` | 原实验的固定请求顺序及 profiles |
| `inputs/` | 当前回放使用的轨迹、tools 和 GLM tokenizer |
| `data/original/` | 原始导出包、无损去重文本及查看工具 |
| `configs/`、`reference/` | 固定服务配置、版本、复现计划及输入哈希 |
| `results/` | 本地结果和提示词快照，已加入 Git 忽略规则 |

原始导出包使用较早的 seed 和聊天模板。精确复现使用顶层 `workloads/` 与 `inputs/`；不要将原始导出的 TXT 重新编码并当作相同输入。数据包的机器私有路径已从元数据和默认值中清理，公开数据集正文保持原样。

每次测量在本地保留 `client-result.json`、`request-records.jsonl` 和 `prompt-snapshots/`。快照存储去重 token 序列和请求切片，能够离线恢复实际发送的每条输入：

```bash
python -m swe_bench snapshot results/w24-r1-off/prompt-snapshots \
  --ordinal 0 --output results/request-0-tokens.json
```

运行结果、提示词快照、个人指令、机器路径和认证信息都属于本地材料，不能提交到 GitHub。发布前应运行仓库的发布检查，并检查 Git 历史。客户端关闭自己的 HTTP 连接；服务的进程、GPU、共享内存和 L2 资源由服务使用者在每组实验后按所有权清理。

```bash
python scripts/check_publication.py --staged --history
```

此检查覆盖已暂存文件和所有可达历史。公开数据集正文中的原始路径仅在文件哈希完全匹配时豁免；机器元数据、提示词快照和凭据没有该豁免。检查工具只报告命中的文件及规则，不输出私有值。

## 测试与来源

```bash
python -m unittest discover -s tests -v
```

请求调度和 token 重建源自 UCM 的 SWE-rebench 回放工具，模块拆分保留了原实验语义。来源和改动说明见 [client-source.json](reference/client-source.json)。代码采用 MIT 许可；上游声明见 [licenses/UCM-MIT.txt](licenses/UCM-MIT.txt)。数据来自 [SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)，固定 revision `35455389ab51bf5e2306bfd436ef72d0f98bf882`，保留 CC BY 4.0 数据许可及 tokenizer 原许可。

English instructions: [Quickstart](docs/quickstart-en.md).
