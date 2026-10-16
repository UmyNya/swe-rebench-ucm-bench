# SWE-rebench 3379 条请求

本目录用于查看固定 W32 数据集的全部请求内容。3379 条请求来自 100 条编程任务轨迹（80 个问题、65 个仓库），每条输入包含当次调用前的完整历史：问题描述、模型历史回答、工具调用及返回内容。它们不是 3379 条独立的简短问题。

## 文件

| 文件 | 内容 |
| --- | --- |
| `prompts/0000.txt` ～ `prompts/3378.txt` | 3379 条完整输入文本，按 W32 请求顺序编号，保留特殊 token 和会话标识符 |
| `requests.csv` | 请求索引，含文本路径、仓库、问题、轨迹、依赖关系和 token 数，可用表格软件打开 |
| `requests.jsonl` | 逐请求完整元数据，每行对应一条请求，并记录文本 SHA256 |
| `TRAJECTORIES_100.md` | 100 条轨迹的来源及选取轮次 |
| `inputs/baseline/` | 原始 W32 order、profiles、manifest 等固定输入 |
| `inputs/source-metadata/` | 原始选中轨迹 Parquet、tools、数据卡和许可证 |
| `inputs/tokenizer/` | 原实验 GLM tokenizer 和 chat template |
| `vendor/` | 原回放和数据构建代码 |
| `export_prompts.py` | 离线文本导出脚本，复用原回放器重建请求，不发送网络请求 |
| `EXPORT.json` | 导出数量、长度统计及来源信息 |
| `text_sources/`、`TEXT_INDEX.jsonl`、`materialize_text.py` | 无损去重文本及还原脚本，已在目标目录还原全部 TXT；仅需 Python 标准库 |
| `SHA256SUMS` | 文件完整性校验 |

## 查看内容

```bash
cd data/original
less prompts/0000.txt
less requests.csv
```

文件编号和 `ordinal` 从 0 开始。通过 `requests.csv` 的 `repo`、`instance_id`、`trajectory_id` 筛选感兴趣的代码仓库或会话，再打开对应文本文件。`depends_on` 表示同一会话的上一条请求。

每条输入正文约 64K～122K tokens，因此文本较长。实际回放发送的是 token ID 数组；这里的 TXT 是这些 token IDs 的解码结果，保留模板特殊 token。需要精确回放时应使用所附原始数据、tokenizer 和回放器，不能把 TXT 重新编码当作已验证的等价输入。

输入开头的会话标识符用于隔离不同轨迹的公共前缀，同一轨迹所有请求保持一致。固定 W32 的交错参数为 W=32、seed=20260725。每次请求只生成 1 token，生成结果不追加到下一条历史。

原始数据：[nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)，revision `35455389ab51bf5e2306bfd436ef72d0f98bf882`。许可证和来源信息见 `inputs/source-metadata/`。
