# whichllm

[![PyPI version](https://img.shields.io/pypi/v/whichllm)](https://pypi.org/project/whichllm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**找到真正能在你的硬件上运行的最佳本地 LLM。**

自动检测你的 GPU/CPU/RAM，从 HuggingFace 中筛选并排名适合你硬件的最佳本地大语言模型。

[English](../README.md) | [日本語](README.ja.md)

## 国内用户

HuggingFace 在国内无法直接访问。whichllm 支持通过镜像站使用：

### 方式一：`--mirror` 参数（推荐）

```bash
whichllm --mirror cn
```

### 方式二：环境变量 `WHICHHF_MIRROR`

```bash
export WHICHHF_MIRROR=cn
whichllm
```

### 方式三：`HF_ENDPOINT`（完全自定义）

```bash
export HF_ENDPOINT=https://hf-mirror.com
whichllm
```

> **说明**：`--mirror cn` 会同时切换模型 API、数据集服务器和模型下载的域名，是最省心的方式。`HF_ENDPOINT` 是 HuggingFace 生态通用变量，仅切换主 API 域名，适合已有代理或需要自定义镜像地址的场景。
>
> 基准数据源（Artificial Analysis、Aider）不在 HuggingFace 域名下，无法通过镜像切换，但这些源失败不会影响基本使用，工具会使用内置的冻结回退数据。

## 快速开始

免安装，一次性运行：

```bash
uvx whichllm@latest
```

模拟指定 GPU（购买前评估）：

```bash
uvx whichllm@latest --gpu "RTX 4090"
```

经常使用则安装：

```bash
uv tool install whichllm
uv tool upgrade whichllm  # 更新
```

其他安装方式：

```bash
brew install andyyyy64/whichllm/whichllm
pip install whichllm
```

## 常用工作流

安装后直接运行 `whichllm`。一次性使用时，用 `uvx whichllm@latest` 替代 `whichllm`。

```bash
# 本机最佳模型
whichllm

# 模拟指定 GPU
whichllm --gpu "RTX 4090"

# 比较升级候选
whichllm upgrade "RTX 4090" "RTX 5090" "H100"

# 反向查询：运行某模型需要什么 GPU
whichllm plan "llama 3 70b"

# 一键下载并聊天
whichllm run "qwen 2.5 1.5b gguf"

# 输出可粘贴的 Python 代码
whichllm snippet "qwen 7b"

# JSON 输出供脚本使用
whichllm --top 1 --json
```

## 输出示例

```
$ whichllm --gpu "RTX 4090"

#1  Qwen/Qwen3.6-27B     27.8B  Q5_K_M   score 92.8    27 t/s
#2  Qwen/Qwen3-32B       32.0B  Q4_K_M   score 83.0    31 t/s
#3  Qwen/Qwen3-30B-A3B   30.0B  Q5_K_M   score 82.7   102 t/s
```

32B 模型完全能塞进 RTX 4090，但 whichllm 把 27B 排第一——因为它的真实基准分数更高、代数更新。这正是 whichllm 与"只看大小"工具的核心区别。

## 各硬件推荐速查

实际排名（2026-05 快照——你的结果跟踪**实时** HuggingFace 数据，这不是静态列表）：

| 硬件 | 显存 | 首推模型 | 速度 |
|------|------|----------|------|
| RTX 5090 | 32 GB | `Qwen3.6-27B` Q6_K (score 94.7) | ~40 t/s |
| RTX 4090 / 3090 | 24 GB | `Qwen3.6-27B` Q5_K_M (score 92.8) | ~27 t/s |
| RTX 4060 | 8 GB | `Qwen3-14B` Q3_K_M (score 71.0) | ~22 t/s |
| Apple M3 Max | 36 GB | `Qwen3.6-27B` Q5_K_M (score 89.4) | ~9 t/s |
| 纯 CPU | — | `gpt-oss-20b` (MoE) Q4_K_M (score 45.2) | ~6 t/s |

`whichllm --gpu "<你的显卡>"` 可在购买前模拟任意 GPU。

## 为什么选 whichllm？

把模型塞进显存只是简单部分。真正的难题是：**在所有能跑的模型里，哪个才是最好的？** 这就是 whichllm 要解决的。

- **证据驱动排名，不是大小启发式** — 首选模型基于合并的真实基准测试（LiveBench、Artificial Analysis、Aider、多模态/视觉、Chatbot Arena ELO、Open LLM Leaderboard），而非"恰好能塞下的最大模型"。

- **时效性感知** — 过时基准分数会沿模型谱系降权，2024 模型无法靠旧分数超越新一代。基准快照日期显示在每次排名下方，过时推荐一目了然。

- **证据分级与防护** — 每个分数标记为 `direct` / `variant` / `base` / `interpolated` / `self-reported` 并按置信度打折。虚假的上传者声称和跨家族继承（小分支借用更大基座的分数）会被主动拒绝。

- **架构感知估算** — VRAM = 权重 + GQA KV 缓存 + 激活 + 开销；速度基于带宽，考虑了量化效率、后端因子、MoE 活跃/总参数比、统一内存 vs 独立显卡部分卸载建模。

- **一条命令，可脚本化** — `whichllm` 打印答案；加 `--json | jq` 可接入管道。无需 TUI，无需记忆快捷键。

- **实时数据** — 模型直接从 HuggingFace API 拉取，离线或限流时使用冻结回退数据。

## 功能特性

- **硬件自动检测** — NVIDIA、AMD、Apple Silicon、纯 CPU
- **智能排名** — 综合显存适配、速度和基准质量评分
- **一键对话** — `whichllm run` 自动下载并启动交互聊天
- **代码片段** — `whichllm snippet` 输出可直接运行的 Python 代码
- **实时数据** — 直接从 HuggingFace 拉取（带缓存）
- **基准感知** — 集成真实评估分数，带置信度衰减
- **任务配置** — 按通用、编程、视觉、数学场景过滤
- **GPU 模拟** — 测试任意 GPU：`whichllm --gpu "RTX 4090"`
- **硬件规划** — 反向查询：`whichllm plan "llama 3 70b"`
- **升级规划** — 比较当前硬件与候选 GPU
- **JSON 输出** — 管道友好：`whichllm --json`

## 运行与代码片段

一条命令即可试用任意模型。无需手动安装——whichllm 通过 `uv` 创建隔离环境、安装依赖、下载模型、启动交互聊天。

```bash
# 与模型聊天（自动选择最佳 GGUF 变体）
whichllm run "qwen 2.5 1.5b gguf"

# 自动选择本机最佳模型并聊天
whichllm run

# 仅 CPU 模式
whichllm run "phi 3 mini gguf" --cpu-only
```

支持**所有模型格式**：

- **GGUF** — 通过 `llama-cpp-python`（轻量快速）
- **AWQ / GPTQ** — 通过 `transformers` + `autoawq` / `auto-gptq`
- **FP16 / BF16** — 通过 `transformers`

获取**可粘贴的 Python 代码片段**：

```bash
whichllm snippet "qwen 7b"
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
)

print(output["choices"][0]["message"]["content"])
```

## 使用方法

```bash
# 自动检测硬件并显示最佳模型
whichllm

# 模拟 GPU（如购买前评估）
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090"

# 指定变体
whichllm --gpu "RTX 5060 16"

# 仅 CPU 模式
whichllm --cpu-only

# 更多结果 / 过滤
whichllm --top 20
whichllm --quant Q4_K_M
whichllm --min-speed 30
whichllm --evidence base   # 允许 id/基座模型匹配
whichllm --evidence strict # 仅 id 精确匹配（同 --direct）
whichllm --direct

# JSON 输出
whichllm --json

# 强制刷新（忽略缓存）
whichllm --refresh

# 仅显示硬件信息
whichllm hardware

# 规划：运行特定模型需要什么 GPU
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768

# 升级：比较当前硬件与候选 GPU
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
whichllm upgrade "Apple M4 Max" --top 5

# 运行：下载并即时聊天
whichllm run "qwen 2.5 1.5b gguf"
whichllm run                       # 自动选择本机最佳

# 代码片段：输出可直接运行的 Python 代码
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

## 与 Ollama 集成

使用 JSON 输出将 HuggingFace ID 映射到本地 Ollama 模型名：

```bash
# 获取最佳模型的 HuggingFace ID
whichllm --top 1 --json | jq -r '.models[0].model_id'

# 获取最佳编程模型 ID
whichllm --profile coding --top 1 --json | jq -r '.models[0].model_id'
```

Ollama 模型名与 HuggingFace 仓库 ID 不总是一致，通常需要一个映射步骤再 `ollama run`。

添加 Shell 别名：

```bash
alias bestllm='whichllm --top 1 --json | jq -r ".models[0].model_id"'
# 使用：ollama run $(bestllm)
```

## 评分体系

每个模型获得 0-100 的综合分数。基准质量和模型规模是核心；证据置信度和运行时适配度缩放；速度、来源可信度、热度作为调整。

| 因子 | 效果 | 说明 |
|------|------|------|
| 基准质量 | 核心 | 合并 LiveBench / Artificial Analysis / Aider / Vision / Arena ELO / Open LLM Leaderboard，按来源置信度加权 |
| 模型规模 | 最高 +35 | `log2` 缩放的知识量代理（MoE 用总参数） |
| 量化 | 乘性惩罚 | 低 bit 量化折扣 |
| 证据置信度 | ×0.55–1.0 | self-reported ×0.55, inherited ×0.78, direct 全分 |
| 运行时适配 | ×0.50–1.0 | partial-offload ×0.72, CPU-only ×0.50 |
| 速度 | -8 ~ +8 | 可用性门槛对比适配的 tok/s 下限 |
| 来源可信度 | -5 ~ +5 | 官方组织加分，已知转载者扣分 |
| 热度 | 同分决胜 | 下载量/点赞数 |

分数标记：
- **`~`**（黄色）— 无直接基准；分数继承/插值自模型家族
- **`!sr`**（亮黄色）— 仅上传者自报基准，未经独立验证
- **`?`**（红色）— 无基准数据

速度标记（`--status` 中）：
- **`~`**（黄色）— 有估算 tok/s 范围
- **`?`**（红色）— 低置信速度估算；后端/运行时敏感性高

## 文档

- [CLI 参考](cli.md)
- [工作原理](how-it-works.md)
- [评分机制](scoring.md)
- [硬件检测与模拟](hardware.md)
- [运行与代码片段](run-snippet.md)
- [故障排查](troubleshooting.md)

## 开发

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --dev
uv run whichllm
uv run pytest
```

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 许可

MIT
