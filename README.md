# llm-checker

**Find the best local LLM that actually runs on your hardware.**

Auto-detects your GPU/CPU/RAM and ranks the top models from HuggingFace that fit your system.

[日本語版はこちら](docs/README.ja.md)

```
$ llm-checker

╭─────────────── Hardware Info ───────────────╮
│ GPU 0: NVIDIA GeForce RTX 4060 — 8.0 GB    │
│ CPU: Intel i7-11700K — 4 cores (AVX2)       │
│ RAM: 23.5 GB                                │
╰─────────────────────────────────────────────╯

  #  Model                         Quant    VRAM   Speed    Score
  1  gemma-3-12b-it-GGUF           Q4_K_M  8.0 GB  21 tok/s  63.1
  2  Qwen2.5-7B-Instruct-GGUF     Q4_K_M  5.5 GB  32 tok/s  61.3
  3  gemma-2-9b-it-GGUF            Q4_K_M  6.5 GB  26 tok/s  61.0
  ...
```

## Install

### pipx (recommended)

```bash
pipx install git+https://github.com/yourname/local-llm-checker.git
```

### pip

```bash
pip install git+https://github.com/yourname/local-llm-checker.git
```

### With NVIDIA GPU detection

```bash
pipx install "local-llm-checker[nvidia] @ git+https://github.com/yourname/local-llm-checker.git"
```

### Development

```bash
git clone https://github.com/yourname/local-llm-checker.git
cd local-llm-checker
uv sync --dev
uv run llm-checker
```

## Usage

```bash
# Auto-detect hardware and show best models
llm-checker

# Simulate a GPU (e.g. planning a purchase)
llm-checker --gpu "RTX 4090"
llm-checker --gpu "RTX 5090"

# CPU-only mode
llm-checker --cpu-only

# More results / filters
llm-checker --top 20
llm-checker --quant Q4_K_M
llm-checker --min-speed 30

# JSON output
llm-checker --json

# Force refresh (ignore cache)
llm-checker --refresh

# Show hardware info only
llm-checker hardware
```

## Scoring

Each model gets a score from 0 to 100.

| Factor | Points | Description |
|--------|--------|-------------|
| Model size | 0-40 | Larger models generally produce better output |
| Benchmark | 0-10 | Arena ELO / Open LLM Leaderboard scores |
| Speed | 0-20 | Higher tok/s = more practical to use |
| Source trust | -5 to +5 | Official repos get a bonus, repackagers get a penalty |
| Popularity | 0-3 | Downloads and likes as tiebreaker |

Score markers:
- **`~`** (yellow) — No direct benchmark yet. Score estimated from the model family
- **`?`** (yellow) — No benchmark data available

## How it works

### Data pipeline

1. Fetches ~900 popular models from **HuggingFace API** (text-generation, GGUF, multimodal)
2. Fetches benchmark scores from **Chatbot Arena ELO** and **Open LLM Leaderboard**, normalized to 0-100
3. All data cached for 24 hours at `~/.cache/local-llm-checker/`

### Ranking engine

1. **Hardware detection** — GPU (NVIDIA/AMD/Apple Silicon), CPU, RAM, disk
2. **VRAM estimation** — model size + quantization + KV cache overhead
3. **Compatibility check** — Full GPU / Partial Offload / CPU-only classification
4. **Speed estimation** — tok/s based on GPU memory bandwidth
5. **Scoring** — combines size, benchmark, speed, source trust, and popularity
6. **Deduplication** — merges GGUF variants and version differences into model families

### Project structure

```
local_llm_checker/
├── cli.py              # Typer CLI entry point
├── constants.py        # GPU bandwidth tables, quantization constants
├── hardware/           # Hardware detection (NVIDIA, AMD, Apple, CPU, RAM)
│   └── gpu_simulator.py  # GPU simulation for --gpu flag
├── models/
│   ├── fetcher.py      # HuggingFace API model fetcher
│   ├── benchmark.py    # Benchmark scores (Arena + Leaderboard)
│   ├── grouper.py      # Model family grouping and dedup
│   └── cache.py        # JSON cache
├── engine/
│   ├── vram.py         # VRAM requirement estimation
│   ├── compatibility.py # Hardware compatibility check
│   ├── performance.py  # Inference speed estimation
│   └── ranker.py       # Scoring and ranking
└── output/
    └── display.py      # Rich table output
```

## Requirements

- Python 3.11+
- NVIDIA GPU detection requires `nvidia-ml-py` (install with `[nvidia]` extra)
- AMD / Apple Silicon detected automatically

## License

MIT
