# whatllm

**あなたのPCで動く最強のローカルLLMを見つけるCLIツール。**

GPU/CPU/RAMを自動検出し、HuggingFaceの人気モデルからハードウェアに合った最適なモデルをランキング表示します。

[English version](../README.md)

```
$ whatllm

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

## インストール

### pipx (推奨)

```bash
pipx install whatllm
```

### pip

```bash
pip install whatllm
```

### NVIDIA GPU を使う場合

```bash
pipx install "whatllm[nvidia]"
```

### 開発用

```bash
git clone https://github.com/yourname/local-llm-checker.git
cd local-llm-checker
uv sync --dev
uv run whatllm
```

## 使い方

```bash
# 自動検出して最適なモデルを表示
whatllm

# GPU をシミュレート (購入検討時など)
whatllm --gpu "RTX 4090"
whatllm --gpu "RTX 5090"

# CPU のみモードで実行
whatllm --cpu-only

# 結果を増やす / フィルタ
whatllm --top 20
whatllm --quant Q4_K_M
whatllm --min-speed 30

# JSON 出力
whatllm --json

# キャッシュを無視して再取得
whatllm --refresh

# ハードウェア情報だけ表示
whatllm hardware
```

## スコアの見方

各モデルに 0〜100 のスコアが付きます。

| 要素 | 配点 | 説明 |
|------|------|------|
| モデルサイズ | 0-40 | パラメータ数が大きいほど高品質 |
| ベンチマーク | 0-10 | Arena ELO / Open LLM Leaderboard のスコア |
| 推論速度 | 0-20 | tok/s が高いほど実用的 |
| ソース信頼度 | -5〜+5 | 公式 org はボーナス、リパッケージはペナルティ |
| 人気度 | 0-3 | ダウンロード数・いいね数 |

スコア横のマーカー:
- **`~`** (黄色) — ベンチマークがまだ公開されていない新しいモデル。同シリーズの推定値を使用
- **`?`** (黄色) — ベンチマークデータなし

## 仕組み

### データ取得

1. **HuggingFace API** から人気テキスト生成モデル + GGUF + マルチモーダルを取得 (計 ~900 モデル)
2. **Chatbot Arena ELO** と **Open LLM Leaderboard** からベンチマークスコアを取得・正規化
3. 24時間キャッシュ (`~/.cache/whatllm/`)

### ランキングエンジン

1. **ハードウェア検出** — GPU (NVIDIA/AMD/Apple), CPU, RAM, ディスクを自動検出
2. **VRAM 見積もり** — モデルサイズ + 量子化 + KV キャッシュから必要 VRAM を計算
3. **互換性チェック** — Full GPU / Partial Offload / CPU-only の判定
4. **推論速度推定** — GPU メモリ帯域幅ベースの tok/s 推定
5. **スコア計算** — サイズ、ベンチマーク、速度、信頼度を総合評価
6. **ファミリー重複排除** — 同じモデルの GGUF バリアントやバージョン違いを統合

### プロジェクト構成

```
src/whatllm/
├── cli.py              # typer CLI エントリポイント
├── constants.py        # GPU 帯域幅テーブル、量子化定数
├── hardware/           # ハードウェア検出 (NVIDIA, AMD, Apple, CPU, RAM)
│   └── gpu_simulator.py  # --gpu フラグ用 GPU シミュレータ
├── models/
│   ├── fetcher.py      # HuggingFace API からモデル取得
│   ├── benchmark.py    # ベンチマークスコア取得 (Arena + Leaderboard)
│   ├── grouper.py      # モデルファミリー分類・重複排除
│   └── cache.py        # JSON キャッシュ
├── engine/
│   ├── vram.py         # VRAM 必要量の推定
│   ├── compatibility.py # ハードウェア互換性チェック
│   ├── performance.py  # 推論速度推定
│   └── ranker.py       # スコア計算・ランキング
└── output/
    └── display.py      # Rich テーブル表示
```

## 動作環境

- Python 3.11+
- NVIDIA GPU 検出には `nvidia-ml-py` が必要 (`[nvidia]` extra)
- AMD / Apple Silicon は自動検出

## ライセンス

MIT
