# Prefill vs Decode: The 115x Gap

> Post 9 in a series on LLM inference internals — benchmarking the two fundamentally different phases of LLM inference and why splitting them onto separate hardware improves throughput by 1.4–2x.

## What this is

LLM inference has two phases with **opposite hardware bottlenecks**:

| Phase | What it does | Bottleneck | Measured speed |
|-------|-------------|-----------|----------------|
| **Prefill** | Processes the full input prompt | GPU compute (FLOPs) | 2,128 tok/s |
| **Decode** | Generates tokens one by one | Memory bandwidth | 18.6 tok/s |

Running both on the same GPU means neither phase runs at peak efficiency. The notebook benchmarks this gap directly — **115x difference per token** — and explains why disaggregation (splitting the phases onto different GPUs) is the production solution used at scale.

## Hardware used

- Tesla T4 (15.6 GB VRAM) on Google Colab Pro
- Model: `Qwen/Qwen2.5-3B-Instruct` (float16)

## Notebook contents

| Section | What it covers |
|---------|---------------|
| 1. Prefill benchmark | Throughput vs context length (128–4096 tokens) |
| 2. Decode benchmark | Throughput vs generation length (32–256 tokens) |
| 3. GPU utilization | Compute vs memory bottleneck comparison |
| 4. Charts | Visual comparison of both phases |
| 5. Theory | Why disaggregation works + key papers |
| 6. Summary | Full results table |
| 7. LinkedIn post | Ready-to-publish write-up |

## Key results

```
Prefill at 1024 tokens:  2128 tok/s  (COMPUTE-bound)
Decode at 128 tokens:      18.6 tok/s  (MEMORY-bound)
Gap:                        115x
```

## The solution: Disaggregation

```
User Request
    │
    ▼
┌─────────────────┐
│  Prefill Server  │  ← Compute-optimized GPU (H100 SXM, high FLOPs)
└────────┬────────┘
         │  KV Cache Transfer (NVLink / InfiniBand)
         ▼
┌─────────────────┐
│  Decode Server   │  ← Memory-optimized GPU (H100 NVL, high BW)
└────────┬────────┘
         ▼
      Response
```

**Real-world gains:**
- [Splitwise](https://arxiv.org/abs/2311.18677) (Microsoft, 2023): 1.4x throughput
- [DistServe](https://arxiv.org/abs/2401.09670) (PKU, 2024): 2x SLO attainment improvement
- [TetriInfer](https://arxiv.org/abs/2401.11181) (2024): dynamic scheduling between phases

## Run it yourself

```bash
# Open in Colab (recommended — needs GPU)
# Or locally with a CUDA GPU:
pip install transformers accelerate matplotlib pandas tqdm
jupyter notebook "Post9_Prefill_Decode (1).ipynb"
```

## Papers

- **Splitwise**: Patel et al., 2023 — [arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)
- **DistServe**: Zhong et al., 2024 — [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)
- **TetriInfer**: 2024 — [arxiv.org/abs/2401.11181](https://arxiv.org/abs/2401.11181)
