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

---

## LinkedIn Post

> Copy-paste ready

---

Most people think LLM inference is one thing. It's actually two — and they're complete opposites.

I ran benchmarks on a T4 GPU with Qwen2.5-3B to prove it.

---

**The numbers don't lie:**

Prefill phase (processing your prompt):
→ 2,128 tokens/sec at 1,024 tokens
→ GPU compute is maxed out
→ All tokens processed **in parallel** — pure matrix multiply

Decode phase (generating the response):
→ 18.6 tokens/sec
→ Memory bandwidth is the bottleneck
→ One token at a time — **115x slower per token**

Same GPU. Same model. 115x throughput difference.

---

**Why does this gap exist?**

Prefill is **compute-bound**: you feed the full prompt as one big matrix multiply. The GPU loves this — FLOPs saturated.

Decode is **memory-bound**: every new token requires reading the entire KV cache from GPU memory. The GPU's compute cores sit idle, waiting on memory reads.

Running both on the same GPU means neither phase runs at peak efficiency. It's like using a Ferrari for both highway racing and parking — you're always compromising.

---

**The solution: Disaggregation**

Split the two phases onto purpose-built hardware:

- **Prefill Server** → High-FLOP GPU (H100 SXM) — 990 TFLOPS
- **Decode Server** → High-bandwidth GPU (H100 NVL) — optimized memory BW
- Transfer the KV cache between them (~1.5 GB for a 7B model at 4K context)

Real-world results from published research:
- **Splitwise** (Microsoft, 2023): 1.4x throughput gain
- **DistServe** (PKU, 2024): 2x improvement in SLO attainment

---

**Why it matters at scale**

If you're serving millions of requests/day, 1.4x throughput = significant cost reduction.

Different workloads also have different ratios:
- Long prompt + short output → need more prefill capacity
- Short prompt + long output → need more decode capacity

Disaggregation lets you scale each independently.

The engineering challenge? That KV cache transfer. At 4K context on a 7B model, you're moving ~1.5 GB per request across GPUs. That needs NVLink or InfiniBand to not become the new bottleneck.

---

Full benchmark notebook + code:
https://github.com/shubh2579/Prefill-Decode-Segregation-Experiment

Papers:
- Splitwise: arxiv.org/abs/2311.18677
- DistServe: arxiv.org/abs/2401.09670
- TetriInfer: arxiv.org/abs/2401.11181

*Measured on Tesla T4 (15.6 GB VRAM), Qwen/Qwen2.5-3B-Instruct, float16*

\#LLM #MachineLearning #MLInfrastructure #AIEngineering #DeepLearning #LLMInference #GPUOptimization
