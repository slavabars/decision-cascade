# Latency budget: batching, input length and dtype

**English** · [Русский](../latency.md)

What a decision costs and which knobs actually speed it up — measured with the `eval.latency.sweep` sweep, not by eye. The numbers come from one machine (Ubuntu 24.04): the CPU is shared, the GPU is an NVIDIA RTX 5060 Ti (driver 595.99.02, CUDA 13.2). Absolute CPU values are noisy; the conclusions are about orders of magnitude and parameter choices.

## 1. How we measure

```bash
.venv/bin/python -m eval.latency.sweep \
  --encoder cointegrated/rubert-tiny2 --dim 312 --kind encoder --device cpu \
  --batch-sizes 1,8,16,32,64 --lengths 32,64,128,256 \
  --dtypes float32,bfloat16,float16 --states 64 --n 10 --repeats 3
```

Three important details of the methodology:

- **every repeat encodes new states.** `DecisionService` caches features by prompt text; with the same text the second measurement shows latency without the encoder (that is what happened in the first version of the sweep — 33,000 states/s);
- **input length is measured on a long state** (~700 tokens), otherwise truncation to 32 and to 256 tokens gives the same time;
- batching is measured through `decide_many` (the `/decisions/batch` path), while length and dtype are measured through a single `decide` with different states.

## 2. Batching

`DECISION_ENCODE_BATCH` is the encoder batch size. On short states (3 questions per state):

| batch | rubert CPU | rubert GPU | gpt2 CPU | gpt2 GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.5 | 2.6 | 267.5 | 13.0 |
| 8 | **1.7** | 0.7 | 179.6 | 9.1 |
| 16 | 1.9 | 0.4 | 175.6 | **8.8** |
| 32 | 1.7 | 0.3 | **175.2** | 9.1 |
| 64 | **1.6** | **0.2** | — | — |

Milliseconds per state in `decide_many` (3 questions). Conclusion: a batch of `>= 8` gives **2.7×** on the small model and **1.5×** on gpt2 on CPU; on GPU the gain is even larger — **12×** for `rubert-tiny2` (2.6 → 0.2 ms) and 1.5× for gpt2. For `/decisions/batch` this is free: `DECISION_ENCODE_BATCH=16` is a reasonable default, `32–64` if memory allows.

## 3. Input length

`DECISION_MAX_LENGTH` is the main lever on CPU: time is almost linear in tokens.

| max_length | rubert CPU | rubert GPU | gpt2 CPU | gpt2 GPU |
| ---: | ---: | ---: | ---: | ---: |
| 32 | **6.2** | — | — | — |
| 64 | 7.0 | **3.1** | **189.3** | **9.4** |
| 128 | 17.6 | 3.4 | 303.3 | 16.3 |
| 256 | 15.0 | 3.9 | 434.8 | 24.8 |

p50 for a single request, ms. (For rubert, 128 and 256 on CPU are within noise — the model hits the length of the actual state, not the limit.)

Conclusion: truncating `256 → 64` saves **2.3×** on gpt2 (CPU and GPU) and up to **2.5×** on the small model. If the task does not need long context, `DECISION_MAX_LENGTH=64` or `128` is the cheapest way to stay within budget.

For comparison with production numbers: gpt2 on GPU at `max_length=256` is 24.8 ms p50 for a single request with three questions; that is the same order as in [`benchmarks.md`](benchmarks.md) (10–46 ms).

## 4. dtype

| dtype | rubert CPU | rubert GPU | gpt2 CPU | gpt2 GPU |
| --- | ---: | ---: | ---: | ---: |
| float32 | 13.5 | 3.8 | — | 24.3 |
| float16 | not supported | **3.0** | not supported | **9.6** |
| bfloat16 | **10.3** | **2.9** | — | **9.5** |

p50 for a single request, ms. Head accuracy on 256 examples (float32 → fp16 / bf16):

| Pack | float32 | float16 | bfloat16 |
| --- | ---: | ---: | ---: |
| AG News (`public-gpt2`) | 0.8776 | 0.8763 | 0.8763 |
| Russian (`ru-rubert-tiny2`) | 0.7240 | 0.7240 | 0.7266 |

- **GPU: `float16`/`bfloat16` give 2.5×** on gpt2 (24.3 → 9.5 ms) and 1.3× on the small model, with accuracy changing within ±0.3 pp (on 256 examples that is noise). For GPU deployment this is the cheapest win.
- **CPU: `bfloat16` gives −24% p50** with no accuracy change on the checked sample; `float16` is not supported on CPU — `src/device.py` fails honestly with a hint instead of silently computing in float32.
- The numbers are a sanity check on 256 examples, not a guarantee: verify accuracy on your domain before switching.

## 5. int8: measured, not adopted

A prototype of `torch.ao.quantization.quantize_dynamic` on rubert-tiny2 (CPU, 64 texts):

| | float32 | dynamic int8 |
| --- | ---: | ---: |
| encoding time | 393.4 ms | **291.2 ms** (1.35×) |
| embedding cosine | — | 0.99586 |
| head accuracy (64 examples) | 0.6719 | 0.6510 |

Decision: **do not adopt**. A 1.35× speedup does not pay for a 2 pp loss on a small sample and for supporting an API that PyTorch marks as deprecated (it recommends `torchao`/pt2e). If int8 is needed, that is a separate task with accuracy validation on the full test and a backend choice.

## 6. What follows from this

- CPU + `rubert-tiny2`: batch 8–16, `max_length=64–128`, `bfloat16` — single-digit milliseconds per state in a batch and tens of milliseconds for a single request.
- CPU + gpt2: ~175 ms/state in a batch and 190–435 ms for a single request depending on truncation. This matches the nightly regression (p50 300–450 ms for three questions), so a `p95 < 1 s` budget on CPU is achievable only with truncation and batching.
- GPU is an order of magnitude faster (p50 9–25 ms for gpt2 depending on truncation), and `float16`/`bfloat16` is the mandatory choice for GPU deployment. The nightly CPU budget and the production budget are different numbers; they must not be confused.

## 7. Reproduction

```bash
# rubert-tiny2 on CPU: batches, length, dtype
.venv/bin/python -m eval.latency.sweep \
  --encoder cointegrated/rubert-tiny2 --dim 312 --kind encoder --device cpu \
  --batch-sizes 1,8,16,32,64 --lengths 32,64,128,256 \
  --dtypes float32,bfloat16,float16 --states 64 --n 10 --repeats 3 \
  --out data/latency/rubert.json --markdown data/latency/rubert.md

# gpt2 on CPU
.venv/bin/python -m eval.latency.sweep \
  --encoder openai-community/gpt2 --dim 768 --kind causal --device cpu \
  --batch-sizes 1,8,16,32 --lengths 64,128,256 --states 64 --n 10 --repeats 3

# the same on GPU, including dtype
DECISION_DEVICE=cuda .venv/bin/python -m eval.latency.sweep \
  --encoder openai-community/gpt2 --dim 768 --kind causal --device cuda \
  --dtypes float32,float16,bfloat16 --states 64 --n 20 --warmup 3
```
