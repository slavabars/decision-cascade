# Results: competitor comparison and quality

**English** · [Русский](../benchmarks.md)

A summary of all project measurements as of November 14, 2026 (v1.7.0; the baseline numbers have not changed since 0.11.0, conformal was added in 1.7). The figures come from real runs on a single 16 GB GPU; methodology, commands and caveats are in [`competitor-benchmark.md`](../competitor-benchmark.md), [`routerbench.md`](../routerbench.md), [`risk-aware-cascade.md`](risk-aware-cascade.md) and [`conformal.md`](conformal.md). Raw JSON and `q` files are attached to the releases (`*-benchmarks.tar.gz`); everything is reproduced with one command, `eval/competitors/reproduce_real.sh`.

The participants are real releases, not mocks:

| Participant | Size | How it was run |
| --- | --- | --- |
| `local` | 124M (gpt2) + heads | our heads, a domain-specific pack |
| `Clef-Flash` | 9.4B | GGUF Q4_K_M in `llama-server` (llama.cpp b11430) |
| `Laya` | 421M (ModernBERT-large) | `Router.predict`, full precision |

## 1. Quality on three datasets

Test splits, n=200 per participant, both competitors in one run:

| Dataset | local | Clef-Flash | Laya | cascade (best) |
| --- | ---: | ---: | ---: | --- |
| `support` — RU tickets (own domain) | **0.915** | 0.667 | 0.500 | **0.949** (11% escalations) |
| AG News — news topics | **0.873** | 0.855 | 0.662 | **0.910** (18%) |
| RouterBench — LLM routing | **0.542** | 0.442 | 0.312 | 0.547 (noise) |

On AG News the pack is trained on 4,000 examples in about 5 minutes; on `support` it comes from a synthetic generator (in-distribution, a pipeline check).

## 2. Latency and size

p50 on the same GPU, same input (`DECISION_MAX_LENGTH=256` for local heads):

| Dataset | local | Clef-Flash | Laya |
| --- | ---: | ---: | ---: |
| `support` | **46 ms** | 306 ms | 13 ms |
| AG News | **10 ms** | 130 ms | 16 ms |
| RouterBench | **21 ms** | 170 ms | 27 ms |

124M local heads against 9.4B: on AG News the same accuracy at 13× lower latency. Laya is faster than us on short text but loses on quality on all three datasets.

## 3. Cascade: three escalation policies

| Policy | `support` | AG News | RouterBench |
| --- | --- | --- | --- |
| no cascade | 0.915 | 0.873 | 0.542 |
| confidence threshold 0.6 | 0.949 / 11.2% | 0.885 / 7.3% | 0.460 / 85% |
| best threshold from `--sweep` | 0.949 / 11.2% | 0.890 / 16.0% | 0.547 / 33% |
| benefit (per-question `q`) | 0.946 / 8.9% | **0.910 / 18.2%** | 0.542 / 0% |

The benefit policy escalates a question only if the external model is more accurate than the local one on it, taking the call cost into account (`p < q − C/(1+D)`), and `q` is refined by feedback. On RouterBench it refuses to pay for a model that is weaker.

## 4. Conformal: a guarantee on the error share

Split conformal on a held-out calibration split plus the exact Clopper–Pearson upper bound; the guarantee is per question (Bonferroni). Public dataset, gpt2, `limit 16000` (2,400 calibration decisions, test 4,000):

| Mode | `alpha` | coverage (calibration) | risk | upper bound | coverage (test) | risk (test) | confidence@coverage | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| soft | 0.10 | 0.915 | 0.0711 | 0.0999 | 0.918 | 0.0785 | 0.0796 | accepted |
| strict | 0.05 | 0.547 | 0.0175 | 0.0482 | 0.534 | 0.0226 | 0.0413 | accepted |

On RouterBench (`limit 8000`, `alpha=0.05`, `min_coverage=0.3`) the certificate honestly reports zero coverage: local accuracy of 0.553 does not allow promising a 5% error share, the gate is rejected, and the cascade falls back to the confidence threshold (`usage.reason = "conformal_gate"`). Details and limitations are in [`conformal.md`](conformal.md).

## 5. Probability calibration

Temperature is fitted by NLL on val and applied only if it improves ECE without hurting Brier (the "do no harm" gate). ECE on test:

| Question | before | after |
| --- | ---: | ---: |
| `route` (RouterBench) | 0.547 | **0.024** |
| `topic` (AG News) | 0.144 | **0.024** |
| `benefit` (RouterBench) | 0.454 | **0.123** |

Calibration does not change the argmax — accuracy stays the same, the meaning of `confidence` changes: it can now be used as a threshold and to compute expected benefit.

## 6. Encoder fine-tuning

Five seeds per configuration, RouterBench test n=1000:

| Variant | mean ± sd |
| --- | --- |
| frozen gpt2 | 0.551 ± 0.040 |
| frozen bert-base | 0.554 ± 0.035 |
| fine-tuned gpt2 | 0.587 ± 0.012 |
| **fine-tuned bert-base** | **0.601 ± 0.019** |

Changing the encoder is noise; fine-tuning on the task gives +5 pp and halves the spread (Mann–Whitney p = 0.016). On AG News fine-tuning gives 0.900 against 0.880 for the pack.

## 7. Where the project is strong (and how that is proven)

1. **Quality per unit of cost.** 124M heads match 9.4B on an in-domain task (0.873 vs 0.855 on AG News) at p50 10 ms vs 130 ms.
2. **Honest probabilities.** ECE 0.547 → 0.024. This is not cosmetic: without calibration the cascade does not escalate at all (`confidence ≈ 1.0`), with it the cascade works.
3. **An escalation policy based on benefit, not magic.** The threshold is derived from the measured accuracy of the external model on a specific question; on AG News this is +2 pp over the best confidence threshold, on RouterBench it avoids harmful calls (0% instead of −2.7 pp).
4. **Determinism and auditability.** One input — the same answer with a probability; every decision is reproducible, and `usage` explains what went outside and why.
5. **Locality.** The model and weights run on your machine, without the network; texts do not go to someone else's API. The weight pack is 540 KB for all domains.
6. **Typed contract.** `Noul / Choice / Score` + `schema_version` + `POST /v1/systemone`: the answer is a structure with probabilities, not text that has to be parsed.
7. **Cheap adaptation.** 4,000 labeled examples and about 5 minutes on a GPU → a pack for your domain; temperature and `q` are calibrated without labels.
8. **Feedback.** `POST /v1/feedback` updates the external model's `q` and (through RLCD) the heads — the policy does not age with the measurement slice.

## 8. Where the project loses (without this section it would be advertising)

- **Zero-shot.** Without labels our heads are random: 0.422 on AG News against 0.855 for Clef-Flash. The project is a trainable layer for a known taxonomy, not a universal classifier.
- **`escalate` on RouterBench.** 0.775 = the constant "always yes" in any configuration: the label is counterfactual and cannot be inferred from the prompt text.
- **Measurement volume.** n=200 on the comparison datasets, n=1000 on the RouterBench/AG News test; confidence intervals are ±2–3 pp. Differences smaller than that are noise.
- **Competitor conditions.** Clef-Flash is 4-bit GGUF (Q8_0 head), Laya and we are in full precision; latencies are comparable only within one table.
- **No production traffic.** All figures are offline benchmarks; behaviour under load, data drift and labeling cost on real domains have not been measured.

## 9. Reproduction

```bash
# the whole comparison: training/calibration of packs, bringing up competitors,
# three datasets, three --sweep runs and the benefit policy
CLEF_GGUF=/path/clef-q4km.gguf \
LLAMA_BIN=/path/llama-server \
LAYA_PYTHONPATH=/path/laya_pkg \
BENEFIT=1 eval/competitors/reproduce_real.sh --specs support,public,routerbench --n 200

# RouterBench: frozen encoder vs fine-tuning
DECISION_DEVICE=cuda DECISION_ENCODER_KIND=encoder \
python -m training.recipe.finetune_encoder --specs routerbench \
    --encoder bert-base-uncased --dim 768 --limit 4000 \
    --steps 300 --lr 5e-5 --seed 0 --out-dir checkpoints/finetune/rb-bert-0
```

Artifacts of each release: `decision_cascade-X.Y.Z-*.whl`, `vX.Y.Z-weights.tar.gz` (packs + temperatures + manifest), `vX.Y.Z-benchmarks.tar.gz` (raw runs, `q` and sha256).

## 10. Nightly regression

Published packs run on pinned splits every night (`.github/workflows/nightly.yml`). The baseline is `eval/nightly_baseline.json`, the tolerance is 1 pp; a drop beyond the tolerance turns the job red.

| Dataset | Pack | n | accuracy | baseline |
| --- | --- | ---: | ---: | ---: |
| AG News | `public-gpt2` | 1000 | 0.8800 | 0.8800 |
| RouterBench | `routerbench-gpt2` | 1000 | 0.5480 | 0.5480 |

Local run:

```bash
python -m eval.nightly --suite all --device cuda
```

The numbers match `metrics.test` from the packs' `metadata.json`: the splits are deterministic (AG News — revision `eb185aa…`, RouterBench — revision `7840214…` and `seed=0`), so the nightly job catches code regressions rather than resampling noise.

The job runs not only on a schedule but also on `push` to `main` when the decision path code or dependencies change (`src/`, `api/`, `eval/`, `training/`, `requirements.txt`, `uv.lock`, `pyproject.toml`), so a regression is visible on the day of the commit.

### Latency budget

Latency from the same run is checked against the budget in `eval/nightly_baseline.json`: `p50`/`p95` with a tolerance factor of `1.25`. The budget has headroom over shared GitHub CPU runners (where the spread is tens of percent: on a slow runner RouterBench p95 reached 1705 ms), so it catches inference degradation rather than load fluctuations:

| Dataset | p50, ms | p95, ms | tolerance | observed (6 runs) |
| --- | ---: | ---: | ---: | --- |
| AG News | 500 | 700 | ×1.3 | p50 313–489, p95 453–693 |
| RouterBench | 1300 | 1750 | ×1.3 | p50 838–1235, p95 1121–1705 |

Modes: `--latency off` (do not measure), `warn` (default: print and record in the report, job stays green) and `fail` (red in the nightly job).

### GPU run

CUDA has its own workflow, `.github/workflows/nightly-gpu.yml`: the same packs and splits, but `--device cuda --latency-profile gpu --latency fail`. GitHub-hosted runners have no GPU, so the job runs on a self-hosted runner labelled `[self-hosted, linux, x64, gpu]`; until the repository variable `NIGHTLY_GPU_ENABLED` is `true`, the job is skipped instead of queueing for a runner that does not exist.

The GPU budget lives in the same `eval/nightly_baseline.json`, under the `gpu_latency` key: CPU numbers are an order of magnitude larger and would hide degradation (a silent fallback to CPU, for example). Measured on an RTX 5060 Ti (float32, `max_length=256`), tolerance ×1.5:

| Dataset | p50, ms | p95, ms | tolerance | observed |
| --- | ---: | ---: | ---: | --- |
| AG News | 50 | 100 | ×1.5 | p50 9, p95 13–14 |
| RouterBench | 80 | 160 | ×1.5 | p50 20–21, p95 25–27 |

Accuracy on GPU matches CPU to the last digit (0.8800 and 0.5480): same float32 and the same weights, only speed changes. The run checks the CUDA path itself — that it works, has not slowed down and is not silently replaced by CPU. Locally:

```bash
python -m eval.nightly --suite all --device cuda --latency-profile gpu --latency fail
```

The runner is registered as a regular self-hosted one (Settings → Actions → Runners → New self-hosted runner) with the `gpu` label; it needs a CUDA torch build, the `gh` CLI and Python 3.11. Without a runner the job is simply skipped — the CPU nightly does not depend on it.
