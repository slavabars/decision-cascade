# decision-cascade

[![CI](https://github.com/slavabars/decision-cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/slavabars/decision-cascade/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/decision-cascade)](https://pypi.org/project/decision-cascade/)
[![Python](https://img.shields.io/pypi/pyversions/decision-cascade)](https://pypi.org/project/decision-cascade/)
[![Coverage gate](https://img.shields.io/badge/coverage_gate-%E2%89%A590%25-brightgreen)](https://github.com/slavabars/decision-cascade/blob/main/pyproject.toml)
[![License](https://img.shields.io/pypi/l/decision-cascade)](https://github.com/slavabars/decision-cascade/blob/main/LICENSE)

[Русский](https://github.com/slavabars/decision-cascade/blob/main/README.md) · **English**

A local, machine-native decision engine in the spirit of **Jev (TypeSafe AI)** — open source, running entirely on your hardware, with the full loop "train → calibrate → evaluate → deploy".

Instead of asking an LLM to generate an answer, the model **decides**: a state becomes features, and decision heads return **typed, probabilistic, calibrated** answers — Noul (yes/no), Choice (pick an option), Score (value on a scale) — without generating a single token.

**Status: stable (1.5.0).** The API contract (`schema_version = "1"`), the pack format and the meaning of `DECISION_*` variables do not break inside `1.x`: changes land only in a major, with a two-minor deprecation window. What exactly is promised, which surfaces are still beta and how deprecation is announced — see [`docs/stability.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/stability.md).

## Demo: two games

The engine does not generate an answer as text — it **decides**: it proposes candidates and chooses when to pay for a deeper search. Games show the whole loop, and the result is measured in points rather than labels.

| Chess vs Stockfish | Pac-Man-style maze |
| :---: | :---: |
| [![Chess: the head decides when to think deeper](https://raw.githubusercontent.com/slavabars/decision-cascade/main/examples/games/chess/demo.gif)](https://github.com/slavabars/decision-cascade/blob/main/examples/games/chess/README.md) | [![Maze: the head decides when to play it safe](https://raw.githubusercontent.com/slavabars/decision-cascade/main/examples/games/maze/demo.gif)](https://github.com/slavabars/decision-cascade/blob/main/examples/games/maze/README.md) |
| 19% escalations close half the gap to "always deep" at 38% of its cost | 94% of the cautious policy's score at 33% escalations; twice the greedy score |
| [code, data and numbers](https://github.com/slavabars/decision-cascade/blob/main/examples/games/chess/README.md) | [code, data and numbers](https://github.com/slavabars/decision-cascade/blob/main/examples/games/maze/README.md) |

Both games are ours: map, environment and rules are written in this repository, no third-party assets. The heads learn from outcomes, not from hints: in chess, when the engine's fast search is unreliable; in the maze, when a greedy step will cost a life.

## Quick start (2 minutes)

You need Python 3.11+ (CI runs 3.11, 3.12 and 3.13) and network access only to install the package: no model download, no GPU, no trained weights — the server starts on a `hash` encoder.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install decision-cascade
python -m uvicorn api.server.main:app --port 8000
```

In a second terminal, one request with all three question types:

```bash
curl -sS -X POST http://127.0.0.1:8000/decisions \
  -H 'content-type: application/json' \
  -d '{
    "state": "Stripe checkout has been failing for 3 days, we are losing revenue",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is the customer reporting urgency?"},
      "team": {"type": "choice", "instructions": "Which team should take it?",
               "options": ["billing", "integration", "sales", "support"]},
      "severity": {"type": "score", "instructions": "How upset is the customer?",
                   "levels": ["calm", "mild", "angry", "furious"]}
    }
  }'
```

The response is typed decisions with probabilities (real output, unabridged):

```json
{
  "schema_version": "1",
  "decisions": {
    "urgent": {"type": "noul", "noul": 0.5031822323799133, "confidence": 0.00636446475982666},
    "team": {
      "type": "choice",
      "choice": "support",
      "probabilities": {"billing": 0.251, "integration": 0.211, "sales": 0.248, "support": 0.290},
      "confidence": 0.29038938879966736
    },
    "severity": {
      "type": "score",
      "score": 1.486896499991417,
      "probabilities": {"calm": 0.255, "mild": 0.240, "angry": 0.268, "furious": 0.237},
      "confidence": 0.26786985993385315,
      "legend": ["calm", "mild", "angry", "furious"]
    }
  },
  "trained": false
}
```

The near-uniform probabilities and `trained: false` are honest: the encoder is `hash` and the heads are untrained. The point of this step is the shape of the response and the fact that everything runs locally and deterministically. Meaningful numbers require a trained pack (~0.5 GB encoder):

```bash
gh release download v1.5.0 --pattern '*-weights.tar.gz' \
  --repo slavabars/decision-cascade
tar -xzf v1.5.0-weights.tar.gz

DECISION_CHECKPOINT_DIR=v1.5.0-weights/public-gpt2 \
DECISION_MODEL_ID=openai-community/gpt2 DECISION_FEATURE_DIM=768 \
  python -m uvicorn api.server.main:app --port 8000
```

`public-gpt2` is trained on AG News: 0.880 on the test split. See the table below.

## Benchmarks on public datasets

Same question set, one run, n=200, one 16 GB GPU. Ours is 124M (gpt2 + heads trained for the domain); the competitors are real releases: `Clef-Flash` 9.4B (GGUF Q4_K_M in `llama-server`) and `Laya` 421M (ModernBERT-large).

| Public dataset | decision-cascade 124M | Clef-Flash 9.4B | Laya 421M |
| --- | ---: | ---: | ---: |
| AG News — accuracy | **0.873** | 0.855 | 0.662 |
| AG News — p50 | **10 ms** | 130 ms | 16 ms |
| RouterBench — accuracy | **0.542** | 0.442 | 0.312 |
| RouterBench — p50 | **21 ms** | 170 ms | 27 ms |

On AG News the local heads match a model 75× larger while being 13× faster. On RouterBench the gap is smaller: the dataset is counterfactual, absolute numbers are low for everyone (the trivial "always yes" constant scores 0.775), so the comparison is between participants, not against 1.0.

Two things people usually ask next:

- **cascade**: +3.4 pp on `support` and +3.7 pp on AG News at 11–18% escalation rate; on RouterBench the benefit policy refuses to pay for a model that is weaker than the local one;
- **calibration**: temperature scaling drops ECE from 0.144 to 0.024 on AG News and from 0.547 to 0.024 on RouterBench without changing accuracy; on the Russian dataset from 0.176 to 0.022.

The synthetic `support` domain (0.915 vs 0.667 and 0.500), three escalation policies, reliability diagrams and caveats are in [`docs/benchmarks.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/benchmarks.md) (Russian); methodology and `--sweep` in [`docs/competitor-benchmark.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/competitor-benchmark.md). Reproduce with one command: `eval/competitors/reproduce_real.sh`. Raw JSON ships in `*-benchmarks.tar.gz` of every release.

## Why this is different

- **No generation.** The whole path is `encode → heads → decide`; latency is bounded and there is no token loop to run away. gpt2 on GPU: 9–25 ms p50 per request depending on input truncation, 0.2 ms per state in a batch on a small encoder ([`docs/latency.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/latency.md)).
- **Probabilities that mean something.** Temperature scaling is fitted on a held-out slice and accepted only if ECE and Brier do not get worse (`accept_calibration`). `confidence` is a probability, not a logit in disguise.
- **Typed answers.** Noul / Choice / Score with a JSON schema (`schema_version = "1"`), validated on the server and in the SDK. Downstream code switches on types instead of parsing prose.
- **Cascade instead of a bigger model.** Local heads answer what they know; only uncertain questions escalate to a remote model. With `DECISION_CASCADE_RISK_BUDGET` escalation is driven by an error budget (`P(mistake)` model) instead of a raw confidence threshold.
- **Feedback loop.** `POST /v1/feedback` collects labelled decisions, retrains the risk model per question and re-fits temperature; the "do not harm" gate compares against a control run before accepting the new pack.

## Use it from Python

```python
from api.sdk.client import DecisionClient

with DecisionClient("http://127.0.0.1:8000") as client:
    body = client.decisions(
        "Stripe checkout has been failing for 3 days, we are losing revenue",
        {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should take it?",
                "options": ["billing", "integration", "sales", "support"],
            },
        },
    )
    print(body["decisions"]["urgent"]["noul"])
```

The response is validated against the schema before you see it (`validate_response`), so a contract break fails loudly instead of silently.

## Train your own heads

```bash
# public AG News: train/val/test with a real test split
.venv/bin/python -m training.recipe.train_heads --specs public \
  --encoder openai-community/gpt2 --dim 768 --limit 4000 \
  --train-steps 500 --out-dir checkpoints/packs/public-gpt2

# calibrate probabilities on val (accuracy unchanged, ECE down)
.venv/bin/python -m training.recipe.calibrate_temperature --specs public \
  --encoder openai-community/gpt2 --dim 768 --limit 4000 \
  --checkpoint-dir checkpoints/packs/public-gpt2
```

Russian sentiment (`--specs ru_sentiment`, rubert-tiny2, 0.708 vs 0.584 majority baseline) — [`docs/ru-benchmark.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/ru-benchmark.md). Feedback retraining with an automatic re-calibration and a "do not harm" gate — [`docs/drift.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/drift.md).

## Deploy

```bash
sudo bash deploy/ubuntu/install.sh   # or deploy/centos/install.sh
```

Installs a venv, a systemd template unit (`decision-api@api`) and reads `DECISION_*` from `.env` (`EnvironmentFile=-<project>/.env`; without the file the service runs on defaults). Production variable set with comments: [`.env.example`](https://github.com/slavabars/decision-cascade/blob/main/.env.example). `GET /info` shows what was actually applied (`config`: encoder, device, dtype, truncation, batch, packs). Upgrade, rollback, backup and restore — [`docs/deploy.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/deploy.md).

## Documentation

The contract, deployment, troubleshooting and risk-aware cascade docs are available in English. The measurement-heavy pages remain in Russian for now; the API, CLI and code comments are self-explanatory, and this page covers the essentials.

| Doc | What is there |
| --- | --- |
| [`docs/en/stability.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/en/stability.md) | what `1.x` promises, beta surfaces, deprecation policy |
| [`docs/en/risk-aware-cascade.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/en/risk-aware-cascade.md) | error-budget cascade, hard gate, `risk.json`, reference numbers |
| [`docs/en/deploy.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/en/deploy.md) | Ubuntu/CentOS install, upgrade, backup, rollback |
| [`docs/en/troubleshooting.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/en/troubleshooting.md) | error codes and known failure modes |
| [`docs/benchmarks.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/benchmarks.md) (Russian) | competitor comparison, cascade policies, calibration |
| [`docs/latency.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/latency.md) (Russian) | batching, input truncation, dtype budgets (CPU and GPU) |
| [`docs/ru-benchmark.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/ru-benchmark.md) (Russian) | Russian sentiment benchmark |
| [`docs/observability.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/observability.md) (Russian) | metrics, alerts, log format |
| [`docs/release.md`](https://github.com/slavabars/decision-cascade/blob/main/docs/release.md) (Russian) | release process (version → tag → PyPI → GitHub) |

The Russian originals of the translated pages are in [`docs/`](https://github.com/slavabars/decision-cascade/tree/main/docs); the risk-aware cascade page there also carries the full measurement log.

## Tests

```bash
.venv/bin/python -m pytest                     # 842 passed, 14 skipped (no GPU)
.venv/bin/python -m pytest --cov=src --cov=api  # coverage gate 90% (pyproject)
```

CI runs ruff, shellcheck, `uv lock --check`, the full test suite with the coverage gate, a wheel smoke test, and a nightly regression on the released packs (quality deltas and a latency budget).

## License

Apache-2.0. `SystemOne` is an external protocol (Jev / TypeSafe AI); the names are kept as-is for compatibility.
