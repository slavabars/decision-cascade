# Chess: the engine proposes, the heads decide

**English** · [Русский](README.md)

A game against Stockfish in which **our trained head decides when the fast search cannot be trusted**. This is not "we taught a model to play chess": the engine plays stronger, and our layer is responsible for something else — the cost of a decision.

![Demo: a game with escalations only where the head considers the fast answer unreliable](demo.gif)

*18 seconds from a real recording: `need_deep` highlights uncertain positions, and those get the deep search. The game in this recording is 1-0, 41 plies, 6 escalations (30%), 103 ms per our move.*

## Why this is here

The project is about typed decisions with honest probabilities. Chess provides what text examples lack: a **closed loop** where every decision is checked by the outcome, not by labels.

The roles are separated explicitly:

| Who | What it does |
| --- | --- |
| Stockfish (`MultiPV 3`) | proposes the three best moves using the fast search |
| the `need_deep` head (trained on our data) | answers: is the fast answer reliable |
| the cascade | at `need_deep ≥ threshold` pays for the deep search, otherwise takes the fast move |

A move is a typed decision (`Choice` from the candidates), and `need_deep` is a `Noul` with a probability that can be compared against the cost of an error. Exactly the same as in tickets: the cheap path by default, the expensive one on uncertainty.

## Installation

```bash
# Stockfish — the official binary from GitHub, no sudo
bash examples/games/chess/fetch_engine.sh          # -> .tools/stockfish

# board and move legality (optional extra, not part of the main install)
.venv/bin/pip install "decision-cascade[games]"

# for GIF recording (optional): asciinema and agg binaries
# https://github.com/asciinema/asciinema/releases
# https://github.com/asciinema/agg/releases
```

## The full loop

```bash
# 1. data: engine self-play games, the label is whether the fast answer
#    disagrees with the deep one. ~1600 positions, 10-15 minutes on 8 cores.
.venv/bin/python -m examples.games.chess.dataset \
  --engine .tools/stockfish --out-dir data/chess

# 2. heads: the project's standard recipe, no separate trainer
.venv/bin/python -m training.recipe.train_heads --specs chess --dataset data/chess \
  --encoder hash --dim 256 --train-steps 600 --out-dir checkpoints/packs/chess

# 3. temperature: so that `need_deep` is a probability, not a logit
.venv/bin/python -m training.recipe.calibrate_temperature --specs chess \
  --dataset data/chess --encoder hash --dim 256 \
  --checkpoint-dir checkpoints/packs/chess

# 4. evaluation: quality against the cost of deep searches
.venv/bin/python -m examples.games.chess.evaluate \
  --pack checkpoints/packs/chess --data data/chess

# 5. a game
.venv/bin/python -m examples.games.chess.play \
  --engine .tools/stockfish --pack checkpoints/packs/chess --skill 0

# 6. record the demo as a GIF
ENGINE=.tools/stockfish PACK=checkpoints/packs/chess \
  bash examples/games/chess/record.sh
```

## What counts as a metric

`need_deep` predicts the event "the engine's fast answer is unreliable": the deep search found a different move or shifted the evaluation by more than 50 centipawns. This event is computed by the engine itself, so labeling needs no human and does not depend on our chess strength.

Then there are two numbers that matter:

- **quality** — the share of positions where the played move matched the deep answer;
- **cost** — the average time per move.

The escalation threshold is the knob between them. "Always deep" gives quality 1.0 at full cost; "always fast" is cheap but errs where the position is tactical.

## Results

Dataset: 40 engine self-play games, **1600 positions** (960/320/320 train/val/test), the "needs deep" share is 17%. Training — 600 steps, a `hash` encoder with 256 features, temperature fitted on val (the "do no harm" gate accepted it).

The head predicts a rare event, so accuracy is almost useless: 0.822 against a base rate of 0.83. Look at AUC and calibration:

| Metric (test, n=320) | Value |
| --- | ---: |
| ROC-AUC | **0.819** |
| Brier | 0.117 |
| ECE (10 bins) | **0.032** |
| Share of positions where the fast move matched the deep one | 0.83 |

Calibration is not decoration here: `need_deep ≈ 0.3` really means "the fast answer will fail in roughly a third of such positions", so the threshold can be chosen by the cost of an error rather than by eye.

## Cost against quality

`evaluate.py` on the same held-out split (fast search 60 ms, deep 260 ms):

| Threshold | Escalations | Quality | ms/move |
| ---: | ---: | ---: | ---: |
| always fast | 0% | 0.828 | 60 |
| 0.4 | 8% | 0.878 | 77 |
| **0.3** | **19%** | **0.912** | **98** |
| 0.2 | 42% | 0.963 | 143 |
| 0.1 | 55% | 0.988 | 169 |
| always deep | 100% | 1.000 | 260 |

Quality is the share of positions where the played move matched the deep search answer.

The default threshold (0.3) closes **half the gap** between "always fast" and "always deep" while spending **38% of the full deep search cost**. That is the product story of the cascade in one number: not "stronger than the engine", but "almost the same reliability for less money".

## Limitations

- This is **not** chess training: the engine provides candidates, and the head only learns to predict the reliability of the fast answer. Playing strength is limited by the engine.
- The `hash` encoder over discrete features is a deliberately simple baseline. It competes with "always deep" only on cost, not on quality; there is nothing to look at with `gpt2` here — the state is not textual in meaning.
- The features are built not to leak: they contain nothing from the deep search, only what is known before the decision (material, phase, spread of candidate evaluations, checks and captures).
- One game in the demo is an illustration, not a measurement. The measurement is `evaluate.py` on the held-out split.
