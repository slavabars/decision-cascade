# Risk-aware cascade: escalation by the probability of an error

**English** · [Русский](../risk-aware-cascade.md)

> **Contract.** Since `1.1` the risk-aware cascade is stable: environment variables, `/info` keys, the `usage.reason` dictionary, the `/v1/feedback` response and hard gate behaviour are fixed in [`stability.md` §7](stability.md#7-risk-aware-cascade-contract-stable-since-11) and checked by `tests/test_risk_contract.py`. This page is about measurements and the reasoning behind the defaults, not about a changing API.

A threshold cascade escalates when `confidence` is below a threshold. The problem is that **confidence is not the probability of a correct answer**: a head can be 0.9 confident and still be wrong in half of such cases. A threshold of 0.6 does not mean "60% correct" and gives no way to state a quality requirement.

The risk-aware cascade reframes the problem as **selective prediction** (delayed decision): we accept locally those decisions whose estimated `P(error)` does not exceed a given **error budget**, and send the rest to the external model.

## How it works

1. **Decision features** (`src/risk.py::decision_features`). The base three are `confidence, entropy, margin`: the maximum of the distribution, the normalised entropy and the gap to the runner-up. Plus three added from measurements (see below): `log_odds_gap` — `ln(p1/p2)` of the top two outcomes (for `k >= 3` this is exactly the gap of raw logits before softmax), `n_classes` — the number of outcomes, `noul_logit` — the signed `noul` logit (the base three are symmetric under `p <-> 1-p`, so the sign is lost in them). For `noul` the distribution is `[p, 1-p]`. Models trained on the three older features keep working: features are taken **by name** (`LEGACY_FEATURE_NAMES`), extras are dropped.
2. **The `P(error)` estimator** (`ErrorRiskModel`) is a logistic regression over these features. Training lives in `training/recipe/risk.py` (on a held-out set). At serving time it is computed with `numpy`; there are no heavy dependencies on the inference path.
3. **Escalation** (`src/backends/cascade.py::RiskAwareCascadeBackend`): a decision goes to the external model if `predict_error(decision) > risk_budget`. The budget is set by `DECISION_CASCADE_RISK_BUDGET`.

The threshold cascade does not go away: it remains the default behaviour, and the risk-aware mode is enabled only with an explicit budget — backward compatibility is preserved.

## Metrics

Selective metrics live in `eval/calibration/selective.py` and do not depend on the model:

- **risk–coverage curve** — the error rate among accepted decisions at each coverage level (the most reliable decisions are accepted first);
- **AURC** (area under the risk–coverage curve) — the coverage/risk trade-off as one number, lower is better;
- **eAURC** — the excess over the oracle (an ideal ranker gives 0);
- **coverage@risk** — how many decisions can be accepted while staying within the error budget.

The key honest comparison is **at equal cost**: with the same share of requests sent to the external model (equal coverage), which method accepts a more accurate subset of decisions.

## How to measure

```bash
.venv/bin/python -m eval.risk.run \
  --specs routerbench --encoder openai-community/gpt2 --dim 768 \
  --train-steps 500 --limit 4000 --out-dir data/risk_routerbench

# A/B over the feature set (same train/val/test, only the features change)
.venv/bin/python -m eval.risk.features \
  --specs routerbench --encoder openai-community/gpt2 --dim 768 \
  --train-steps 500 --limit 4000 --seeds 0 1 2

# plus a transfer check: train without one question and measure on it (--loo)
.venv/bin/python -m eval.risk.features \
  --specs routerbench --encoder openai-community/gpt2 --dim 768 \
  --train-steps 500 --limit 4000 --seeds 0 1 2 --loo

# plus the few-shot curve on a new question
.venv/bin/python -m eval.risk.features \
  --specs routerbench --encoder openai-community/gpt2 --dim 768 \
  --train-steps 500 --limit 4000 --seeds 0 1 2 --shots 0 5 20 100
```

`eval.risk.run` trains the heads on train, fits the risk estimator on **validation** and compares on test: local decisions, a threshold cascade (a grid of thresholds) and the risk-aware cascade (a grid of budgets). The fallback is treated as ideal, so the measurement is about **routing**, not about a particular external model. Artifacts: `risk.json` (the estimator itself) and `metrics.json` (all numbers). At the end the hard gate verdict is printed (`gate (risk vs confidence): ACCEPTED/REJECTED`) and written into `risk.json` — see "Hard gate".

`eval.risk.features` isolates the contribution of features: the same heads, the same split, only the risk model's feature set changes. The `production` variant is literally what `training.recipe.risk` trains, and its numbers match `risk_model` from `eval.risk.run` (checked bit-for-bit). The `--loo` flag additionally checks transfer to a **new** question — the main caveat for every number in the "all questions" column.

## Reference numbers

All numbers use `--limit 4000`, seed 0, `--train-steps 500`, `eval.risk.run` (measured at the previous default `C=1.0`; with the current `C=0.2` in-dist changes within ±0.001, and transfer only gets better). Local accuracy (test) is ≈ **0.58** on RouterBench with both encoders: the dataset is hard, almost half of the decisions are wrong, so a "small" error budget requires almost full escalation.

Accuracy among accepted decisions **at an equal escalation rate** (the honest equal-cost comparison):

| Encoder | Accepted | by `confidence` | by `P(error)` | Gain |
| --- | --- | --- | --- | --- |
| hash, dim 64 | 50% | 0.685 | 0.714 | +2.9 pp |
| hash, dim 64 | 70% | 0.648 | 0.665 | +1.7 pp |
| hash, dim 64 | 90% | 0.597 | 0.611 | +1.4 pp |
| gpt2, dim 768 | 50% | 0.660 | 0.718 | +5.8 pp |
| gpt2, dim 768 | 70% | 0.627 | 0.657 | +3.0 pp |
| gpt2, dim 768 | 90% | 0.585 | 0.591 | +0.6 pp |

Ranking "being correct" (AURC/eAURC, lower is better):

| Encoder | Ranker | AURC | eAURC | coverage@5% | coverage@10% |
| --- | --- | --- | --- | --- | --- |
| hash | `confidence` | 0.2839 | 0.1843 | 0.087 | 0.107 |
| hash | `P(error)` | **0.2547** | **0.1551** | **0.088** | **0.130** |
| gpt2 | `confidence` | 0.3121 | 0.2051 | 0.076 | 0.090 |
| gpt2 | `P(error)` | **0.2633** | **0.1564** | **0.096** | **0.133** |

## Which features carry signal

`eval/risk/features.py`: the same heads and split, only the feature set changes (3 seeds, mean±std). `base(3)` is the old model, `production` is the one wired into the service (the base three + three new ones).

Differences `variant − confidence` by AURC (hash — 10 seeds/30 pairs, gpt2 — 5 seeds/15 pairs; "✓" means significantly better, "✗" significantly worse, "ns" not resolved):

| Variant | in-dist hash | LOO hash | in-dist gpt2 | LOO gpt2 |
| --- | --- | --- | --- | --- |
| `base(3)` | −0.029 ✓ | +0.000 ns | −0.004 ns | **+0.019 ✗** |
| `+n_classes` | −0.042 ✓ | +0.000 ns | −0.082 ✓ | +0.005 ns |
| `+log_odds_gap` | −0.028 ✓ | **+0.006 ✗** | −0.010 ns | +0.000 ns |
| `+noul_logit` | −0.041 ✓ | **+0.007 ✗** | −0.075 ✓ | **+0.032 ✗** |
| `production` | −0.043 ✓ | **+0.008 ✗** | −0.091 ✓ | +0.002 ns |

`n_classes` delivers most of the in-dist gain on both encoders and **costs nothing in transfer** (≈ baseline there and there). `noul_logit` is the opposite: it helps noticeably in the pool and **significantly breaks transfer on both encoders** (+0.007 and +0.032). `production` itself on hash is significantly worse than `+n_classes` in transfer (+0.008), but on gpt2 it is indistinguishable — so "a poorer feature set transfers better" is **hash-specific and was not confirmed on the working encoder**.

## Calibration of `P(error)`

Ranking is not enough: the cascade compares the risk estimate with a **budget**, so `P(error)` must be an honest probability, not just an ordered score. Therefore `run_loo` and `shots_report` compute **ECE** and **Brier score** on the error probability `1 − score` against the fact `1 − correct`. Both are "lower is better".

The key measurement (10 seeds hash / 5 seeds gpt2, bootstrap CI in "Stability"): **in the pool of questions the difference is resolved; on one new question it is not.**

In the pool (in-dist A/B, ~3000 test rows) the Brier score of `production` beats `confidence` on **all three** datasets and significantly: −0.019 (RouterBench/hash), −0.028 (RouterBench/gpt2), −0.018 (AG News). ECE behaves differently: on RouterBench/hash it is also better (−0.031, significant), while on both gpt2 datasets it is **worse** (+0.015 and +0.006, significant). The reason is visible in absolute numbers: raw gpt2 confidence is already nearly honest (ECE 0.013 and 0.016), there is almost nothing to improve, and a small absolute regression flips the sign. Brier accounts for the whole scale rather than bins and sees no regression. Hence the rule: **ECE and Brier can disagree in sign**, and a calibration metric cannot be the only criterion.

On **one** new question (LOO) ECE is not resolved: the 95% CI of the difference covers zero on all three datasets (−0.008 [−0.031, +0.015] for hash, −0.025 [−0.076, +0.023] for gpt2/RouterBench, +0.008 [−0.010, +0.030] for AG News). Brier on gpt2 leans towards `production` (−0.018 and −0.010, significant), on hash it does not.

The few-shot curve shows that calibration is what own examples fix most reliably. ECE (macro over questions, example order averaged):

| Own rows | RB/hash prod | RB/gpt2 prod | AG/gpt2 prod | RB/hash conf | RB/gpt2 conf | AG/gpt2 conf |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (LOO) | 0.192 | 0.171 | 0.053 | 0.200 | 0.196 | 0.045 |
| 5 | 0.127 | 0.127 | 0.062 | 0.198 | 0.194 | 0.042 |
| 20 | 0.103 | 0.097 | 0.052 | 0.193 | 0.189 | 0.041 |
| 100 | 0.061 | 0.073 | 0.043 | 0.173 | 0.172 | 0.039 |
| all (in-dist) | 0.038 | 0.043 | 0.034 | 0.123 | 0.127 | 0.031 |

`confidence` as a single-feature model barely notices added examples (ranking does not change, only the scale shifts slightly), while the risk model adjusts the scale to the question: five labeled rows already cut ECE roughly in half. On RouterBench (hash and gpt2) `production` overtakes `confidence` by ECE at `n≈5–20`; on AG News only at `n≈100`, because raw confidence there is already honest (ECE 0.045 in LOO vs 0.20 on RouterBench).

## Hard gate: accept only the better option

The project principle is **always measure and accept only the better decision**. So a variant is not "chosen" but **passes a gate**: `gate_risk_vs_confidence` (`eval/risk/core.py`) compares the candidate with the `confidence` baseline across metrics and returns a rejection if at least one regressed beyond **its own** tolerance. Each metric has its own tolerance (`GateMetric`): strict `0.0005` for decisions and `0.02`/`0.01` for calibration — based on **measured** spread, otherwise binarised ECE would reject variants on pure noise:

| Metric | Direction | Tolerance | Where it applies |
| --- | --- | --- | --- |
| AURC | lower is better | 0.0005 | everywhere |
| eAURC | lower is better | 0.0005 | everywhere |
| coverage@5% risk | higher is better | 0.0005 | everywhere |
| acc@0.7 (accuracy at coverage 0.7) | higher is better | 0.0005 | everywhere |
| ECE | lower is better | 0.02 | pool only |
| Brier | lower is better | 0.01 | pool only |

The `0.0005` tolerances are a project decision: a regression is not forgiven, the tolerance only absorbs numerical noise (a false rejection is safe — it falls back to `confidence`).

The gate is computed in four places of the `eval/risk/features.py` harness (printed in the report and written to `features.json` → `gate`):

- **in-dist** — variant against `confidence` on the same data (per seed, pool of questions);
- **LOO** — on a new question, trained without it (`--loo`);
- **few-shot curve** (`--shots`) — a point is compared with the baseline **at the same number of examples**; points are not mixed with each other.

In addition, `eval/risk/run.py` runs the gate over ranking metrics and **writes the verdict into `risk.json`** (`gate`): production reads it when loading the model.

The order of own rows in the curve (`--shot-seeds`) is a nuisance variable: points are averaged over orders **before** the gate (`average_shot_seeds`), so the verdict does not depend on a lucky or unlucky sample of examples.

### What the measurements showed

With the new runs (10 seeds hash, 5 seeds gpt2, `--shots 0 5 10 20 50 100`, `--shot-seeds 0 1 2`), the share of runs where `production` passed the gate against `confidence`:

| Dataset | Encoder | Seeds | in-dist | LOO (0-shot) |
| --- | --- | --- | --- | --- |
| RouterBench | hash | 10 | **8/10** | **13/30** |
| RouterBench | gpt2 | 5 | **3/5** | **3/15** |
| AG News | gpt2 | 5 | **5/5** | **12/15** |

Few-shot curve (share of passed runs against `confidence` at the same `n`, example order averaged):

| Dataset | `n=0` | `n=5` | `n=10` | `n=20` | `n=50` | `n=100` | all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RouterBench, hash (10 seeds) | 13/30 | 16/30 | 14/30 | 14/30 | 15/30 | 13/30 | 15/30 |
| RouterBench, gpt2 (5 seeds) | 3/15 | 8/15 | 7/15 | 5/15 | 8/15 | 7/15 | 9/15 |
| AG News, gpt2 (5 seeds) | 12/15 | 10/15 | 12/15 | 12/15 | 12/15 | 12/15 | 12/15 |

With more seeds, rejections stopped being "rare": on RouterBench/hash in-dist rejects 2/10 (by `coverage@5%`, a gap of 0.001–0.0014 — that is tolerance noise), on RouterBench/gpt2 — 2/5 (by ECE, +0.025 and +0.039 against the baseline). The gate catches a regression in the pool too, not only on a new question.

## Rules for moving to stable

The result of steps 1–6: the risk mode is enabled **deliberately** and only where the measurement showed "not worse". These are the enabling rules (also the beta→stable conditions).

**1. Enabling (deployment).** The risk cascade is enabled by `DECISION_CASCADE_RISK_BUDGET` — the share of errors we are willing to accept locally — and requires a trained `risk.json` certificate (`DECISION_RISK_MODEL_PATH`, default `<DECISION_CHECKPOINT_DIR>/risk.json`). Without the variable, the basic confidence-threshold cascade runs: the default behaviour does not change.

**2. Certificate (hard gate).** A variant is applied only if it is **not worse** than `confidence` on all working metrics (AURC, eAURC, coverage@5%/10%, ECE, Brier). The verdict is recorded in the certificate (`gate.accepted`/`gate.worse`) and recomputed from `gate.excess` — per-metric excesses over tolerance:

- `accepted: false` — the risk rule falls back to `confidence` (`usage.reason = "risk_gate"`, a warning in the log, `/info → risk_gate` = `rejected`);
- no certificate — no measurements were made, the mode works as before;
- `accepted: true` — the risk rule is applied (`usage.reason = "risk"`).

The gate tolerance can be overridden in production with `DECISION_RISK_GATE_DELTA` (`> 0` relaxes, `< 0` tightens) without retraining the model: `gate.excess` is written into the certificate exactly for this (`src.risk.gate_verdict`).

**3. A known set of questions (zero-shot).** On *own* questions the risk model wins through inter-question ordering (`n_classes` is a question constant, and it transfers from validation to test). The gain is confirmed by the gate; on rejection, `confidence` works.

**4. A new question (few-shot).** On a new question a coefficient taken from other questions applies poorly, so the risk model must be retrained. Rule: at `n >= DECISION_RISK_FEW_SHOT_N` (default `RISK_FEW_SHOT_MIN_N = 10`; a value `< 2`, in particular `0`, disables retraining) retrain the logistic regression on {other rows + own `n`} and apply it **only to this question**. Measurement: a dozen own rows usually removes most of the transfer loss; on RouterBench/gpt2 the difference becomes significant at `n ≈ 50`. `add_examples` in `RiskAwareCascadeBackend` is numpy, CPU-only, without heavy dependencies (see `src.risk.fit_logistic`).

**Transport of examples — `POST /v1/feedback`.** The client repeats the decision the service returned and says whether it was correct:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/feedback -H 'content-type: application/json' -d '{
  "examples": [
    {"question": "route", "decision": {"type": "choice", "choice": "A",
     "probabilities": {"A": 0.55, "B": 0.3, "C": 0.15}, "confidence": 0.55},
     "correct": false}
  ]
}'
```

An example is `(question, decision, correctness)`: the risk model's features are derived from the decision alone (`src.risk.decision_features`), so repeating the decision is enough and the state does not need to be encoded again. The question is named as in the decision request. Other rows come from accumulated examples of **other** questions — the same meaning as `others` in the offline curve (`eval.risk.features.shots_report`), so the service reproduces the measured scheme, not a simplified one. While there are no other examples, training uses own rows only, and that is not harmless: the mode is measured separately and is worse while own rows are few, so accumulating examples across **several questions** pays off. The endpoint response names the number of other rows (`reference`) — the client can see the mode in which the model was actually retrained. The store lives in the cascade and is accessed under a lock (FastAPI endpoints run in a threadpool). The endpoint answers `409 FEEDBACK_UNSUPPORTED` if the risk cascade is not selected or few-shot is off (`/info → accepts_feedback` shows support). Once the threshold is reached, the cascade retrains the question's model and applies it to that question; the hard gate is not relaxed — see "Rules for moving to stable".

**5. Reversion.** A "rejected" verdict (with any `DECISION_RISK_GATE_DELTA`) means a fallback to `confidence`: the cost of a mistake is a lost gain, not a regression against the baseline.

## Honest conclusion

**As a router on its own set of questions the risk model is noticeably better than raw confidence** — on gpt2 it is +5.8 pp of accepted accuracy at 50% escalation and AURC 0.2633 vs 0.3121, on hash +2.9 pp and 0.2547 vs 0.2839. This reverses the earlier conclusion of this document: previously the features were only derivatives of the same distribution as `confidence` and gave no gain on a strong head.

But the mechanism must be named honestly: it is **inter-question ordering**, that is, tuning the trigger point for each known question, not a more accurate risk estimate for a particular decision. What follows:

- **On a fixed set of questions** (how the service actually works — specs are given in advance) the gain is real and confirmed: `n_classes` is a question constant, and it transfers from validation to test.
- **Within a question and on a new question there is no gain**: there `confidence` is no worse, and sometimes better. For a new question the risk model must be retrained, otherwise it applies a coefficient from other questions.
- An honest comparison must still be made **at equal cost** (equal escalation share), not by a single end-to-end number.
- **The gate turned the conclusion into an active rule**: when there is no "not worse" on AURC, eAURC, coverage@5% and acc@0.7, the variant is rejected and falls back to `confidence` (see "Hard gate") — on a new RouterBench question (0-shot) `production` currently passes the gate in 5/9 (hash) and 0/3 (gpt2) cases, on AG News in 3/3.

What else was measured and **not** wired in:

- `option_similarity` — the average pairwise similarity of option directions ("option difficulty"). It requires a field in `DecisionResult` and an API change, and gives at best +0.3 pp (gpt2) and zero on the other datasets, and is worse on a new question (LOO 0.2011 on `support`). Not worth the schema cost — see `option-scoring.md`.
- A signal from the **frozen** encoder state — measured and not wired in: it transfers to a new question only on AG News, and only with tuned regularisation. The variant with a **fine-tuned** encoder was also measured and gives no transfer — that direction is closed.

Separately: the error budget is **absolute**. With a base error rate of ~42%, a 5% budget means escalating almost everything (93% of decisions). This is the correct answer of selective prediction, not a defect. The value of the risk model is the calibratable interface (`DECISION_CASCADE_RISK_BUDGET`), not an arbitrary confidence threshold.

The next modelling question — "how to transfer to a new question" — remains open, but replacing the surrogate does not close it **structurally**: the simulation confirmed that both `n_classes` and one-hot give a gain **only** by aligning question levels, and the level of a new question cannot be taken from anywhere. What closes it is either **few-shot** on examples of the new question (calibration — 1–2 rows, ranking under drift — dozens; it does not work for `route`/`benefit` — see "Hard gate"), or retraining the risk model on the new question.

> The full measurement log — per-question feature ablations, LOO details, few-shot curves with bootstrap CIs, synthetic applicability bounds and the fine-tuned-encoder experiments — is in the [Russian original](../risk-aware-cascade.md). The contract and the numbers above are the part that matters for using the cascade.
