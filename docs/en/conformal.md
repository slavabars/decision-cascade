# Conformal cascade: a guarantee on the error share

**English** · [Русский](../conformal.md)

The confidence threshold and the error budget express a **wish**: "escalate everything below 0.6" or "we want at most 5% local errors". But the threshold is fitted on a finite sample, and on new data the error share moves — especially when calibration is short. The conformal cascade turns the wish into a **probabilistic promise**: a threshold on `P(error)` is chosen on a held-out calibration split such that the upper bound of the error share among accepted decisions stays below `alpha`, with no assumptions about the model — only exchangeability of calibration and new data.

```text
P(share of errors among locally accepted decisions ≤ alpha) ≥ 1 − delta
```

The guarantee is given **for every question simultaneously**: the `delta` level is split between questions by Bonferroni. This matters because decisions on different questions for the same state are correlated, and they cannot be pooled into one independent sample.

## How it is computed

1. Validation is split in two halves: the `P(error)` estimator is trained on the first (the same logistic regression as in the risk-aware cascade), the certificate is fitted on the second.
2. Decisions are sorted by `P(error)`. For every candidate threshold, the accepted decisions, the number of errors and the **exact Clopper–Pearson upper bound** at level `1 − delta / Q` are computed, where `Q` is the number of questions.
3. The widest threshold is taken such that the bound fits into `alpha` for **all** questions and coverage is at least `min_coverage`. If no such threshold exists, the certificate honestly reports zero coverage: accepting nothing is better than promising more than the data allows.

The Clopper–Pearson bound is not optimistic: with zero errors it equals `1 − delta ** (1 / n)`, so a dozen perfect examples proves nothing. To certify `alpha = 0.05` at `delta = 0.05` with no errors you need about 59 examples, and when `delta` is split across three questions — about 67 per question.

## Commands

The certificate is fitted offline; the server only reads the finished file.

```bash
# public / gpt2, soft guarantee: alpha=0.1, delta=0.1
python -m eval.conformal.run --specs public --encoder openai-community/gpt2 \
    --dim 768 --train-steps 300 --limit 16000 \
    --alpha 0.1 --delta 0.1 --min-coverage 0.5 \
    --out-dir data/conformal_public

# public / gpt2, strict guarantee: alpha=0.05
python -m eval.conformal.run --specs public --encoder openai-community/gpt2 \
    --dim 768 --train-steps 300 --limit 16000 \
    --alpha 0.05 --delta 0.1 --min-coverage 0.0 \
    --out-dir data/conformal_public_strict
```

The CLI writes `conformal.json` (read by production) and `metrics.json` (all numbers and the gate slice). Defaults: `alpha=0.05`, `delta=0.1`, `min_coverage=0.3`. `delta=0.1` is the practical default: after splitting across questions, `0.05` requires too many calibration rows per question.

Enabling it in the service:

```bash
DECISION_BACKEND=cascade \
DECISION_CASCADE_POLICY=conformal \
DECISION_CONFORMAL_PATH=data/conformal_public/conformal.json \
DECISION_RISK_MODEL_PATH=checkpoints/risk.json \
DECISION_CASCADE_FALLBACK_URL=http://127.0.0.1:30000 \
  uvicorn api.server.main:app --port 8000
```

The `P(error)` estimator can be trained separately (`python -m eval.risk.run`) or taken from the `eval.conformal.run` run — there it is already fitted on the first half of validation. The escalation reason in `usage` is `"conformal"`; with a rejected certificate it is `"conformal_gate"`.

## Reference numbers

Public dataset (AG News questions `topic`, `business`, `tech_level`), gpt2, `limit 16000`: train 14400, val 1600 (800 for the risk model + 800 for calibration), test 4000. Calibration holds 2400 decisions (three questions per example).

| Mode | `alpha` | Calibration: coverage | risk | upper bound | Test: coverage | risk | confidence@coverage | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| soft | 0.10 | 0.915 | 0.0711 | 0.0999 | 0.918 | 0.0785 | 0.0796 | accepted |
| strict | 0.05 | 0.547 | 0.0175 | 0.0482 | 0.534 | 0.0226 | 0.0413 | accepted |

In the strict mode, at the same coverage the conformal threshold is almost twice as accurate as ranking by confidence (risk 0.0226 vs 0.0413), and the promised 5% holds on test. In the soft mode coverage is 92% at a 7.9% risk — less of an insurance policy and more a way to avoid paying for the external model on most decisions.

RouterBench (`--specs routerbench --limit 8000`, gpt2, `alpha=0.05`, `delta=0.1`, `min_coverage=0.3`) yields **zero coverage**: on three questions with local accuracy 0.553 the data is not enough to honestly promise a 5% error share on even 30% of decisions. The certificate refuses, the gate is rejected, and the cascade falls back to `DECISION_CASCADE_THRESHOLD` with `reason = "conformal_gate"`. This is not a failure but expected behaviour: on a weak domain an honest guarantee simply does not exist.

## The "do no harm" gate

The certificate carries a verdict measured on the **held-out test**, not on calibration:

- coverage is at least `min_coverage`;
- risk is at most `alpha` with tolerance `tol = 0.01`;
- risk is no worse than ranking by confidence at the same coverage.

If the verdict is negative, the conformal threshold is not applied: escalation follows `DECISION_CASCADE_THRESHOLD`, `usage.reason = "conformal_gate"`, and the reason is visible in `GET /info → engine.conformal_gate_worse`. The gate cannot be relaxed by an environment variable — only by refitting the certificate on data where it passes.

## Limitations

- **Exchangeability.** The guarantee holds while new data resembles the calibration. A change of domain, labels or head model requires a new certificate; the old file keeps working, but the promise does not extend to the new data.
- **The price of Bonferroni.** The more questions, the smaller `delta` per question and the more calibration rows are needed. With ten questions at `delta=0.1` only `0.01` remains per question — without thousands of decisions coverage will most likely be zero.
- **The certificate is fixed.** The conformal cascade does not accept feedback: "retraining" the threshold without recomputing the bound would break the promise. Refitting is an offline command only.
- **Short questions.** The gate rests on overall risk: on a question with a hundred decisions the noise is not comparable to `alpha`, so per-question numbers in `conformal_test.groups` are diagnostics, not a separate guarantee.
- **Zero coverage is a valid answer.** If the data is insufficient, the certificate refuses. That is better than a threshold that looks usable but guarantees nothing.

The contract for variables, `/info` and `usage` is pinned in [`stability.md`](stability.md) §9 and checked by `tests/test_conformal_contract.py`.
