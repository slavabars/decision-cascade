# Feedback without bias: exploratory escalation and IPS

[Русский](../exploration.md) · **English**

The benefit cascade refines the external model accuracy `q` from feedback. The catch is where that feedback comes from: labels arrive for answers that went outside, and what goes outside is the hard cases. If difficulty correlates with external model quality, `q` is estimated on a biased sample — and the loop closes: an underestimated `q` lowers the threshold, fewer requests escalate, and the sample gets even harder.

This is not a hypothesis about careless clients: any feedback collected on actual calls works this way. Below is the measured size of the bias, the cost of fixing it, and how to turn the fix on in the service.

## Mechanics

`q` shifts in a Bayesian way; the prior weighs `W = DECISION_CASCADE_FEEDBACK_SHRINKAGE` (50) pseudo-observations:

```text
q_new = (q_prior * W + Σ w_i * y_i) / (W + Σ w_i)
```

where `y_i` is whether the external answer was correct and `w_i` is the observation weight. Policy escalation is deterministic, so its weight is 1: without exploration the formula is the same as before. An exploratory escalation happens with probability `epsilon`, and its weight `1 / epsilon` gives the observation the "representation" of similar decisions the policy kept local. This is inverse probability weighting (IPS), also known as off-policy evaluation: the estimate converges to external model quality on the whole population, not on the escalated subsample.

The weight is capped: `min(1 / epsilon, 20)`. At `epsilon = 0.01` the weight would be 100, and one observation would move `q` by tens of percent. The cap biases the estimate only in the tail of the weight distribution but keeps variance under control; at `epsilon >= 0.05` the cap never applies.

The escalation probability is written to the `DECISION_FEEDBACK_PATH` journal together with the observation, so IPS sums are restored after a restart instead of starting from scratch.

## Simulation

The simulation isolates the estimator, not model quality (`python -m eval.feedback.run`). Rows are generated from two strata: easy (local confidence 0.65–0.95, external accuracy `q_easy`) and hard (confidence 0.45–0.64, `q_hard`). The policy sees only confidence; external correctness depends on the stratum. Utility is measured in "correct answer = 1" units assuming calibrated local confidence: local gives `p`, escalation gives `q` minus the call cost. `regret` is the loss against an oracle that knows external model quality on every row.

Run parameters: `q_prior = 0.75`, `W = 50`, `C = 0.05`, `D = 1.0`, 5 seeds, 30 waves of 1000 decisions. The `correlated` scenario has difficult-for-local cases also difficult for the external model (`q_easy = 0.95`, `q_hard = 0.60`, 35% hard rows, true `q = 0.8275`). The `anti` scenario has the external model stronger on the hard cases (`q_easy = 0.70`, `q_hard = 0.95`, true `q = 0.7875`).

| Scenario | Estimator | `epsilon` | `q_hat` | Bias | Escalation rate | Call rate | Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correlated | naive | 0 | 0.6140 ± 0.0074 | −0.2135 | 0.298 | 0.298 | 0.0676 |
| correlated | naive | 0.05 | 0.6480 ± 0.0110 | −0.1795 | 0.352 | 0.384 | 0.0651 |
| correlated | ips | 0.05 | 0.8272 ± 0.0041 | −0.0003 | 0.671 | 0.687 | 0.0172 |
| anti | naive | 0 | 0.8285 ± 0.0017 | +0.0410 | 0.681 | 0.681 | 0.0251 |
| anti | naive | 0.05 | 0.8265 ± 0.0016 | +0.0390 | 0.677 | 0.693 | 0.0280 |
| anti | ips | 0.05 | 0.7896 ± 0.0042 | +0.0021 | 0.601 | 0.621 | 0.0184 |

How to read it:

- **Without exploration the estimate drifts by 21 points** in the scenario where hard-for-local cases are hard for the external model, and by 4 points the other way where the external model is stronger on hard cases. The direction depends on the domain, so both signs are checked.
- **A random slice without weights is not enough.** At `epsilon = 0.05` with naive counting the bias stays at −18 points: observations were added, but each still represents only itself, not the population.
- **IPS removes the bias.** With 5% exploratory calls the estimate lands within ±0.005 of the truth. The price is a call rate rising from 0.30 to 0.69: the corrected `q` raises the threshold, and the policy starts escalating what it used to underestimate. In utility terms this is a win: regret drops from 0.068 to 0.017, recovering almost all of the oracle gap.
- **The reverse scenario is fixed too**: +4.1 points of bias → +0.2, regret 0.025 → 0.018.

Measurement limitations: the strata are synthetic, local confidence is assumed calibrated, and the call cost is constant. The simulation answers "what does the estimator do given a known bias structure", not "how good is the external model on your domain". The latter needs a run on your own data.

## Live run

The simulation checks the estimator; the same path was verified on a running service. Two cascades with the same upstream (local heads on the hash encoder, a `choice` question), 400 decisions each:

- `DECISION_CASCADE_EXPLORE=0.05`: 17 escalations out of 400 (4.25%), all exploratory (`explore_calls = 17`; the policy did not fire at `q = 0.05`). After five correct labels through `/v1/feedback`, the `q` estimate rose from 0.05 to `(0.05·50 + 20·5) / (50 + 100) = 0.6833`, the per-question threshold from 0.05 to 0.6833, and the next decision went outside by policy. Observations in the journal carry `propensity = 0.05`.
- `DECISION_CASCADE_EXPLORE=0`: no escalations and no labels at all — exactly the sample from which a biased `q` cannot be estimated.

The live-run numbers are about plumbing, not quality: the heads are untrained, the labels are synthetic, and the upstream is the same local service. They confirm that exploratory calls reach the journal, the probability is stored, the IPS weight is applied, and feedback moves the threshold.

## How to enable

```bash
DECISION_BACKEND=cascade \
DECISION_CASCADE_POLICY=benefit \
DECISION_CASCADE_REMOTE_ACCURACY=data/public_laya_accuracy.json \
DECISION_CASCADE_EXPLORE=0.05 \
DECISION_CASCADE_FALLBACK_URL=http://127.0.0.1:30000 \
  uvicorn api.server.main:app --port 8000
```

- `DECISION_CASCADE_EXPLORE` — share of decisions the policy would keep local that still go outside (`0…1`, default `0` — behaviour unchanged).
- `DECISION_CASCADE_EXPLORE_SEED` — generator seed for reproducibility (random by default).
- `GET /info → engine`: `explore_rate` is the configured share, `explore_calls` is how many exploratory calls this process made.
- `usage.escalated` includes exploratory calls; `usage.reason` stays `"benefit"`.

Start with 1–5%: at `epsilon = 0.05` the weight cap never applies and the estimate already converges; smaller values are cheaper but noisier, and below 0.05 the weight hits the cap.

Reproduce the measurement:

```bash
.venv/bin/python -m eval.feedback.run --json data/feedback_sim/report.json
```
