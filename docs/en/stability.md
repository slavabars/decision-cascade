# Stability: what 1.x promises and what stays beta

**English** · [Русский](../stability.md)

The project has been stable since `1.0.0`. Every exit criterion is closed and has been checked automatically ever since, not just claimed: the API contract, the pack format and the meaning of `DECISION_*` variables do not break inside the `1.x` line — changes land only in a major, with a two-minor deprecation window.

Stability is about the **contract and reproducibility**, not about universality: some surfaces (the benefit cascade, the option-scoring head, encoder fine-tuning and feedback retraining) remain beta — they are measured, but they depend on the domain and may change in minor releases. The risk-aware cascade is stable since `1.1`: its contract is described in §7. The full list is in §2.

## 1. What "stable" means for this project

Three promises that `1.0` makes:

1. **The API contract does not break.** `POST /decisions`, `/decisions/batch`, `/v1/systemone`, `/v1/feedback`, `GET /info` and `schema_version = "1"` change only with a major bump and a deprecation window (2 minor versions).
2. **Artifacts stay readable.** A pack (`heads.pt` + `metadata.json` + `temperature.json`) and a checkpoint written under `schema_version` are read by every version of the `1.x` line.
3. **The published numbers reproduce.** Any figure from [`benchmarks.md`](../benchmarks.md) is produced by a command from the documentation on Ubuntu 22.04+ with a 16 GB GPU, not "on the author's laptop".

Stability does **not** mean: universality (zero-shot), beating LLMs in open domains, or a cloud SLA. This is a local decision engine, and the promise covers only the contract, reproducibility and behaviour on your own domain.

## 2. What is already stable and what is experimental

| Surface | Status | Comment |
| --- | --- | --- |
| `/decisions`, `/decisions/batch` | stable | schema `schema_version = "1"`, contract tests |
| `/v1/systemone` | stable | external Jev/TypeSafe protocol, must not change |
| `/v1/feedback` | stable (beta policy) | transport is fixed; the meaning depends on the cascade |
| `GET /info` | stable | the key set only grows, never shrinks |
| `GET /health`, `GET /ready` | stable | liveness/readiness for systemd and load balancers; `503` while not ready |
| Pack `heads.pt` + `metadata.json` + `temperature.json` | stable | reading older versions is mandatory |
| `DECISION_*` variables from the README | stable | new ones are added, old ones keep their meaning |
| Confidence threshold (`DECISION_CASCADE_THRESHOLD`) | stable | the default policy |
| Risk-aware cascade (`DECISION_CASCADE_RISK_BUDGET`) | stable since 1.1 | §7 contract: variables, `/info`, `usage.reason`, `/v1/feedback`, hard gate |
| Benefit cascade (`DECISION_CASCADE_POLICY=benefit`) | beta | new, needs `q` and feedback |
| Option-scoring head | beta | option transfer is measured, not used in the published packs |
| Encoder fine-tuning (`finetune_encoder`) | beta | measured (+5 pp), weights ship as a release asset |
| Feedback retraining (`training.feedback`) | beta | "do not harm" gate and drift numbers — `../drift.md` |
| `eval/*`, `judge`, benchmark scripts | no guarantees | internal tooling |

## 3. Exit criteria for 1.0 (all met)

Each item is verifiable, not a "feeling of maturity".

| # | Criterion | Now | What was left |
| --- | --- | --- | --- |
| 1 | No breaking changes for 3 minor releases in a row | 0.10 → 0.15 without breaking the contract | — |
| 2 | Deprecation policy published | published in §5, registry `api/server/deprecation.py` | — |
| 3 | Contract tests for every endpoint and the SDK | present for every endpoint | — (reference pack 0.8 in `tests/fixtures/`) |
| 4 | Coverage of `src` + `api` ≥ 90% | **94%** | — (gate `--cov-fail-under=90` in CI) |
| 5 | CI: ruff, shellcheck, pytest, build and smoke of the installed wheel | all of the above | — (`scripts/smoke_wheel.sh`) |
| 6 | Quality regression pinned in CI | nightly job on release packs, 1 pp threshold; p50/p95 latency budget | — (`eval/nightly.py`) |
| 7 | Data reproducibility | RouterBench and AG News pinned by revision, `uv.lock` present | — |
| 8 | Operations: liveness/readiness, timeouts, graceful shutdown | `/health`, `/ready`, budget and retry, SIGTERM test | — |
| 9 | Security: threat model, no PII in logs, dependency audit | `SECURITY.md`, log test, `pip-audit` in CI | — |
| 10 | Documentation: quick start in a clean environment, Ubuntu/CentOS deploy, upgrade/rollback | `deploy.md`, `troubleshooting.md`, automatic rollback | — (a live VM run follows the checklist) |
| 11 | Measurements published with raw data and commands | [`benchmarks.md`](../benchmarks.md) + release assets | — |
| 12 | Two releases in a row without a fire (a hotfix for breaking behaviour) | 0.13 → 0.15 clean | — |

Items 1 and 12 were closed by the `0.10 → 0.15` series: there were no breaking contract changes and no hotfixes for breaking behaviour. By `1.0.0` all twelve items were closed, and each one keeps being checked: contract tests and coverage in CI, the nightly regression on the current release packs, deploy and rollback by tests and the `deploy.md` checklist. If a criterion stops holding, CI or the nightly job will show it, not a user after a release.

## 4. How we got to 1.0

### P0 — done in 0.13.0 (contract and operations)

1. ✅ `/health` (liveness, always answers) and `/ready` (readiness: encoder, readable heads, `degraded` on a cascade with an unreachable fallback). `503` while the service is not ready.
2. ✅ A `DECISION_BACKEND_TIMEOUT` budget for the whole external call, one retry (`DECISION_BACKEND_RETRIES`) for network/429/5xx and `502 BACKEND_UNAVAILABLE` once retries are exhausted. 4xx is not retried.
3. ✅ Coverage gate `--cov-fail-under=90` in `pyproject.toml` + CI and a smoke test of the installed wheel (`scripts/smoke_wheel.sh`: clean venv, `/health`, `/ready`, `POST /decisions`).
4. ✅ AG News pinned by revision (`AG_NEWS_REVISION`), `uv.lock` added; CI checks it is current with `uv lock --check`.
5. ✅ `tests/test_logging_privacy.py`: `state`, `images` and `/v1/feedback` bodies never reach the logs; external backend errors do not contain the request body.

### P1 — done in 0.14.0 (contract stability)

6. ✅ Deprecation policy published (§5): registry `api/server/deprecation.py`, `Deprecation`/`Sunset`/`Link` headers, a warning in the log once per process, and a "not earlier than +2 minor" check at startup and in tests.
7. ✅ Reference pack `tests/fixtures/pack_v0.8.0.pt` (built by the `v0.8.0` tag) is read by the current version and serves decisions — `tests/test_pack_compat.py`.
8. ✅ Nightly job `.github/workflows/nightly.yml`: packs from the latest release, AG News and RouterBench on pinned revisions and seeds, failure on a regression above 1 pp, plus a p50/p95 latency budget (`--latency fail`) and a run on `push` to `main` when decision code paths change (`eval/nightly.py`, `eval/nightly_baseline.json`).
9. ✅ Upgrade/rollback: `deploy/upgrade.sh` (backup → `git pull` → venv → restart → `/ready`, automatic rollback of data and code), `deploy/backup.sh`, `deploy/restore.sh`; Ubuntu/CentOS parity is checked by `distro_from_release` on temporary `os-release` files, and the full cycle was run live on Ubuntu 24.04 and CentOS Stream 9 (`deploy.md` §5).
10. ✅ Troubleshooting: `troubleshooting.md` — every API error code, `/ready` 503, missing `q`, `choice_head`/`dim` mismatch.
11. ✅ Graceful shutdown: `KillSignal=SIGTERM`, `TimeoutStopSec=30`, `--timeout-graceful-shutdown 30`; the contract is pinned by `tests/test_graceful_shutdown.py`.
12. ✅ `pip-audit -r requirements.txt` in CI: no known vulnerabilities.

### P2 — done in 0.15.0

13. ✅ The fine-tuned encoder was published as the `v0.12.0-encoder-routerbench-bert.tar.gz` asset (405 MB) in release `v0.12.0`; `routerbench.md` was updated, the recipe remains for reproduction.
14. ✅ Prometheus metrics (`GET /metrics`) and structured logs with `X-Request-ID` (`DECISION_LOG_FORMAT=json`) — `api/server/metrics.py`, `api/server/observability.py`, `observability.md`.
15. ✅ Weak labeling: `python -m training.labeling.run` labels unlabeled texts with SystemOne teacher answers and writes a dataset that `train_heads` accepts — `weak-labeling.md`.
16. ✅ Multi-domain packs: the service looks for `*/heads.pt` + `metadata.json` and picks a pack by `questions_signature`; duplicate signatures are resolved by the first in sort order with a `WARNING`, and the source is visible in `GET /info` — `tests/test_multidomain.py`.

### 1.0.0 — stable status

Functionally `1.0.0` is identical to `0.16.0`: there are no new features, only status and promises change. The package carries the `5 - Production/Stable` classifier, the README and this document; the API contract, pack format and meaning of `DECISION_*` are frozen by the rules of §5. After that, minor versions add features, patches fix bugs, and breaking changes happen only in a major.

## 5. Deprecation policy (in effect since 1.0)

- A breaking change in the API, pack format or meaning of `DECISION_*` — only in a major.
- Deprecation is announced at least 2 minor versions ahead: a warning in the logs, a note in the README and an entry in the CHANGELOG (`### Deprecated`).
- A minor version may add response fields and environment variables; it may not remove them.
- A patch version changes neither behaviour nor format.

The mechanism is the `DEPRECATIONS` registry in `api/server/deprecation.py`:

```python
Deprecation(
    item="/decisions",          # path, response field or DECISION_* name
    since="1.0.0",              # where it is announced
    remove_in="1.2.0",          # where it disappears: not earlier than +2 minor
    alternative="/v2/decisions",
    sunset="Tue, 01 Sep 2026 00:00:00 GMT",
)
```

What this gives a client and operations:

| Signal | Where it is visible |
| --- | --- |
| `Deprecation: true` | response header on a deprecated path (RFC 9745) |
| `Sunset: <date>` | header when a shutdown date is announced (RFC 8594) |
| `Link: <...>; rel="deprecation"` | header with a link to the replacement |
| `WARNING deprecated ...` | `decision.deprecation` log, once per process lifetime |
| `### Deprecated` | CHANGELOG of the release where it is announced |

The registry is validated at startup (`validate_registry`) and by a test: a `remove_in` earlier than two minor versions ahead is an error, not a warning. A surface may only be removed in a major, after it has lived through the announced period.

## 6. Risks that remain after 1.0

`1.0` is a promise about the contract, not about universality. These limitations remain after the release, and no version can "close" them:

1. **Labels.** All value starts after a few thousand labeled examples per domain. Weak labeling (`weak-labeling.md`) lowers the entry price but does not replace validation on real labels.
2. **Zero-shot.** Competitors work out of the box; we do not. Positioning must admit this, otherwise the first comparison without labels devalues the project.
3. **Domain drift.** Heads age without new examples. The answer is measured (`drift.md`): retraining with replay and a gate gives a limited gain, and a change of the class set requires a separate pack. "Stable" refers to the contract, not to quality on the cut-off date.
4. **One person.** Bus factor: documentation, reproducibility and artifacts are the only insurance; that is why they are in the exit criteria rather than "later".

## 7. Risk-aware cascade contract (stable since 1.1)

Enabled by the pair `DECISION_BACKEND=cascade` + `DECISION_CASCADE_RISK_BUDGET`. Everything below is a promise of the `1.x` line: it can only change in a major.

### Environment variables

| Variable | Meaning | Allowed values | Default |
| --- | --- | --- | --- |
| `DECISION_CASCADE_RISK_BUDGET` | share of errors we are willing to accept locally | `0…1` | — (mode off) |
| `DECISION_RISK_MODEL_PATH` | file of the `P(error)` estimator | path | `<DECISION_CHECKPOINT_DIR>/risk.json` |
| `DECISION_RISK_FEW_SHOT_N` | how many own examples before retraining | `>= 2` enables; `0`/`1` disables | `10` |
| `DECISION_RISK_GATE_DELTA` | tolerance for the hard gate certificate | `> 0` relaxes, `< 0` tightens | `0.0` |

An invalid value (budget outside `[0, 1]`, not a number) is a startup error, not silently different behaviour. A missing `risk.json` with the budget enabled is also a startup error with a hint on how to train the model.

### `GET /info -> engine`

| Key | Type | When present |
| --- | --- | --- |
| `risk_budget` | float | always with the risk cascade |
| `risk_gate` | `"ok"` / `"rejected"` | always |
| `risk_gate_delta` | float | always |
| `risk_few_shot_n` | int | when few-shot is enabled |
| `risk_few_shot` | list of question names | when few-shot is enabled |

Keys may be added, but never renamed or removed.

### `usage` in the decision response

- `escalated` — how many questions went outside (int);
- `reason` — `"risk"` (risk rule) or `"risk_gate"` (escalation by confidence threshold because the certificate was rejected); the field appears only on escalation;
- `fallback` — `usage` of the external call.

### `POST /v1/feedback`

`200` with `accepted` and `questions[]`, where each question has `question`, `accepted`, `examples`, `reference`, `retrained`. If few-shot is off — `409 FEEDBACK_UNSUPPORTED`, and the same is visible in `GET /info -> accepts_feedback`.

### Hard gate

A rejected certificate (`risk.json -> gate.accepted = false`) is **not relaxed**: the risk rule is not applied, escalation follows `DECISION_CASCADE_THRESHOLD`, `reason = "risk_gate"`. The gate can only be relaxed with an explicit `DECISION_RISK_GATE_DELTA > 0`; this is written to the log and visible in `/info`.

### `risk.json`

Fields `weights`, `bias`, `feature_names`, `gate`. The default feature names are `confidence`, `entropy`, `margin`. Files without `gate` are read as "no measurements were made" (the gate does not reject).

The contract is pinned by `tests/test_risk_contract.py`, which also checks that the documentation and the code have not drifted apart.
