# Troubleshooting

**English** · [Русский](../troubleshooting.md)

Typical service errors: what a code means, where it comes from and what to do. For operations (systemd, upgrade/rollback) see [`deploy.md`](deploy.md).

## 1. Quick triad

```bash
curl -fsS http://127.0.0.1:8080/health    # is the process alive?
curl -sS  http://127.0.0.1:8080/ready     # is it ready to serve? (503 + reasons)
curl -fsS http://127.0.0.1:8080/info      # backend, modalities, engine
journalctl -u decision-api@api -n 100 --no-pager
```

`/health` always answers while the process is alive. If it is silent, check `systemctl status decision-api@api.service` and the port (`ss -ltnp | grep 8080`). `/ready` is the only correct source for a load balancer: `503` means "do not send traffic".

## 2. API error codes

All errors are JSON `{"error": {"code": ..., "message": ...}}`.

| Code | HTTP | When |
| --- | --- | --- |
| `UNTRAINED_HEADS` | 409 | there is no trained pack for the question set, and `DECISION_UNTRAINED=refuse`/`auto` with an explicit `DECISION_CHECKPOINT_DIR` |
| `MODALITY_NOT_SUPPORTED` | 422 | `images`/`videos` arrived at `/v1/systemone`, but the backend cannot handle them |
| `BACKEND_UNAVAILABLE` | 502 | the external SystemOne did not answer within the `DECISION_BACKEND_TIMEOUT` budget (all retries exhausted) |
| `FEEDBACK_UNSUPPORTED` | 409 | `/v1/feedback` was called without a cascade, without `DECISION_CASCADE_RISK_BUDGET`, or with `DECISION_RISK_FEW_SHOT_N < 2` |
| `INVALID_QUESTION` | 422 | a question from `/v1/systemone` could not be converted into a spec |
| `422` (FastAPI) | 422 | the request body failed pydantic validation (see §4) |

## 3. `UNTRAINED_HEADS`

```json
{"error": {"code": "UNTRAINED_HEADS", "signature": "d4dcbc937478268a",
           "checkpoint_dir": "checkpoints"}}
```

The heads exist, but there is no trained pack for the **questions signature**. The signature is computed from names, types, instructions, options and criteria — changing any of them yields a different signature.

What to check:

```bash
ls checkpoints/packs/*/metadata.json
.venv/bin/python - <<'EOF'
import json, glob
for path in glob.glob("checkpoints/packs/*/metadata.json"):
    m = json.load(open(path))
    print(path, m.get("specs"), m.get("questions_signature"))
EOF
```

- Does `DECISION_CHECKPOINT_DIR` point to the directory with packs?
- Does the request use the same `instructions`/`options`/`criteria` as training?
- Is the pack in `checkpoints/packs/<name>/` rather than in `checkpoints/`?

Options:

- retrain the heads for the current questions (`python -m training.recipe.train_heads ...`);
- put a pack with a matching signature in place;
- for a demo/debugging, allow untrained heads: `DECISION_UNTRAINED=allow` (the response carries `trained: false`, the numbers are random).

The startup log points more precisely:

```
WARNING no checkpoint <sig>.pt for questions signature <sig>; serving untrained heads
```

## 4. `422` from pydantic

Common causes:

- `options`/`levels` with duplicates — options must be unique;
- `criteria` of a `score` does not match `levels` in length;
- `states` in `/decisions/batch` contains an empty string;
- `confidence`/`probabilities` in `/v1/feedback` outside `[0, 1]` or not summing to 1 (examples with an invalid decision are not accepted, so we do not train on them).

The FastAPI response body contains `detail[].loc` — the path to the field.

## 5. `MODALITY_NOT_SUPPORTED`

```json
{"error": {"code": "MODALITY_NOT_SUPPORTED", "supported": ["text"],
           "requested": ["image"]}}
```

A local backend without a vision encoder accepts text only. To enable CLIP:

```bash
DECISION_VISION_MODEL_ID=openai/clip-vit-base-patch32 \
  .venv/bin/uvicorn api.server.main:app --port 8000
```

`/info` → `modalities` shows the actual set. Video requires `DECISION_VISION_MODEL_ID` and frame decoding; local CLIP does not support video.

## 6. `BACKEND_UNAVAILABLE`

```json
{"error": {"code": "BACKEND_UNAVAILABLE", "backend": "systemone", "status_code": null}}
```

The external service (`DECISION_BACKEND=http` or the cascade fallback) did not answer within the call budget. The request body (`state`, `images`) never appears in the response or the log.

```bash
# is the external service alive and is the URL correct
curl -fsS "$DECISION_BACKEND_URL/health"
# budget and retries
echo "$DECISION_BACKEND_TIMEOUT $DECISION_BACKEND_RETRIES"
journalctl -u decision-api@api -n 50 --no-pager | grep -i unavailable
```

- `status_code: 4xx` — the external service rejected the request; there will be no retries, fix the contract/authorization.
- `status_code: 5xx` or `null` — network/timeout/overload; increase `DECISION_BACKEND_TIMEOUT`, check the network.
- With an unreachable fallback the cascade keeps answering locally, and `/ready` shows `degraded: true` — this is not an error, but escalations do not work.

## 7. `FEEDBACK_UNSUPPORTED`

`/v1/feedback` is not a universal label intake, but transport for the cascade. One of these modes is required:

- `DECISION_BACKEND=cascade` + `DECISION_CASCADE_RISK_BUDGET` + `DECISION_RISK_FEW_SHOT_N >= 2` — few-shot retraining of the risk model;
- `DECISION_BACKEND=cascade` + `DECISION_CASCADE_POLICY=benefit` — refining the external model accuracy `q`.

Check `/info` → `accepts_feedback`. A plain confidence threshold does not accept feedback.

## 8. The cascade does not start: no `q`, risk model or URL

These are configuration errors at startup (the process does not come up), not HTTP codes:

| Message | What to do |
| --- | --- |
| `DECISION_BACKEND=http requires DECISION_BACKEND_URL` | set the external SystemOne URL |
| `DECISION_BACKEND=cascade requires DECISION_CASCADE_FALLBACK_URL` | set the fallback URL |
| `DECISION_CASCADE_POLICY=benefit requires DECISION_CASCADE_REMOTE_ACCURACY` | fit `q`: `python -m eval.competitors.fit_remote_accuracy ...`, put the JSON/number into the variable |
| `DECISION_CASCADE_RISK_BUDGET requires a fitted risk model at ...` | train `risk.json` (`python -m eval.risk.run`) or point `DECISION_RISK_MODEL_PATH` at it |
| `DECISION_UNTRAINED must be one of (...)` | `allow`, `refuse`, `fallback` or `auto` |

## 9. `/ready` answers `503`

The body contains `reasons`:

| Reason | Meaning | What to do |
| --- | --- | --- |
| `encoder is not loaded` | the encoder did not load | check the startup log, `DECISION_MODEL_ID`, memory/disk |
| `vision encoder is not loaded` | CLIP did not load | check `DECISION_VISION_MODEL_ID` and the HF cache |
| `no trained checkpoints for untrained_policy=refuse` | the policy requires a pack, there are none | add a pack or set `DECISION_UNTRAINED=allow` |
| `checkpoints are not readable by this service` | packs exist, but none matched | see §10 (dim/choice_head/signature) |

`heads_loaded` and `checkpoints` in the response show how many packs the service sees and how many are actually loaded.

## 10. A pack was not picked up: `dim` or `choice_head`

In the warmup log:

```
WARNING skip checkpoint heads.pt: choice_head=option_scoring, service choice_head=linear
WARNING checkpoint dim=768 does not match features dim=64; ignoring ...
```

- `DECISION_FEATURE_DIM` must match `feature_dim` from the pack's `metadata.json` (gpt2 — 768, hash encoder — 64).
- The head mode is set by `DECISION_CHOICE_HEAD` (`linear` or `option_scoring`) and must match how the pack was trained. Retrain for another mode: `python -m training.recipe.train_heads --choice-head ...`.
- The questions signature (`questions_signature`) must match: compare `questions` in `metadata.json` with the request.

## 11. The first request is slow

By default packs are warmed up at startup (`DECISION_WARMUP=1`). If warmup is off (`DECISION_WARMUP=0`), the first request pays for reading checkpoints and loading the encoder — that is expected, but in this mode `/ready` does not guarantee that the heads are already in memory.

## 12. Port is busy

```
ERROR: [Errno 98] Address already in use
```

```bash
ss -ltnp | grep 8080
systemctl status decision-api@api.service
```

One instance per port; for several — `decision-api@<instance>.service` with different ports in the unit.

## 13. Several domains: the wrong pack

`DECISION_CHECKPOINT_DIR` may point to a directory with packs (`checkpoints/packs/`), where each domain lives separately:

```
checkpoints/packs/
  support-gpt2/    heads.pt + metadata.json
  routerbench-gpt2/ heads.pt + metadata.json
```

A pack is selected by `questions_signature` from `metadata.json`, not by directory name. To check which pack actually serves a request:

```bash
curl -fsS http://127.0.0.1:8080/info | \
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["engine"]["heads"])'
```

- `trained: false` or `source: null` — there is no pack for the questions signature;
- two packs with the same signature: the first in sort order wins, and the log gets `skip checkpoint ... already served by ...` — remove the duplicate;
- a directory without `metadata.json` is not considered a pack (we do not guess from `heads.pt`).

Metrics and request id — [`observability.md`](../observability.md).
