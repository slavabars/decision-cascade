# Weight packs on Hugging Face: download and run

[Русский](../huggingface.md) · **English**

Trained decision heads live in the [`slavabars/decision-cascade-packs`](https://huggingface.co/slavabars/decision-cascade-packs) repository. These are ready-made weights for the `decision-cascade` service: download a pack, point the service at your encoder, and it answers with trained heads instead of random ones.

## What a pack is

A pack is a directory with three files: `heads.pt` (head weights and question schemas), `metadata.json` (encoder, dimension, question signature, metrics) and `temperature.json` (calibration report). A pack is tens of kilobytes.

Important: **this is not a `transformers` model**. There is no `AutoModel.from_pretrained` or `pipeline("decision")` here; the encoder (gpt2, distilgpt2, gpt2-medium) is a separate model on the Hub, selected by `DECISION_MODEL_ID`. A pack contains only the heads, so the download is light and encoder weights come through the usual `transformers` path.

## Which pack to choose

| Pack | Encoder | `DECISION_FEATURE_DIM` | Dataset | Test accuracy | Questions |
| --- | --- | ---: | --- | ---: | --- |
| `support-gpt2` | `openai-community/gpt2` | 768 | synthetic (support) | **0.926** | 6: urgency, team, cause, heat, escalation, retention |
| `support-gpt2-medium` | `openai-community/gpt2-medium` | 1024 | synthetic (support) | 0.914 | the same 6 |
| `support-distilgpt2` | `distilgpt2` | 768 | synthetic (support) | 0.868 | the same 6 |
| `public-gpt2` | `openai-community/gpt2` | 768 | AG News | 0.880 | 3: topic, business, tech_level |
| `public-distilgpt2` | `distilgpt2` | 768 | AG News | 0.879 | the same 3 |
| `public-gpt2-medium` | `openai-community/gpt2-medium` | 1024 | AG News | 0.873 | the same 3 |
| `routerbench-gpt2` | `openai-community/gpt2` | 768 | RouterBench 0-shot | 0.548 | 3: escalate, route, benefit |

The numbers are the pack's test split, not a promise for your domain: recalibrate on your own data. Questions in a request must match the pack's question set (meaning and criteria), otherwise the service finds no matching signature and stays without trained heads. `support-*` packs match the default service questions, `public-*` match `PUBLIC_QUESTIONS` (AG News), `routerbench-*` match `ROUTERBENCH_QUESTIONS`.

## Download and run

One pack:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    "slavabars/decision-cascade-packs",
    revision="v1.5.0",
    allow_patterns=["packs/public-gpt2/*"],
)
print(path)
```

```bash
DECISION_MODEL_ID=openai-community/gpt2 \
DECISION_FEATURE_DIM=768 \
DECISION_CHECKPOINT_DIR=<path>/packs/public-gpt2 \
  uvicorn api.server.main:app --port 8000
```

Without a manual download the service can fetch a pack from the Hub itself (beta):

```bash
DECISION_PACK_REPO=slavabars/decision-cascade-packs@v1.5.0 \
DECISION_MODEL_ID=openai-community/gpt2 \
  uvicorn api.server.main:app --port 8000
```

- the cache is `DECISION_PACK_CACHE` (default `~/.cache/decision-cascade/packs`); a second start needs no network;
- the manifest and the sha256 of every file are verified; a broken cache is re-downloaded;
- a download failure is a startup error, not a silent fallback to random heads;
- `DECISION_PACK_REPO` and `DECISION_CHECKPOINT_DIR` are mutually exclusive.

You may also point at the whole `packs/` directory; the service picks a pack by questions, dimension and encoder. For a specific model, however, pointing at a specific pack is safer: `public-gpt2` and `public-distilgpt2` share the same questions and dimension and differ only in the `encoder` metadata field.

For `gpt2-medium` do not forget `DECISION_FEATURE_DIM=1024` (gpt2 and distilgpt2 use 768).

## Check that the pack loaded

- `GET /info → engine.heads[].source` — the `heads.pt` path actually used;
- `GET /ready` — service readiness; with `DECISION_UNTRAINED=refuse` the service will not substitute random heads for a missing pack;
- `trained: true` in the `POST /decisions` response.

If `trained: false`, the usual reasons are mismatched questions (signature) or encoder dimension.

## Limitations

- `risk.json` and `conformal.json` are not part of a pack: they are separate artifacts fitted on your splits (`eval.risk.run`, `eval.conformal.run`). Without them, the confidence threshold and the benefit policy still work.
- Card metrics are a specific dataset and test split. They promise nothing on your domain.
- `heads.pt` is a pickle; the loader reads it with `weights_only=True`.
- A pack does not replace the encoder: without `DECISION_MODEL_ID` the service uses the hash encoder and will not match a pack by dimension.

## Licenses

The code and the head weights are Apache-2.0. Dataset terms are separate: AG News is marked `license: unknown` on the Hub, RouterBench declares no license, and `support` is a synthetic dataset of this project. Dataset files are not included; the card lists sources and metrics only.

## More

- Card with all packs and metrics: https://huggingface.co/slavabars/decision-cascade-packs
- Measurements and competitor comparison: [`benchmarks.md`](benchmarks.md)
