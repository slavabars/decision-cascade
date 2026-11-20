# Weight packs on Hugging Face

[Русский](../huggingface.md) · **English**

Trained decision heads are published to the Hugging Face Hub as a separate repository, [`slavabars/decision-cascade-packs`](https://huggingface.co/slavabars/decision-cascade-packs). This repository is **not** a `transformers` model: there is no `AutoModel.from_pretrained` and no `pipeline("decision")` in it. The encoder (gpt2, distilgpt2, gpt2-medium) stays a separate model on the Hub and is selected by `DECISION_MODEL_ID`; a pack contains only heads, metadata and the calibration report.

## What is inside

```text
README.md          # card: metrics, licenses, usage
manifest.json      # version, commit, sizes and sha256 of every file
packs/
  public-gpt2/{heads.pt, metadata.json, temperature.json}
  public-distilgpt2/
  public-gpt2-medium/
  support-gpt2/
  support-distilgpt2/
  support-gpt2-medium/
  routerbench-gpt2/
```

Datasets (`dataset.jsonl`) are not published: they are reproduced from public datasets, while their hash and split sizes live in `metadata.json`. The `chess`, `maze` and `ru-rubert-tiny2` packs are not included: the first two are demo assets, and the third has unclear dataset terms.

## Download and run

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

`DECISION_CHECKPOINT_DIR` may also point at the whole `packs/` directory; the service picks a pack by question signature and feature dimension, and when both match, by the `metadata.json` encoder compared with `DECISION_MODEL_ID` (aliases `gpt2` and `openai-community/gpt2` count as the same model). If the encoder id is unknown (hash), sort order remains, so pointing at a specific pack is safer for a specific model.

A downloaded pack directory works as `DECISION_CHECKPOINT_DIR` as is: the service recognises `metadata.json` + `heads.pt` directly at its root. This is the same format as in the GitHub Release archive, minus the duplicate `<signature>.pt` file.

Check `GET /info → engine.heads[].source` or `GET /ready` to confirm the pack was loaded; with `DECISION_UNTRAINED=refuse` the service will not silently substitute random heads.

## One click: `DECISION_PACK_REPO`

Instead of calling `snapshot_download` by hand, the service can fetch packs from the Hub itself:

```bash
DECISION_PACK_REPO=slavabars/decision-cascade-packs@v1.5.0 \
DECISION_MODEL_ID=openai-community/gpt2 \
  uvicorn api.server.main:app --port 8000
```

- the format is `owner/name[@revision]`; without a revision, `main` is used;
- the cache is `DECISION_PACK_CACHE` (default `~/.cache/decision-cascade/packs`); if the manifest and hashes in the cache match, no network is needed;
- the downloaded manifest is verified against the sha256 of every file; a broken cache is re-downloaded instead of being served;
- a download failure is a startup error, not a silent fallback to random heads;
- `DECISION_PACK_REPO` and `DECISION_CHECKPOINT_DIR` are mutually exclusive: there is no sensible precedence, so startup fails;
- `risk.json` and `conformal.json` are not part of a pack, so with the risk budget enabled set `DECISION_RISK_MODEL_PATH` explicitly.

The mode is beta: the spec and cache behaviour may be refined, but it is off by default and changes nothing until the variable is set.

## Publishing

The publisher is `training/recipe/publish_hf.py`. It builds the manifest and card from the pack directory, checks the schema, hashes, dimensions and dataset licenses, and only then uploads.

```bash
# check packs and build the manifest without uploading
python -m training.recipe.publish_hf --check

# staging directory: manifest.json, README.md, packs/<name>/...
python -m training.recipe.publish_hf --dry-run --out dist/hf

# publish and create a revision tag
HF_TOKEN=... python -m training.recipe.publish_hf \
  --publish --repo slavabars/decision-cascade-packs --revision v1.5.0

# verify the published files against the manifest (read access is enough)
python -m training.recipe.publish_hf --verify \
  --repo slavabars/decision-cascade-packs --revision v1.5.0
```

Rules:

- authenticate with `HF_TOKEN` in the environment or `huggingface-cli login` (the token is stored in `~/.cache/huggingface/token`); a token is a secret, never paste it into chats, issues or logs, and revoke it if it leaks;
- the token is read only from `HF_TOKEN` or the local login and never printed;
- republishing an existing revision does not overwrite it: files are compared by sha256, and a mismatch is an error ("bump the revision");
- `--library-version` pins which release the packs belong to (the installed package version by default);
- `--exclude` drops packs by directory name; `chess`, `maze` and `ru-rubert-tiny2` are excluded by default;
- `--from-stage` publishes an already built staging directory (CI uses this to avoid rebuilding the manifest).

## CI

`.github/workflows/publish-hf.yml` runs manually (`workflow_dispatch`). This is deliberate: packs are not in git, their artifact is a GitHub Release archive, so the workflow first downloads `*-weights.tar.gz` for the requested release, then checks and builds staging.

- `packs_release` — GitHub Release tag with the weight archive (default `v1.5.0`);
- `revision` — tag in the HF repository; empty publishes to `main` only;
- `dry_run` — `true` by default: checks and staging without writing to the Hub.

Writing is enabled only when `vars.HF_PUBLISH_ENABLED == 'true'` and `secrets.HF_TOKEN` is set; without a token the job finishes with a warning. After upload, a smoke step downloads `public-gpt2` via `snapshot_download`, starts `DecisionService` with gpt2 and checks that the pack is trained and answers — catching a broken archive and a question signature mismatch.

## Licenses and provenance

The code and the head weights are Apache-2.0. Dataset terms are separate and listed per pack in the card: AG News is marked `license: unknown` on the Hub, RouterBench does not declare a license, and `support` is a synthetic dataset of this project. Dataset files are not redistributed; the manifest keeps dataset names, split sizes and the hash of the original `dataset.jsonl`.

## Limitations

- A pack does not replace calibration on your domain: card metrics are the test split of a specific dataset, not a guarantee on your data.
- `risk.json` and `conformal.json` are not part of a pack: they are separate artifacts fitted on your splits (`eval.risk.run`, `eval.conformal.run`).
- `heads.pt` is a pickle; the loader reads it with `weights_only=True`, but moving to safetensors is a separate task, not a publishing requirement.
- Updating a pack means a new revision, not overwriting a tag: `revision` stays reproducible.
