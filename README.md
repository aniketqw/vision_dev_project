# VisionDev v2: debug a vision model by tagging and explaining its failures

```
CIFAR-10 CNN  ──►  failures tagged by a YOLO distortion tagger  ──►  3 typical + 3 outlier per tag
  (model.py)        (callback.py + tagger/)                           (selection/: ResNet18 → PCA → t-SNE)
                                                                              │
                          diagnostic report  ◄──  VLM reads the images + stats (diagnosis/)
```

| step | what it does | code | script |
|---|---|---|---|
| 1 | Train the CNN. A Lightning callback logs metrics, and for every misclassified validation image stores the image and its distortion tags | `visiondev/model.py`, `visiondev/callback.py`, `visiondev/tagger/infer.py` | `scripts/01_train_cnn.py` |
| 2 | Group failures by tag, embed with ResNet18 → PCA(50) → t-SNE, pick the 3 images nearest each tag's centroid (typical) and the 3 furthest (outlier) | `visiondev/selection/archetypes.py` | `scripts/02_select_archetypes.py` |
| 3 | Send the picked images plus failure statistics to a vision-language model (observe → hypothesise → verify with stat tools → conclude), render a markdown report | `visiondev/diagnosis/*` | `scripts/03_diagnose.py` |
| tagger | The distortion tagger: recipe, dataset, training, evaluation | `visiondev/tagger/*`, `notebooks/tagger_v2.ipynb`, `weights/tagger_v2.pt` | `python -m visiondev.tagger.<make_dataset\|train\|evaluate>` |

## Run it

```bash
pip install -r requirements.txt
python scripts/01_train_cnn.py --epochs 3          # outputs/logs/training_log_*.json + misclassified_*.json
python scripts/02_select_archetypes.py             # outputs/reports/distortion_report.json + t-SNE plot
python scripts/03_diagnose.py --no-vlm             # stats-only report; drop --no-vlm once a VLM is served
```

Step 3 talks to any OpenAI-compatible endpoint on `--port` (default 8000), e.g. vLLM. For Ollama
(`qwen2.5vl:7b`), run `scripts/ollama_proxy.py`, which translates the OpenAI call to Ollama's native API and sets
the context size Ollama otherwise ignores on `/v1`.

## The tagger: read this before trusting the tags

The tagger itself is accurate (91-95% recall per distortion on held-out injected images, 97.6% of clean images left
untagged) and generalises to a second injection recipe. Full results, how to run it and the notebook are in
[`results/tagger/README.md`](results/tagger/README.md).

But on **natural** CIFAR-10 failures it says "none" for about 96% of images, and tagged failures are no more
common than tagged correct images. So on clean data the four distortions mostly do not explain failures; those are
mainly semantic confusion (cat/dog, truck/car), label noise and model capacity. The pipeline therefore keeps `none` as
a first-class outcome (it is a cluster in step 2). Step 3 analyses all five buckets (blur, jpeg, pixelate, noise, none). The `none` bucket gets its own prompts, which ask about ambiguity, pose, crop, background and likely label problems instead of image degradation, and its own recommendation (audit labels and confusions, not augmentation). Each bucket sends 3 typical + 3 outlier images (`--samples 6`).

## What is in and what was dropped

Kept: the CNN, the callback (cleaned), the tagger (new), selection (ResNet18/PCA/t-SNE), the v3 diagnosis modules
(RAG store, agent graph, schemas, tools, renderer), the Ollama adapter, and `rag_kb/` (15 fault→fix cards, a knowledge base
for the planned fix-recommendation step; not wired into step 3 yet).

Removed from this branch (still on `main`): the whole old `pipe/`, `exp/`, `downloadModel/`, logs and loose scripts. In particular dropped from `pipe/`: the v1 and v2 report generators (superseded by v3), the separate
`export_misclassified.py` (the callback writes that file itself), the old DIV2K-trained `best.pt` tagger, duplicated
initialisation blocks and dead code in the old logger, per-script `sys.path` hacks, and the old PCA-only cluster plot
(replaced by step 2).

## Layout

```
visiondev/        model.py  callback.py  config.py  log.py
  tagger/         recipe.py  make_dataset.py  train.py  evaluate.py  infer.py
  selection/      archetypes.py
  diagnosis/      data_loader  image_sampler  embeddings  rag  schemas  tools  chains  graph  renderer  server_utils
scripts/          01_train_cnn.py  02_select_archetypes.py  03_diagnose.py  ollama_proxy.py
notebooks/        tagger_v2.ipynb
weights/          tagger_v2.pt
results/tagger/   README.md  results.csv  eval_new.json  eval_old.json  train.log  dataset_stats.json  recipe_preview.png
rag_kb/           fix_cards.jsonl  build_cards.py  README.md
```
