# Distortion tagger v2 (YOLO11n)

Tags an image with the distortions it contains: `blur`, `noise`, `pixelate`, `jpeg`. Used in VisionDev to tag the
images a CIFAR-10 CNN misclassifies.

The notebook [`tagger_v2.ipynb`](tagger_v2.ipynb) walks through everything with stored outputs: how faults are
injected, the training run, the accuracy tests and live inference. This file is the short version: how to run it
and what the numbers were.

## Why a v2

The original tagger (`best.pt` in the upstream repo) was trained on 512 px DIV2K crops with one strong local fault
per image, and its noise injection had a bug (uint8 wrap-around turned half of every noisy patch white). At CIFAR
scale (32x32) it detected almost nothing: on injected faults its recall was 0-3%, and it never answers "clean".

v2 changes:

| | original | v2 |
|---|---|---|
| training images | DIV2K 512 px crops | CIFAR-10 train, 32x32 (run at imgsz 64) |
| faults per image | one, local | 0 to 3, global or local patch |
| clean images | none | 25% (empty label, so the model learns to say nothing) |
| noise | wraps around to white | signed arithmetic + clip |
| augmentation | default | geometric aug off (mosaic, scale, translate...) so faults are not resampled away |

## Files

| file | purpose |
|---|---|
| `visiondev/tagger/recipe.py` | the fault functions and `make_sample` (what gets injected, plus the YOLO labels) |
| `visiondev/tagger/make_dataset.py` | builds 24,000 train + 3,000 val images from CIFAR-10 **train** only |
| `visiondev/tagger/train.py` | trains YOLO11n from scratch (no pretrained weights) |
| `visiondev/tagger/evaluate.py` | the four tests below, for any tagger weights |
| `visiondev/tagger/infer.py` | `Tagger` class used by the training callback and the pipeline |
| `notebooks/tagger_v2.ipynb` | notebook with stored outputs |
| `visiondev/weights/tagger_v2.pt` | trained weights (5.3 MB) |
| `results/tagger/` | `results.csv` (per-epoch), `eval_new.json`, `eval_old.json`, `train.log`, `dataset_stats.json` |

## How to run

From the repo root:

```bash
python -m visiondev.tagger.make_dataset --cifar-root ./data --out outputs/tagger_dataset      # about 1 min
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m visiondev.tagger.train --data outputs/tagger_dataset/data.yaml --device mps --epochs 40
python -m visiondev.tagger.evaluate --weights outputs/tagger_runs/tagger_v2/weights/best.pt --out outputs/eval_new.json \
    --predictions preds.npy --misclassified visiondev_runs/<run>/misclassified_<ts>.json
```

The 40-epoch run takes about 40 min on Apple MPS when nothing else uses the GPU (the logged run took 98 min because other jobs shared it).
The committed weights are the result of exactly this run; you do not need to retrain to use them.

Inference on one image:

```python
from visiondev.tagger.infer import Tagger
tagger = Tagger()                  # loads the bundled weights
tagger.tags(img_rgb)               # img_rgb: HxWx3 uint8 -> {"blur": 0.97} ; {} means no fault found
```

`Tagger` handles the BGR conversion and uses `imgsz=64` and `conf=0.25`, the settings behind every result below.

## Training

YOLO11n architecture from scratch, imgsz 64, batch 128, 40 epochs (about 55 s per epoch on Apple MPS),
horizontal flip and colour jitter kept. Best validation (3,000 images): mAP50 0.383, mAP50-95 0.247,
precision 0.41, recall 0.47 at box level. Box metrics are low by nature (a global fault's box is the whole image);
the fault *name* accuracy below is what matters.

## Results

All on CIFAR-10 **test** images, which were never used for training. A tag counts at confidence >= 0.25.

### A. Held-out recipe test (1,000 images, v2 recipe)

| fault | old recall | new precision | new recall | new F1 |
|---|---|---|---|---|
| blur | 1.4% | 96.0% | 91.0% | 0.935 |
| noise | 0% | 98.0% | 94.1% | 0.960 |
| pixelate | 0% | 97.5% | 90.1% | 0.937 |
| jpeg | 2.9% | 98.5% | 95.4% | 0.969 |

Clean images correctly left untagged: old 98.8%, new 97.6%. Exact tag-set match: single-fault images 97.3%,
images with two or more faults 73.2%.

### B. Dose-response with a different recipe (PIL, global, 200 images per cell)

Recall by severity (1 mild, 5 strong):

| fault | 1 | 2 | 3 | 4 | 5 | old tagger at 3 |
|---|---|---|---|---|---|---|
| blur | 19.5% | 100% | 100% | 100% | 100% | 0% |
| noise | 91.5% | 100% | 100% | 100% | 100% | 0% |
| pixelate | 28.5% | 96% | 100% | 100% | 100% | 0% |
| jpeg | 42.5% | 100% | 100% | 100% | 100% | 2% |

Weakest at severity 1, where most misses are "no fault detected".

### C. Failed vs correct on clean CIFAR (1,500 each)

| tag | failed | correct | p |
|---|---|---|---|
| blur | 1.9% | 2.3% | 0.52 |
| noise | 0.4% | 0.4% | 1.00 |
| pixelate | 0.0% | 0.1% | 0.48 |
| jpeg | 1.0% | 2.1% | 0.019 |
| none | 96.7% | 95.1% | 0.034 |

### D. The 4,821 stored failures re-tagged

96.3% get no tag. Of the rest, blur 96, jpeg 62, noise 20, pixelate 1 (unique images).
The old tags (blur 2,382, jpeg 1,476, pixelate 829, noise 134) were artifacts of the old tagger guessing.

## What this does and does not show

* As a detector of injected faults, v2 is accurate and generalises across injection recipes.
* It does **not** explain why the CNN fails on clean CIFAR-10: failures are not more likely to carry a fault tag
  (tests C and D). Those failures are mostly semantic confusion, label noise or model capacity.
* Trained on synthetic faults only. Real camera blur, sensor noise and recompression are not tested.
* Mixed-fault images are the weak spot (73% exact match); low-severity faults are often missed.
