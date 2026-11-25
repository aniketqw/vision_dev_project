# rag_kb: fault -> fix knowledge base for the SimpleCNN

`fix_cards.jsonl` (15 cards, built by `build_cards.py`) maps a diagnosed fault (blur, noise, jpeg,
pixelate, mixed) to a concrete change to `pipe/test.py`. Each card: distortions, symptom, cause, fix,
code snippet, expected effect, evidence level, source, and `verified_by_run` (empty until measured).

Evidence levels: **literature** (4; source URL opened and checked: AugMix, BlurPool, Albumentations
compression docs), **standard** (7; common technique, implementation written here, effect unmeasured),
**hypothesis** (4; our own reasoning, must be confirmed by a retrain-and-score run).

Rebuild and check: `../pipe/.venv/bin/python build_cards.py --smoke`
(validates the schema, syntax-checks every snippet, and runs the torch/PIL snippets; the
albumentations cards are syntax-checked only because that package is not installed locally).

Not included yet: CIFAR-10-C is an external benchmark, not card content. Sources not yet verified
(kuangliu/pytorch-cifar, timm, RobustBench, JPEG-aware training) are deliberately not cited.

Planned feedback loop: after each retrain, write measured per-distortion accuracy deltas into the
card's `verified_by_run`, so retrieval can prefer fixes that worked for this model.
