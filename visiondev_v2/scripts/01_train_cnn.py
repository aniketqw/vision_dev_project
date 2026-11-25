"""Train the CIFAR-10 CNN with the DebugLogger callback and the distortion tagger.

    python scripts/01_train_cnn.py --epochs 3
Writes outputs/logs/training_log_<ts>.json and outputs/logs/misclassified_<ts>.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytorch_lightning as pl

from visiondev.callback import DebugLogger
from visiondev.config import DEFAULT_LOGS_DIR, DEFAULT_TAGGER
from visiondev.model import SimpleCNN, cifar10_loaders
from visiondev.tagger.infer import Tagger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cifar-root", default="./data")
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    ap.add_argument("--tagger", default=str(DEFAULT_TAGGER), help="tagger weights, or 'none' to skip tagging")
    a = ap.parse_args()

    tagger = None if a.tagger == "none" else Tagger(a.tagger)
    train_loader, val_loader = cifar10_loaders(a.cifar_root, a.batch_size)
    trainer = pl.Trainer(max_epochs=a.epochs, callbacks=[DebugLogger(str(a.logs_dir), tagger)], logger=False,
                         default_root_dir=str(a.logs_dir.parent))
    trainer.fit(SimpleCNN(), train_loader, val_loader)



if __name__ == "__main__":  # guard needed: DataLoader workers re-import this file on macOS
    main()
