"""Example: train a small CIFAR-10 CNN with the VisionDev callback. The CNN is just a stand-in; use your own model.

    pip install -e ".[diagnosis]"
    python examples/cifar10_simplecnn.py --epochs 3 --diagnose            # needs a VLM endpoint on --port
    python examples/cifar10_simplecnn.py --epochs 3 --diagnose --no-vlm   # statistics-only report

Everything lands in visiondev_runs/<timestamp>/ , the deliverable being report.md.
"""
import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(pl.LightningModule):
    """3 x (conv + maxpool) -> fc 256 -> dropout -> fc 10. About 71% on CIFAR-10 after 3 epochs."""

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(self.dropout(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        self.log("accuracy", (logits.argmax(1) == y).float().mean())
        return loss


def cifar10_loaders(root: str = "./data", batch_size: int = 32, num_workers: int = 2):
    """Train/val loaders. Images are normalised to [-1, 1]; the callback un-normalises them to save PNGs."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    train = datasets.CIFAR10(root, train=True, transform=tf, download=True)
    val = datasets.CIFAR10(root, train=False, transform=tf, download=True)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def main():
    import pytorch_lightning as pl

    from visiondev import DebugLogger
    from visiondev.pipeline import DiagnosisConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cifar-root", default="./data")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--diagnose", action="store_true", help="run selection + VLM + report after training")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    a = ap.parse_args()
    train_loader, val_loader = cifar10_loaders(a.cifar_root, a.batch_size)
    cfg = DiagnosisConfig(use_vlm=not a.no_vlm, vlm_port=a.port, vlm_model=a.model)
    cb = DebugLogger(save_dir=a.run_dir, diagnose=a.diagnose, diagnosis=cfg)
    pl.Trainer(max_epochs=a.epochs, callbacks=[cb], logger=False, default_root_dir=str(cb.run_dir)).fit(
        SimpleCNN(), train_loader, val_loader)
    print("report:", cb.report_path)


if __name__ == "__main__":  # guard needed: DataLoader workers re-import this file on macOS
    main()
