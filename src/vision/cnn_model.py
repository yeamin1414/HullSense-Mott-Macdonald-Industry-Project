"""A small convolutional neural network that learns to localise hull stains.

This is the deep-learning counterpart to `classical_detector.py`. Rather than
predicting a full-resolution segmentation mask, the network's final layer is
pooled down to the same (rows x cols) grid used everywhere else in the
system, so its output plugs directly into the orchestrator. This keeps the
model tiny (trains in well under a minute on CPU) while still being a real
trained artificial neural network doing genuine per-pixel feature learning —
not a lookup table.

`torch` is an optional dependency: every caller in this repo checks
`TORCH_AVAILABLE` first and falls back to `ClassicalStainDetector` if it is
missing, so the rest of the system still runs on a machine without PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch installed
    TORCH_AVAILABLE = False


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "stain_cnn.pt"


if TORCH_AVAILABLE:

    class StainSegNet(nn.Module):
        """4-layer conv encoder -> adaptive-pooled to a (rows, cols) dirt-probability grid."""

        def __init__(self, rows: int = 4, cols: int = 6) -> None:
            super().__init__()
            self.rows, self.cols = rows, cols
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feats = self.features(x)
            logits = self.head(feats)  # (B, 1, H', W')
            pooled = F.adaptive_avg_pool2d(logits, (self.rows, self.cols))  # (B, 1, rows, cols)
            return torch.sigmoid(pooled).squeeze(1)  # (B, rows, cols)


def _to_tensor(image: np.ndarray) -> "torch.Tensor":
    arr = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


class CNNStainDetector:
    """Inference wrapper: loads trained weights (or a fresh/untrained net if
    none exist yet) and predicts a dirtiness grid for a hull image."""

    def __init__(self, rows: int = 4, cols: int = 6, weights_path: str | Path = DEFAULT_WEIGHTS_PATH):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed — CNNStainDetector unavailable. "
                                "Install with `pip install torch` or use ClassicalStainDetector instead.")
        self.rows, self.cols = rows, cols
        self.model = StainSegNet(rows, cols)
        self.weights_path = Path(weights_path)
        self.trained = False
        if self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
            self.trained = True
        self.model.eval()

    def predict_grid(self, image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(_to_tensor(image))
        return out.squeeze(0).numpy()

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path


def train(
    n_samples: int = 600,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 1e-3,
    rows: int = 4,
    cols: int = 6,
    image_size: int = 128,
    seed: int = 42,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    log_fn=print,
) -> "CNNStainDetector":
    """Trains StainSegNet on procedurally generated + labelled hull images.

    Because labels come from the same generator that draws the stains, this
    is a self-supervised-in-spirit training loop with zero manual annotation
    — a practical necessity given no real hull photos are available yet
    (see the "honest limits" discussion in docs/architecture.md).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed — run `pip install torch` to train the CNN.")

    from src.vision.synthetic_data import generate_dataset

    torch.manual_seed(seed)
    images, labels = generate_dataset(n_samples, size=image_size, rows=rows, cols=cols, seed=seed)

    x = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    y = torch.from_numpy(labels)

    n_val = max(1, int(0.15 * n_samples))
    x_train, y_train = x[n_val:], y[n_val:]
    x_val, y_val = x[:n_val], y[:n_val]

    model = StainSegNet(rows, cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n_train = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = (((val_pred > 0.35) == (y_val > 0.35)).float().mean().item())
        log_fn(f"epoch {epoch:2d}/{epochs} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | val_cell_acc={val_acc:.3f}")

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    log_fn(f"saved trained weights -> {weights_path}")

    detector = CNNStainDetector(rows, cols, weights_path)
    return detector
