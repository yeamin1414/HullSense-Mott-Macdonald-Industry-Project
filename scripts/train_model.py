"""Trains the StainSegNet CNN (src/vision/cnn_model.py) on procedurally
generated, auto-labelled hull images and saves weights to models/stain_cnn.pt.

Usage:
    python scripts/train_model.py --epochs 15 --samples 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG
from src.vision.cnn_model import TORCH_AVAILABLE, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Run `pip install torch` first, "
              "or use the classical CV detector — no training required.")
        raise SystemExit(1)

    cfg = DEFAULT_CONFIG.grid
    train(
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rows=cfg.rows,
        cols=cfg.cols,
        image_size=cfg.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
