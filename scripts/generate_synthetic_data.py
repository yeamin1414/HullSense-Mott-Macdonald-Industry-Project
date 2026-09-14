"""Generates a handful of sample hull images (with ground-truth masks) into
data/sample_images/ — used by the README preview, the dashboard, and as a
quick sanity check that the synthetic generator works end to end.

Usage:
    python scripts/generate_synthetic_data.py --n 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.vision.synthetic_data import generate_sample

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_images"


def main(n: int, rows: int, cols: int, size: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    for i in range(n):
        sample = generate_sample(size, rows, cols, rng)
        plt.imsave(OUT_DIR / f"hull_{i:02d}.png", sample.image)

        ax_img, ax_grid = axes[0, i], axes[1, i]
        ax_img.imshow(sample.image)
        ax_img.set_title(f"sample {i}", fontsize=9)
        ax_img.axis("off")

        ax_grid.imshow(sample.grid_labels, cmap="Reds", vmin=0, vmax=1)
        ax_grid.set_title("dirt grid", fontsize=9)
        ax_grid.axis("off")

    fig.suptitle("Synthetic hull samples with ground-truth dirt grids")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview_grid.png", dpi=130)
    print(f"Wrote {n} hull images + preview_grid.png -> {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.n, args.rows, args.cols, args.size, args.seed)
