"""Procedural hull-image generator used for demos, dashboarding and CNN training.

Real hull photos from the wash bay camera (a LUCID Triton IP67 unit per the
concept design) are not available in this repo, so this module produces
photorealistic-*enough* synthetic images: a hull-coloured base plate with
randomly placed biofouling/algae "stains" (irregular dark-green/brown blobs)
plus sensor-plausible noise (JPEG-ish speckle, vignette, wet-surface glare).

Every image ships with a ground-truth binary mask, so it doubles as a labelled
dataset for the CNN in `cnn_model.py` — no manual annotation required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HULL_BASE_COLORS = [
    (210, 214, 218),  # white gelcoat
    (60, 90, 120),    # navy hull
    (150, 40, 40),    # red hull
]

STAIN_COLORS = [
    (60, 75, 40),    # algae green
    (70, 55, 35),    # biofouling brown
    (35, 45, 30),    # dark mould-like patch
]


@dataclass
class HullSample:
    image: np.ndarray  # (H, W, 3) uint8
    mask: np.ndarray  # (H, W) float32 in [0, 1], 1 = stained
    grid_labels: np.ndarray  # (rows, cols) float32 in [0, 1], stain coverage per cell


def _draw_blob(mask: np.ndarray, rng: np.random.Generator, cx: float, cy: float, radius: float) -> None:
    """Rasterise an irregular blob (a noisy-radius polygon) onto `mask` in-place."""
    h, w = mask.shape
    n_pts = 14
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = radius * (0.6 + 0.6 * rng.random(n_pts))
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)

    yy, xx = np.mgrid[0:h, 0:w]
    # Point-in-polygon via signed angular winding is overkill here; instead
    # approximate the irregular blob as a distance field modulated by the
    # per-angle radius, which is fast and looks convincingly organic.
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    r_at_theta = np.interp(theta, np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi]),
                            np.concatenate([radii, radii, radii]))
    blob = (dist <= r_at_theta).astype(np.float32)
    mask[:] = np.maximum(mask, blob)


def generate_hull_image(
    size: int = 128,
    num_stains: int | None = None,
    rng: np.random.Generator | None = None,
) -> HullSample:
    rng = rng or np.random.default_rng()
    base_color = np.array(HULL_BASE_COLORS[rng.integers(0, len(HULL_BASE_COLORS))], dtype=np.float32)

    # Base plate with a soft vertical shading gradient (hull curvature) + noise.
    gradient = np.linspace(-18, 18, size).reshape(-1, 1)
    image = np.tile(base_color, (size, size, 1))
    image[:, :, :] += gradient[:, :, None]
    image += rng.normal(0, 4.5, size=(size, size, 3))

    mask = np.zeros((size, size), dtype=np.float32)
    n_stains = num_stains if num_stains is not None else rng.integers(0, 6)
    for _ in range(n_stains):
        cx = rng.uniform(0.1, 0.9) * size
        cy = rng.uniform(0.1, 0.9) * size
        radius = rng.uniform(0.04, 0.16) * size
        _draw_blob(mask, rng, cx, cy, radius)

    stain_color = np.array(STAIN_COLORS[rng.integers(0, len(STAIN_COLORS))], dtype=np.float32)
    stain_noise = rng.normal(0, 6.0, size=(size, size, 3))
    image = image * (1 - mask[:, :, None]) + (stain_color + stain_noise) * mask[:, :, None]

    # Wet-hull specular glare: a few bright streaks (also a real failure mode
    # called out in the concept design — glare confusing the vision model).
    if rng.random() < 0.35:
        streak_row = rng.integers(0, size)
        band = slice(max(0, streak_row - 2), min(size, streak_row + 2))
        image[band, :, :] += 40

    image = np.clip(image, 0, 255).astype(np.uint8)
    return HullSample(image=image, mask=mask, grid_labels=np.zeros((1, 1), dtype=np.float32))


def grid_coverage(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Downsample a pixel mask into per-cell stain coverage fractions."""
    h, w = mask.shape
    cell_h, cell_w = h // rows, w // cols
    coverage = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = mask[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            coverage[r, c] = cell.mean() if cell.size else 0.0
    return coverage


def generate_sample(size: int, rows: int, cols: int, rng: np.random.Generator | None = None) -> HullSample:
    rng = rng or np.random.default_rng()
    sample = generate_hull_image(size=size, rng=rng)
    sample.grid_labels = grid_coverage(sample.mask, rows, cols)
    return sample


def generate_dataset(
    n_samples: int, size: int = 128, rows: int = 4, cols: int = 6, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images [N,H,W,3] uint8, grid_labels [N,rows,cols] float32)."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, size, size, 3), dtype=np.uint8)
    labels = np.zeros((n_samples, rows, cols), dtype=np.float32)
    for i in range(n_samples):
        sample = generate_sample(size, rows, cols, rng)
        images[i] = sample.image
        labels[i] = sample.grid_labels
    return images, labels
