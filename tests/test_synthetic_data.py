import numpy as np

from src.vision.synthetic_data import generate_dataset, generate_hull_image, generate_sample, grid_coverage


def test_generate_hull_image_shape():
    sample = generate_hull_image(size=64, num_stains=3, rng=np.random.default_rng(1))
    assert sample.image.shape == (64, 64, 3)
    assert sample.image.dtype == np.uint8
    assert sample.mask.shape == (64, 64)
    assert sample.mask.min() >= 0.0 and sample.mask.max() <= 1.0


def test_no_stains_gives_empty_mask():
    sample = generate_hull_image(size=32, num_stains=0, rng=np.random.default_rng(2))
    assert sample.mask.sum() == 0


def test_grid_coverage_shape_and_range():
    sample = generate_sample(64, rows=4, cols=6, rng=np.random.default_rng(3))
    assert sample.grid_labels.shape == (4, 6)
    assert sample.grid_labels.min() >= 0.0 and sample.grid_labels.max() <= 1.0


def test_grid_coverage_matches_manual_mean():
    mask = np.ones((10, 10), dtype=np.float32)
    mask[:5, :] = 0.0  # top half clean, bottom half fully stained
    coverage = grid_coverage(mask, rows=2, cols=1)
    assert coverage[0, 0] == 0.0
    assert coverage[1, 0] == 1.0


def test_generate_dataset_shapes():
    images, labels = generate_dataset(5, size=32, rows=2, cols=3, seed=0)
    assert images.shape == (5, 32, 32, 3)
    assert labels.shape == (5, 2, 3)
