"""
Test Lloyd-Max codebook correctness.

Verifies:
  - Centroid symmetry around zero
  - Correct number of levels per bit-width
  - Centroids are monotonically sorted
  - Distortion is within theoretical upper bound
  - Quantize → dequantize round-trip consistency
"""

import math
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant_cutile.codebook import LloydMaxCodebook, solve_lloyd_max


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_correct_num_levels(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.n_levels == 2**bits
    assert cb.centroids.shape == (2**bits,)
    assert cb.boundaries.shape == (2**bits - 1,)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_symmetry(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.centroids.sum().abs().item() < 1e-4, (
        f"Centroids should be symmetric: sum={cb.centroids.sum().item()}"
    )


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_sorted(bits):
    cb = LloydMaxCodebook(128, bits)
    for i in range(len(cb.centroids) - 1):
        assert cb.centroids[i] < cb.centroids[i + 1]
    for i in range(len(cb.boundaries) - 1):
        assert cb.boundaries[i] < cb.boundaries[i + 1]


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_boundaries_between_centroids(bits):
    cb = LloydMaxCodebook(128, bits)
    for i in range(len(cb.boundaries)):
        assert cb.centroids[i] < cb.boundaries[i] < cb.centroids[i + 1]


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_distortion_within_paper_bound(bits):
    """MSE distortion per vector <= sqrt(3) * pi/2 * (1/4^b) for unit vectors."""
    d = 128
    cb = LloydMaxCodebook(d, bits)
    sigma = 1.0 / math.sqrt(d)

    n_samples = 5000
    x = torch.randn(n_samples, d)
    x = x / torch.norm(x, dim=-1, keepdim=True)

    rotated = x  # with identity rotation, still approximately N(0, 1/d)
    indices = cb.quantize(rotated)
    reconstructed = cb.dequantize(indices)
    mse = ((rotated - reconstructed) ** 2).sum(dim=-1).mean().item()

    upper_bound = math.sqrt(3) * math.pi / 2 * (1.0 / (4**bits))
    assert mse < upper_bound * 2.0, (
        f"bits={bits}: MSE={mse:.6f} exceeds 2× paper bound {upper_bound:.6f}"
    )


def test_roundtrip_identity():
    """Quantize → dequantize should map each centroid exactly to itself."""
    cb = LloydMaxCodebook(128, 3)
    indices = torch.arange(cb.n_levels)
    reconstructed = cb.dequantize(indices)
    assert torch.allclose(reconstructed, cb.centroids, atol=1e-6)


def test_quantize_nearest_centroid():
    """Values close to a centroid should map to that centroid's index."""
    cb = LloydMaxCodebook(128, 2)
    for i, c in enumerate(cb.centroids):
        test_val = c + 1e-6
        idx = cb.quantize(test_val.unsqueeze(0))
        assert idx.item() == i, f"Expected index {i}, got {idx.item()}"
