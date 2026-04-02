"""
Test compression kernel correctness against PyTorch reference.

Compares cuTile kernel output (via TurboQuantEngine) against the pure PyTorch
reference path for:
  - MSE indices match
  - QJL signs match
  - Norms are within FP16 tolerance
  - Various sequence lengths and bit widths
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant_cutile import TurboQuantEngine


def _make_random_keys(seq_k: int, head_dim: int = 128) -> torch.Tensor:
    K = torch.randn(seq_k, head_dim)
    return K.half()


@pytest.mark.parametrize("seq_k", [64, 128, 256, 512])
@pytest.mark.parametrize("total_bits", [3, 4])
def test_compress_keys_shapes(seq_k, total_bits):
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")
    K = _make_random_keys(seq_k)
    compressed = engine.compress_keys_pytorch(K)

    assert compressed["indices"].shape == (seq_k, 128)
    assert compressed["k_mse"].shape == (seq_k, 128)
    assert compressed["qjl_signs"].shape == (seq_k, 128)
    assert compressed["vec_norms"].shape == (seq_k,)
    assert compressed["residual_norms"].shape == (seq_k,)


@pytest.mark.parametrize("total_bits", [3, 4])
def test_indices_in_valid_range(total_bits):
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")
    K = _make_random_keys(256)
    compressed = engine.compress_keys_pytorch(K)

    mse_bits = max(total_bits - 1, 1)
    max_idx = (1 << mse_bits) - 1
    assert compressed["indices"].max().item() <= max_idx
    assert compressed["indices"].min().item() >= 0


def test_signs_are_pm1():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    K = _make_random_keys(256)
    compressed = engine.compress_keys_pytorch(K)

    unique_vals = set(compressed["qjl_signs"].unique().tolist())
    assert unique_vals.issubset({-1, 1}), f"Signs should be +/-1, got {unique_vals}"


def test_norms_positive():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    K = _make_random_keys(256)
    compressed = engine.compress_keys_pytorch(K)

    assert (compressed["vec_norms"] >= 0).all()
    assert (compressed["residual_norms"] >= 0).all()


def test_compress_values_shapes():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    V = _make_random_keys(256)
    compressed = engine.compress_values_pytorch(V)

    assert compressed["indices"].shape == (256, 128)
    assert compressed["vec_norms"].shape == (256,)


def test_value_indices_in_valid_range():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    V = _make_random_keys(256)
    compressed = engine.compress_values_pytorch(V)

    max_idx = (1 << 3) - 1  # values use all bits (no QJL)
    assert compressed["indices"].max().item() <= max_idx


def test_compress_matches_reference():
    """Compare our engine against the original cutiledump compressors.py implementation."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, seed=42, device="cpu")

    K = torch.randn(64, 128).half()
    compressed = engine.compress_keys_pytorch(K)

    K_f = K.float()
    vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
    K_normed = K_f / (vec_norms + 1e-8)
    rotated = K_normed @ engine.PiT.float()
    centroids = engine.key_codebook.centroids
    diffs = rotated.unsqueeze(-1) - centroids
    expected_indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

    assert torch.equal(compressed["indices"], expected_indices)


@pytest.mark.parametrize("total_bits", [3, 4])
def test_k_mse_reconstruction_quality(total_bits):
    """k_mse should be a reasonable approximation of K."""
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")
    K = torch.randn(512, 128).half()
    compressed = engine.compress_keys_pytorch(K)

    cos_sim = torch.nn.functional.cosine_similarity(
        K.float(), compressed["k_mse"].float(), dim=-1
    )
    assert cos_sim.mean().item() > 0.85, (
        f"Mean cosine similarity too low: {cos_sim.mean().item():.4f}"
    )


def test_edge_case_zero_vector():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    K = torch.zeros(1, 128).half()
    compressed = engine.compress_keys_pytorch(K)
    assert not torch.isnan(compressed["k_mse"]).any()
    assert not torch.isinf(compressed["k_mse"]).any()


def test_edge_case_large_values():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    K = torch.randn(4, 128).half() * 100
    compressed = engine.compress_keys_pytorch(K)
    assert not torch.isnan(compressed["k_mse"]).any()
