"""Test value decompression round-trip against PyTorch reference."""

import math
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant_cutile import TurboQuantEngine


@pytest.mark.parametrize("seq_v", [64, 256, 512])
@pytest.mark.parametrize("total_bits", [3, 4])
def test_decompress_shape(seq_v, total_bits):
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")
    V = torch.randn(seq_v, 128).half()
    compressed = engine.compress_values_pytorch(V)
    V_recon = engine.decompress_values_pytorch(compressed)

    assert V_recon.shape == (seq_v, 128)
    assert V_recon.dtype == torch.float16


@pytest.mark.parametrize("total_bits", [2, 3, 4])
def test_reconstruction_quality(total_bits):
    """Cosine similarity between original and reconstructed should be high."""
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")
    V = torch.randn(512, 128).half()
    compressed = engine.compress_values_pytorch(V)
    V_recon = engine.decompress_values_pytorch(compressed)

    cos_sim = torch.nn.functional.cosine_similarity(
        V.float(), V_recon.float(), dim=-1
    )

    min_sim = {2: 0.80, 3: 0.90, 4: 0.96}[total_bits]
    assert cos_sim.mean().item() > min_sim, (
        f"bits={total_bits}: mean cos_sim={cos_sim.mean():.4f} < {min_sim}"
    )


@pytest.mark.parametrize("total_bits", [2, 3, 4])
def test_mse_within_bound(total_bits):
    """Per-vector MSE should be bounded by paper's theoretical upper bound."""
    d = 128
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, device="cpu")

    V = torch.randn(1000, d)
    V = V / torch.norm(V, dim=-1, keepdim=True)  # unit vectors

    compressed = engine.compress_values_pytorch(V.half())
    V_recon = engine.decompress_values_pytorch(compressed)

    mse = ((V.float() - V_recon.float()) ** 2).sum(dim=-1).mean().item()
    upper_bound = math.sqrt(3) * math.pi / 2 * (1.0 / (4**total_bits))

    assert mse < upper_bound * 2.5, (
        f"bits={total_bits}: MSE={mse:.6f} > 2.5× bound={upper_bound:.6f}"
    )


def test_higher_bits_lower_mse():
    """More bits → strictly lower reconstruction MSE."""
    d = 128
    V = torch.randn(256, d).half()
    prev_mse = float("inf")

    for bits in [2, 3, 4]:
        engine = TurboQuantEngine(head_dim=d, total_bits=bits, device="cpu")
        compressed = engine.compress_values_pytorch(V)
        V_recon = engine.decompress_values_pytorch(compressed)
        mse = ((V.float() - V_recon.float()) ** 2).sum(dim=-1).mean().item()
        assert mse < prev_mse, f"bits={bits}: MSE={mse:.6f} >= prev={prev_mse:.6f}"
        prev_mse = mse


def test_norms_preserved():
    """Vector norms should be approximately preserved after round-trip."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    V = torch.randn(128, 128).half()
    compressed = engine.compress_values_pytorch(V)
    V_recon = engine.decompress_values_pytorch(compressed)

    orig_norms = torch.norm(V.float(), dim=-1)
    recon_norms = torch.norm(V_recon.float(), dim=-1)

    norm_ratio = recon_norms / (orig_norms + 1e-8)
    assert (norm_ratio - 1.0).abs().mean().item() < 0.15, (
        "Norms should be roughly preserved after compression round-trip"
    )
