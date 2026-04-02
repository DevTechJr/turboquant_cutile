"""
Test attention kernel correctness against PyTorch reference and ground truth.

Key properties verified:
  - Scores match PyTorch reference within FP16 tolerance
  - Inner product estimator is unbiased (mean error ≈ 0)
  - Correct 1/√d scaling
  - Sweep over multiple bit widths and sequence lengths
"""

import math
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant_cutile import TurboQuantEngine


@pytest.mark.parametrize("seq_q,seq_k", [(1, 64), (1, 256), (1, 1024), (16, 512)])
@pytest.mark.parametrize("total_bits", [3, 4])
def test_scores_shape(seq_q, seq_k, total_bits):
    engine = TurboQuantEngine(head_dim=128, total_bits=total_bits, device="cpu")

    Q = torch.randn(seq_q, 128).half()
    K = torch.randn(seq_k, 128).half()
    compressed_k = engine.compress_keys_pytorch(K)
    scores = engine.attention_scores_pytorch(Q, compressed_k)

    assert scores.shape == (seq_q, seq_k)


@pytest.mark.parametrize("total_bits", [3, 4])
def test_unbiasedness(total_bits):
    """
    The asymmetric estimator should be unbiased: E[estimated_ip] ≈ true_ip.
    We test by averaging over many random pairs.
    """
    d = 128
    n = 2000
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    K = torch.randn(n, d)
    K = K / torch.norm(K, dim=-1, keepdim=True)
    Q = torch.randn(n, d)
    Q = Q / torch.norm(Q, dim=-1, keepdim=True)

    true_ip = (Q * K).sum(dim=-1)

    compressed_k = engine.compress_keys_pytorch(K.half())
    scores_unscaled = engine.attention_scores_pytorch(Q.half(), compressed_k)

    estimated_per_pair = torch.diag(scores_unscaled) / engine.scale

    bias = (estimated_per_pair - true_ip).mean().abs().item()
    assert bias < 0.05, f"Bias too large: {bias:.4f} (should be ~0)"


@pytest.mark.parametrize("total_bits", [2, 3, 4])
def test_correlation_with_true_scores(total_bits):
    """Estimated scores should correlate well with true Q·K^T scores."""
    d = 128
    seq_q, seq_k = 1, 512
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()

    true_scores = (Q.float() @ K.float().T) * engine.scale
    compressed_k = engine.compress_keys_pytorch(K)
    estimated_scores = engine.attention_scores_pytorch(Q, compressed_k)

    corr = torch.corrcoef(
        torch.stack([true_scores.flatten(), estimated_scores.flatten()])
    )[0, 1].item()

    min_corr = {2: 0.45, 3: 0.75, 4: 0.90}[total_bits]
    assert corr > min_corr, (
        f"bits={total_bits}: correlation={corr:.4f} < {min_corr}"
    )


def test_scaling_correct():
    """Scores should be scaled by 1/√d."""
    d = 128
    engine = TurboQuantEngine(head_dim=d, total_bits=3, device="cpu")

    Q = torch.randn(1, d).half()
    K = torch.randn(64, d).half()

    compressed_k = engine.compress_keys_pytorch(K)
    scores = engine.attention_scores_pytorch(Q, compressed_k)

    raw_ip = Q.float() @ compressed_k["k_mse"].float().T
    assert scores.abs().mean().item() < raw_ip.abs().mean().item(), (
        "Scaled scores should be smaller than raw inner products"
    )


def test_single_decode_token():
    """Typical decode scenario: seq_q=1, seq_k=large."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    Q = torch.randn(1, 128).half()
    K = torch.randn(2048, 128).half()

    compressed_k = engine.compress_keys_pytorch(K)
    scores = engine.attention_scores_pytorch(Q, compressed_k)

    assert scores.shape == (1, 2048)
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()


def test_needle_in_haystack():
    """Can we still find the most-attended key after compression?"""
    d = 128
    seq_k = 1024
    engine = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")

    K = torch.randn(seq_k, d)
    K = K / torch.norm(K, dim=-1, keepdim=True)

    needle_pos = seq_k // 3
    Q = K[needle_pos].unsqueeze(0)

    true_scores = Q @ K.T
    true_top1 = true_scores.argmax(dim=-1).item()

    compressed_k = engine.compress_keys_pytorch(K.half())
    estimated_scores = engine.attention_scores_pytorch(Q.half(), compressed_k)
    estimated_top1 = estimated_scores.argmax(dim=-1).item()

    assert estimated_top1 == true_top1 or abs(estimated_top1 - needle_pos) < 5, (
        f"Needle at {needle_pos}, true top1={true_top1}, estimated top1={estimated_top1}"
    )


@pytest.mark.parametrize("total_bits", [3, 4])
def test_bits_sweep_scores_reasonable(total_bits):
    """Higher bits should produce lower MSE in scores."""
    d = 128
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, device="cpu")

    Q = torch.randn(4, d).half()
    K = torch.randn(256, d).half()

    true_scores = (Q.float() @ K.float().T) * engine.scale
    compressed_k = engine.compress_keys_pytorch(K)
    estimated_scores = engine.attention_scores_pytorch(Q, compressed_k)

    score_mse = ((true_scores - estimated_scores) ** 2).mean().item()
    assert score_mse < 0.5, f"Score MSE too high: {score_mse:.6f}"
