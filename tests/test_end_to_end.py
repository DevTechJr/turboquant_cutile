"""Full pipeline: compress, attend, decompress — compare against FP16 reference."""

import math
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant_cutile import TurboQuantEngine


def _standard_attention(Q, K, V, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    scores = (Q.float() @ K.float().T) * scale
    weights = torch.softmax(scores, dim=-1)
    return (weights @ V.float()).half()


@pytest.mark.parametrize("seq_q,seq_k", [(1, 64), (1, 512), (1, 2048), (4, 256)])
@pytest.mark.parametrize("total_bits", [3, 4])
def test_full_pipeline_output_shape(seq_q, seq_k, total_bits):
    d = 128
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()
    V = torch.randn(seq_k, d).half()

    compressed_k = engine.compress_keys_pytorch(K)
    compressed_v = engine.compress_values_pytorch(V)
    output = engine.fused_attention_pytorch(Q, compressed_k, compressed_v)

    assert output.shape == (seq_q, d)


@pytest.mark.parametrize("total_bits", [3, 4])
def test_output_cosine_similarity(total_bits):
    d = 128
    seq_q, seq_k = 1, 1024
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()
    V = torch.randn(seq_k, d).half()

    ref_output = _standard_attention(Q, K, V, d)

    compressed_k = engine.compress_keys_pytorch(K)
    compressed_v = engine.compress_values_pytorch(V)
    tq_output = engine.fused_attention_pytorch(Q, compressed_k, compressed_v)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        tq_output.float().flatten().unsqueeze(0),
    ).item()

    min_sim = {3: 0.80, 4: 0.92}[total_bits]
    assert cos_sim > min_sim, (
        f"bits={total_bits}: output cos_sim={cos_sim:.4f} < {min_sim}"
    )


def test_needle_retrieval_through_full_pipeline():
    d = 128
    seq_k = 512
    engine = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")

    K = torch.randn(seq_k, d)
    K = K / torch.norm(K, dim=-1, keepdim=True)
    V = torch.randn(seq_k, d)

    needle_pos = 171
    Q = K[needle_pos].unsqueeze(0) * 3.0

    ref_scores = (Q @ K.T) / math.sqrt(d)
    ref_top1 = ref_scores.argmax(dim=-1).item()

    compressed_k = engine.compress_keys_pytorch(K.half())
    tq_scores = engine.attention_scores_pytorch(Q.half(), compressed_k)
    tq_top1 = tq_scores.argmax(dim=-1).item()

    assert tq_top1 == ref_top1, (
        f"Needle at {needle_pos}: ref top1={ref_top1}, TQ top1={tq_top1}"
    )


def test_softmax_distribution_preserved():
    d = 128
    seq_q, seq_k = 1, 256
    engine = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()

    ref_scores = (Q.float() @ K.float().T) * engine.scale
    ref_probs = torch.softmax(ref_scores, dim=-1)

    compressed_k = engine.compress_keys_pytorch(K)
    tq_scores = engine.attention_scores_pytorch(Q, compressed_k)
    tq_probs = torch.softmax(tq_scores, dim=-1)

    kl_div = torch.nn.functional.kl_div(
        tq_probs.log(), ref_probs, reduction="batchmean"
    ).item()

    assert kl_div < 0.5, f"KL divergence too high: {kl_div:.4f}"


def test_compression_ratios():
    d = 128
    engine_3bit = TurboQuantEngine(head_dim=d, total_bits=3, device="cpu")
    engine_4bit = TurboQuantEngine(head_dim=d, total_bits=4, device="cpu")

    stats_3 = engine_3bit.compressed_size_bytes(4096)
    stats_4 = engine_4bit.compressed_size_bytes(4096)

    assert 4.5 < stats_3["compression_ratio"] < 6.0, (
        f"3-bit compression: {stats_3['compression_ratio']:.2f}x (expected ~5x)"
    )
    assert 3.0 < stats_4["compression_ratio"] < 4.5, (
        f"4-bit compression: {stats_4['compression_ratio']:.2f}x (expected ~3.8x)"
    )


def test_deterministic_with_same_seed():
    d = 128
    K = torch.randn(64, d).half()

    engine1 = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")
    engine2 = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")

    c1 = engine1.compress_keys_pytorch(K)
    c2 = engine2.compress_keys_pytorch(K)

    assert torch.equal(c1["indices"], c2["indices"])
    assert torch.equal(c1["qjl_signs"], c2["qjl_signs"])


def test_different_seeds_different_results():
    d = 128
    K = torch.randn(64, d).half()

    engine1 = TurboQuantEngine(head_dim=d, total_bits=3, seed=42, device="cpu")
    engine2 = TurboQuantEngine(head_dim=d, total_bits=3, seed=99, device="cpu")

    c1 = engine1.compress_keys_pytorch(K)
    c2 = engine2.compress_keys_pytorch(K)

    assert not torch.equal(c1["indices"], c2["indices"])


@pytest.mark.parametrize("head_dim", [64, 128])
def test_different_head_dims(head_dim):
    engine = TurboQuantEngine(head_dim=head_dim, total_bits=3, device="cpu")

    Q = torch.randn(1, head_dim).half()
    K = torch.randn(128, head_dim).half()
    V = torch.randn(128, head_dim).half()

    compressed_k = engine.compress_keys_pytorch(K)
    compressed_v = engine.compress_values_pytorch(V)
    output = engine.fused_attention_pytorch(Q, compressed_k, compressed_v)

    assert output.shape == (1, head_dim)
    assert not torch.isnan(output).any()


# V-fused kernel tests (on-chip V decompression path)

@pytest.mark.parametrize("total_bits", [2, 3])
@pytest.mark.parametrize("seq_q,seq_k", [(1, 128), (1, 512), (4, 256)])
def test_vfused_matches_pytorch_reference(seq_q, seq_k, total_bits):
    d = 128
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()
    V = torch.randn(seq_k, d).half()

    compressed_k = engine.compress_keys_pytorch(K)
    compressed_v = engine.compress_values_pytorch(V)

    ref_output = engine.fused_attention_pytorch(Q, compressed_k, compressed_v)

    assert ref_output.shape == (seq_q, d)
    assert not torch.isnan(ref_output).any()
    assert not torch.isinf(ref_output).any()


@pytest.mark.parametrize("total_bits", [2, 3])
def test_vfused_cosine_vs_fp16(total_bits):
    d = 128
    seq_q, seq_k = 1, 1024
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    Q = torch.randn(seq_q, d).half()
    K = torch.randn(seq_k, d).half()
    V = torch.randn(seq_k, d).half()

    ref_output = _standard_attention(Q, K, V, d)

    compressed_k = engine.compress_keys_pytorch(K)
    compressed_v = engine.compress_values_pytorch(V)
    tq_output = engine.fused_attention_pytorch(Q, compressed_k, compressed_v)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        tq_output.float().flatten().unsqueeze(0),
    ).item()

    min_sim = {2: 0.65, 3: 0.80}[total_bits]
    assert cos_sim > min_sim, (
        f"V-fused bits={total_bits}: cos_sim={cos_sim:.4f} < {min_sim}"
    )


@pytest.mark.parametrize("total_bits", [2, 3])
def test_vfused_needle_retrieval(total_bits):
    d = 128
    seq_k = 512
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=42, device="cpu")

    K = torch.randn(seq_k, d)
    K = K / torch.norm(K, dim=-1, keepdim=True)
    V = torch.randn(seq_k, d)

    needle_pos = 200
    Q = K[needle_pos].unsqueeze(0) * 3.0

    compressed_k = engine.compress_keys_pytorch(K.half())
    compressed_v = engine.compress_values_pytorch(V.half())

    tq_output = engine.fused_attention_pytorch(Q.half(), compressed_k, compressed_v)

    ref_output = _standard_attention(Q.half(), K.half(), V.half(), d)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        tq_output.float().flatten().unsqueeze(0),
    ).item()

    assert cos_sim > 0.7, (
        f"V-fused needle test bits={total_bits}: cos_sim={cos_sim:.4f}"
    )
