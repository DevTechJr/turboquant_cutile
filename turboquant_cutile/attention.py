"""
Asymmetric attention kernels for compressed keys.

score(q, k) ~ <q, k_mse> + ||r|| * sqrt(pi/2)/m * <S*q, signs>

Score-only variant outputs raw scores; fused variant does online softmax + V
accumulation. USE_SWIZZLE enables Blackwell fast paths (exp2 ftz, approx div,
pipelined loads, block interleaving).
"""

import math

try:
    import cuda.tile as ct
    from cuda.tile import RoundingMode as RMd
except ImportError:
    import cutile as ct  # type: ignore
    RMd = None

from .constants import BLOCK_Q, BLOCK_KV, HEAD_DIM

INV_LOG_2 = 1.0 / math.log(2)
ConstBool = ct.Constant[bool]


@ct.kernel(occupancy=2)
def turboquant_attention_scores(
    Q, K_mse, Signs, R_norms, Q_proj, Output,
    scale: float,
    correction_scale: float,
    seq_k: int,
    USE_SWIZZLE: ConstBool,
):
    """Score-only kernel. One program per query block, streams over KV."""
    q_block = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    q_tile = ct.load(Q, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                     padding_mode=zero_pad)
    qp_tile = ct.load(Q_proj, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                      padding_mode=zero_pad)

    num_kv_blocks = ct.num_tiles(K_mse, axis=0, shape=(BLOCK_KV, HEAD_DIM))

    for kv_block in range(num_kv_blocks):
        if USE_SWIZZLE:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad,
                             latency=2)
        else:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)

        term1 = ct.mma(q_tile, ct.transpose(k_tile),
                       ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))

        s_tile = ct.load(Signs, index=(kv_block, 0),
                         shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        s_float = ct.astype(s_tile, ct.float16)

        qjl_ip = ct.mma(qp_tile, ct.transpose(s_float),
                        ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))

        rn = ct.load(R_norms, index=(kv_block,), shape=(BLOCK_KV,),
                     padding_mode=zero_pad)
        rn_f32 = ct.astype(rn, ct.float32)

        term2 = correction_scale * qjl_ip * ct.expand_dims(rn_f32, axis=0)

        scores = (term1 + term2) * scale
        ct.store(Output, index=(q_block, kv_block), tile=scores)


@ct.kernel(occupancy=2)
def turboquant_fused_attention(
    Q, K_mse, Signs, R_norms, Q_proj, V, Output,
    scale: float,
    correction_scale: float,
    seq_k: int,
    USE_SWIZZLE: ConstBool,
):
    """Fused attention + online softmax. V is pre-decompressed FP16."""
    zero_pad = ct.PaddingMode.ZERO

    if USE_SWIZZLE:
        half = ct.cdiv(ct.num_blocks(0), 2)
        q_block = ct.bid(0) // 2 + (ct.bid(0) % 2) * half
    else:
        q_block = ct.bid(0)

    q_tile = ct.load(Q, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                     padding_mode=zero_pad)
    qp_tile = ct.load(Q_proj, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                      padding_mode=zero_pad)

    m_i = ct.full((BLOCK_Q,), -1e30, dtype=ct.float32)
    l_i = ct.zeros((BLOCK_Q,), dtype=ct.float32)
    acc = ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32)

    scale_log2 = scale * INV_LOG_2

    num_kv_blocks = ct.num_tiles(K_mse, axis=0, shape=(BLOCK_KV, HEAD_DIM))

    for kv_block in range(num_kv_blocks):
        if USE_SWIZZLE:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad,
                             latency=2)
            v_tile = ct.load(V, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad,
                             latency=4)
        else:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
            v_tile = ct.load(V, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)

        s_tile = ct.load(Signs, index=(kv_block, 0),
                         shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        rn = ct.load(R_norms, index=(kv_block,), shape=(BLOCK_KV,),
                     padding_mode=zero_pad)

        term1 = ct.mma(q_tile, ct.transpose(k_tile),
                       ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))

        s_float = ct.astype(s_tile, ct.float16)
        qjl_ip = ct.mma(qp_tile, ct.transpose(s_float),
                        ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))
        rn_f32 = ct.astype(rn, ct.float32)
        term2 = correction_scale * qjl_ip * ct.expand_dims(rn_f32, axis=0)

        raw_scores = term1 + term2

        if USE_SWIZZLE:
            m_new = ct.maximum(m_i, ct.max(raw_scores, axis=1) * scale_log2)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)
            p = ct.exp2(raw_scores * scale_log2 - ct.expand_dims(m_new, axis=1),
                        flush_to_zero=True)
        else:
            scores = raw_scores * scale
            m_new = ct.maximum(m_i, ct.max(scores, axis=1))
            alpha = ct.exp(m_i - m_new)
            p = ct.exp(scores - ct.expand_dims(m_new, axis=1))

        l_i = alpha * l_i + ct.sum(p, axis=1)

        p_fp16 = ct.astype(p, ct.float16)
        acc = ct.expand_dims(alpha, axis=1) * acc + ct.mma(
            p_fp16, v_tile, ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32))

        m_i = m_new

    if USE_SWIZZLE:
        out = ct.truediv(acc, ct.expand_dims(l_i, axis=1),
                         flush_to_zero=True, rounding_mode=RMd.APPROX)
    else:
        out = acc / ct.expand_dims(l_i, axis=1)

    ct.store(Output, index=(q_block, 0), tile=out)


@ct.kernel(occupancy=2)
def turboquant_fused_attention_vfused_3bit(
    Q, K_mse, Signs, R_norms, Q_proj,
    V_Indices, V_Norms, Pi, Output,
    scale: float,
    correction_scale: float,
    seq_k: int,
    vc0: float, vc1: float, vc2: float, vc3: float,
    vc4: float, vc5: float, vc6: float, vc7: float,
    USE_SWIZZLE: ConstBool,
):
    """Fused attention + online softmax, decompresses 3-bit V on-chip."""
    zero_pad = ct.PaddingMode.ZERO

    if USE_SWIZZLE:
        half = ct.cdiv(ct.num_blocks(0), 2)
        q_block = ct.bid(0) // 2 + (ct.bid(0) % 2) * half
    else:
        q_block = ct.bid(0)

    q_tile = ct.load(Q, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                     padding_mode=zero_pad)
    qp_tile = ct.load(Q_proj, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                      padding_mode=zero_pad)

    # Pi stays resident across all KV blocks
    pi_tile = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    m_i = ct.full((BLOCK_Q,), -1e30, dtype=ct.float32)
    l_i = ct.zeros((BLOCK_Q,), dtype=ct.float32)
    acc = ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32)

    scale_log2 = scale * INV_LOG_2

    num_kv_blocks = ct.num_tiles(K_mse, axis=0, shape=(BLOCK_KV, HEAD_DIM))

    for kv_block in range(num_kv_blocks):
        if USE_SWIZZLE:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad,
                             latency=2)
        else:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)

        # decompress V on-chip: indices -> centroids -> un-rotate -> scale
        v_idx = ct.load(V_Indices, index=(kv_block, 0),
                        shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        v_nrm = ct.load(V_Norms, index=(kv_block,),
                        shape=(BLOCK_KV,), padding_mode=zero_pad)

        vi_f32 = ct.astype(v_idx, ct.float32)
        y_hat = ct.full((BLOCK_KV, HEAD_DIM), vc0, dtype=ct.float32)
        y_hat = ct.where(vi_f32 > 0.5, vc1, y_hat)
        y_hat = ct.where(vi_f32 > 1.5, vc2, y_hat)
        y_hat = ct.where(vi_f32 > 2.5, vc3, y_hat)
        y_hat = ct.where(vi_f32 > 3.5, vc4, y_hat)
        y_hat = ct.where(vi_f32 > 4.5, vc5, y_hat)
        y_hat = ct.where(vi_f32 > 5.5, vc6, y_hat)
        y_hat = ct.where(vi_f32 > 6.5, vc7, y_hat)

        v_recon = ct.mma(ct.astype(y_hat, ct.float16), pi_tile,
                         ct.zeros((BLOCK_KV, HEAD_DIM), dtype=ct.float32))
        v_nrm_f32 = ct.astype(v_nrm, ct.float32)
        v_tile = ct.astype(
            v_recon * ct.expand_dims(v_nrm_f32, axis=1), ct.float16)

        s_tile = ct.load(Signs, index=(kv_block, 0),
                         shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        rn = ct.load(R_norms, index=(kv_block,), shape=(BLOCK_KV,),
                     padding_mode=zero_pad)

        term1 = ct.mma(q_tile, ct.transpose(k_tile),
                       ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))

        s_float = ct.astype(s_tile, ct.float16)
        qjl_ip = ct.mma(qp_tile, ct.transpose(s_float),
                        ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))
        rn_f32 = ct.astype(rn, ct.float32)
        term2 = correction_scale * qjl_ip * ct.expand_dims(rn_f32, axis=0)

        raw_scores = term1 + term2

        if USE_SWIZZLE:
            m_new = ct.maximum(m_i, ct.max(raw_scores, axis=1) * scale_log2)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)
            p = ct.exp2(raw_scores * scale_log2 - ct.expand_dims(m_new, axis=1),
                        flush_to_zero=True)
        else:
            scores = raw_scores * scale
            m_new = ct.maximum(m_i, ct.max(scores, axis=1))
            alpha = ct.exp(m_i - m_new)
            p = ct.exp(scores - ct.expand_dims(m_new, axis=1))

        l_i = alpha * l_i + ct.sum(p, axis=1)

        p_fp16 = ct.astype(p, ct.float16)
        acc = ct.expand_dims(alpha, axis=1) * acc + ct.mma(
            p_fp16, v_tile, ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32))

        m_i = m_new

    if USE_SWIZZLE:
        out = ct.truediv(acc, ct.expand_dims(l_i, axis=1),
                         flush_to_zero=True, rounding_mode=RMd.APPROX)
    else:
        out = acc / ct.expand_dims(l_i, axis=1)

    ct.store(Output, index=(q_block, 0), tile=out)


@ct.kernel(occupancy=2)
def turboquant_fused_attention_vfused_2bit(
    Q, K_mse, Signs, R_norms, Q_proj,
    V_Indices, V_Norms, Pi, Output,
    scale: float,
    correction_scale: float,
    seq_k: int,
    vc0: float, vc1: float, vc2: float, vc3: float,
    USE_SWIZZLE: ConstBool,
):
    """Fused attention + online softmax, decompresses 2-bit V on-chip."""
    zero_pad = ct.PaddingMode.ZERO

    if USE_SWIZZLE:
        half = ct.cdiv(ct.num_blocks(0), 2)
        q_block = ct.bid(0) // 2 + (ct.bid(0) % 2) * half
    else:
        q_block = ct.bid(0)

    q_tile = ct.load(Q, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                     padding_mode=zero_pad)
    qp_tile = ct.load(Q_proj, index=(q_block, 0), shape=(BLOCK_Q, HEAD_DIM),
                      padding_mode=zero_pad)

    pi_tile = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    m_i = ct.full((BLOCK_Q,), -1e30, dtype=ct.float32)
    l_i = ct.zeros((BLOCK_Q,), dtype=ct.float32)
    acc = ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32)

    scale_log2 = scale * INV_LOG_2

    num_kv_blocks = ct.num_tiles(K_mse, axis=0, shape=(BLOCK_KV, HEAD_DIM))

    for kv_block in range(num_kv_blocks):
        if USE_SWIZZLE:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad,
                             latency=2)
        else:
            k_tile = ct.load(K_mse, index=(kv_block, 0),
                             shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)

        v_idx = ct.load(V_Indices, index=(kv_block, 0),
                        shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        v_nrm = ct.load(V_Norms, index=(kv_block,),
                        shape=(BLOCK_KV,), padding_mode=zero_pad)

        vi_f32 = ct.astype(v_idx, ct.float32)
        y_hat = ct.full((BLOCK_KV, HEAD_DIM), vc0, dtype=ct.float32)
        y_hat = ct.where(vi_f32 > 0.5, vc1, y_hat)
        y_hat = ct.where(vi_f32 > 1.5, vc2, y_hat)
        y_hat = ct.where(vi_f32 > 2.5, vc3, y_hat)

        v_recon = ct.mma(ct.astype(y_hat, ct.float16), pi_tile,
                         ct.zeros((BLOCK_KV, HEAD_DIM), dtype=ct.float32))
        v_nrm_f32 = ct.astype(v_nrm, ct.float32)
        v_tile = ct.astype(
            v_recon * ct.expand_dims(v_nrm_f32, axis=1), ct.float16)

        s_tile = ct.load(Signs, index=(kv_block, 0),
                         shape=(BLOCK_KV, HEAD_DIM), padding_mode=zero_pad)
        rn = ct.load(R_norms, index=(kv_block,), shape=(BLOCK_KV,),
                     padding_mode=zero_pad)

        term1 = ct.mma(q_tile, ct.transpose(k_tile),
                       ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))

        s_float = ct.astype(s_tile, ct.float16)
        qjl_ip = ct.mma(qp_tile, ct.transpose(s_float),
                        ct.zeros((BLOCK_Q, BLOCK_KV), dtype=ct.float32))
        rn_f32 = ct.astype(rn, ct.float32)
        term2 = correction_scale * qjl_ip * ct.expand_dims(rn_f32, axis=0)

        raw_scores = term1 + term2

        if USE_SWIZZLE:
            m_new = ct.maximum(m_i, ct.max(raw_scores, axis=1) * scale_log2)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)
            p = ct.exp2(raw_scores * scale_log2 - ct.expand_dims(m_new, axis=1),
                        flush_to_zero=True)
        else:
            scores = raw_scores * scale
            m_new = ct.maximum(m_i, ct.max(scores, axis=1))
            alpha = ct.exp(m_i - m_new)
            p = ct.exp(scores - ct.expand_dims(m_new, axis=1))

        l_i = alpha * l_i + ct.sum(p, axis=1)

        p_fp16 = ct.astype(p, ct.float16)
        acc = ct.expand_dims(alpha, axis=1) * acc + ct.mma(
            p_fp16, v_tile, ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32))

        m_i = m_new

    if USE_SWIZZLE:
        out = ct.truediv(acc, ct.expand_dims(l_i, axis=1),
                         flush_to_zero=True, rounding_mode=RMd.APPROX)
    else:
        out = acc / ct.expand_dims(l_i, axis=1)

    ct.store(Output, index=(q_block, 0), tile=out)
