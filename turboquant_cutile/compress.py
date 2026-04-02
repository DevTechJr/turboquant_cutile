"""TurboQuant compression kernels: normalize, rotate, quantize, QJL."""

try:
    import cuda.tile as ct
except ImportError:
    import cutile as ct  # type: ignore

from .constants import BLOCK_S, HEAD_DIM


@ct.kernel
def turboquant_compress_2bit(
    K, Pi_T, Pi, S_T,
    Indices, Signs, Norms, RNorms,
    c0: float, c1: float, c2: float, c3: float,
    b1: float, b2: float, b3: float,
    seq_k: int,
):
    """2-bit MSE (4 centroids) + 1-bit QJL. total_bits=3."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    k_tile = ct.load(K, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)

    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    pi = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    s_t = ct.load(S_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    k_f32 = ct.astype(k_tile, ct.float32)
    norms = ct.sqrt(ct.sum(k_f32 * k_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    k_normed = k_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(k_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx > 0.5, c1, y_hat)
    y_hat = ct.where(idx > 1.5, c2, y_hat)
    y_hat = ct.where(idx > 2.5, c3, y_hat)

    k_bar_hat = ct.mma(ct.astype(y_hat, ct.float16), pi,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    k_mse = k_bar_hat * ct.expand_dims(norms, axis=1)

    residual = k_f32 - k_mse
    r_norms = ct.sqrt(ct.sum(residual * residual, axis=1))

    projected = ct.mma(ct.astype(residual, ct.float16), s_t,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    signs = ct.where(projected >= 0.0, 1.0, -1.0)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Signs, index=(block_id, 0), tile=ct.astype(signs, ct.int8))
    ct.store(Norms, index=(block_id,), tile=ct.astype(norms, ct.float16))
    ct.store(RNorms, index=(block_id,), tile=ct.astype(r_norms, ct.float16))


@ct.kernel
def turboquant_compress_3bit(
    K, Pi_T, Pi, S_T,
    Indices, Signs, Norms, RNorms,
    c0: float, c1: float, c2: float, c3: float,
    c4: float, c5: float, c6: float, c7: float,
    b1: float, b2: float, b3: float, b4: float,
    b5: float, b6: float, b7: float,
    seq_k: int,
):
    """3-bit MSE (8 centroids) + 1-bit QJL. total_bits=4."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    k_tile = ct.load(K, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)

    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    pi = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    s_t = ct.load(S_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    k_f32 = ct.astype(k_tile, ct.float32)
    norms = ct.sqrt(ct.sum(k_f32 * k_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    k_normed = k_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(k_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)
    idx = ct.where(y > b4, 4.0, idx)
    idx = ct.where(y > b5, 5.0, idx)
    idx = ct.where(y > b6, 6.0, idx)
    idx = ct.where(y > b7, 7.0, idx)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx > 0.5, c1, y_hat)
    y_hat = ct.where(idx > 1.5, c2, y_hat)
    y_hat = ct.where(idx > 2.5, c3, y_hat)
    y_hat = ct.where(idx > 3.5, c4, y_hat)
    y_hat = ct.where(idx > 4.5, c5, y_hat)
    y_hat = ct.where(idx > 5.5, c6, y_hat)
    y_hat = ct.where(idx > 6.5, c7, y_hat)

    k_bar_hat = ct.mma(ct.astype(y_hat, ct.float16), pi,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    k_mse = k_bar_hat * ct.expand_dims(norms, axis=1)

    residual = k_f32 - k_mse
    r_norms = ct.sqrt(ct.sum(residual * residual, axis=1))

    projected = ct.mma(ct.astype(residual, ct.float16), s_t,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    signs = ct.where(projected >= 0.0, 1.0, -1.0)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Signs, index=(block_id, 0), tile=ct.astype(signs, ct.int8))
    ct.store(Norms, index=(block_id,), tile=ct.astype(norms, ct.float16))
    ct.store(RNorms, index=(block_id,), tile=ct.astype(r_norms, ct.float16))


@ct.kernel
def turboquant_compress_values_3bit(
    V, Pi_T,
    Indices, Norms,
    c0: float, c1: float, c2: float, c3: float,
    c4: float, c5: float, c6: float, c7: float,
    b1: float, b2: float, b3: float, b4: float,
    b5: float, b6: float, b7: float,
    seq_v: int,
):
    """MSE-only value compression, 3-bit (8 levels). No QJL."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    v_tile = ct.load(V, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    v_f32 = ct.astype(v_tile, ct.float32)
    norms = ct.sqrt(ct.sum(v_f32 * v_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    v_normed = v_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(v_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)
    idx = ct.where(y > b4, 4.0, idx)
    idx = ct.where(y > b5, 5.0, idx)
    idx = ct.where(y > b6, 6.0, idx)
    idx = ct.where(y > b7, 7.0, idx)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Norms, index=(block_id,), tile=ct.astype(norms, ct.float16))


@ct.kernel
def turboquant_compress_values_2bit(
    V, Pi_T,
    Indices, Norms,
    c0: float, c1: float, c2: float, c3: float,
    b1: float, b2: float, b3: float,
    seq_v: int,
):
    """MSE-only value compression, 2-bit (4 levels)."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    v_tile = ct.load(V, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    v_f32 = ct.astype(v_tile, ct.float32)
    norms = ct.sqrt(ct.sum(v_f32 * v_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    v_normed = v_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(v_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Norms, index=(block_id,), tile=ct.astype(norms, ct.float16))
