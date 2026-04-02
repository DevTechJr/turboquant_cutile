"""Value decompression: v_hat = norm * (centroids[idx] @ Pi)."""

try:
    import cuda.tile as ct
except ImportError:
    import cutile as ct  # type: ignore

from .constants import BLOCK_S, HEAD_DIM


@ct.kernel
def turboquant_decompress_3bit(
    Indices, Norms, Pi, Output,
    c0: float, c1: float, c2: float, c3: float,
    c4: float, c5: float, c6: float, c7: float,
    seq_v: int,
):
    """8-level centroid lookup, un-rotate, rescale."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    idx_tile = ct.load(Indices, index=(block_id, 0),
                       shape=(BLOCK_S, HEAD_DIM), padding_mode=zero_pad)
    norm_tile = ct.load(Norms, index=(block_id,), shape=(BLOCK_S,),
                        padding_mode=zero_pad)
    pi = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    idx_f32 = ct.astype(idx_tile, ct.float32)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx_f32 > 0.5, c1, y_hat)
    y_hat = ct.where(idx_f32 > 1.5, c2, y_hat)
    y_hat = ct.where(idx_f32 > 2.5, c3, y_hat)
    y_hat = ct.where(idx_f32 > 3.5, c4, y_hat)
    y_hat = ct.where(idx_f32 > 4.5, c5, y_hat)
    y_hat = ct.where(idx_f32 > 5.5, c6, y_hat)
    y_hat = ct.where(idx_f32 > 6.5, c7, y_hat)

    reconstructed = ct.mma(ct.astype(y_hat, ct.float16), pi,
                           ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    norm_f32 = ct.astype(norm_tile, ct.float32)
    output = reconstructed * ct.expand_dims(norm_f32, axis=1)

    ct.store(Output, index=(block_id, 0), tile=ct.astype(output, ct.float16))


@ct.kernel
def turboquant_decompress_2bit(
    Indices, Norms, Pi, Output,
    c0: float, c1: float, c2: float, c3: float,
    seq_v: int,
):
    """4-level centroid lookup, un-rotate, rescale."""
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    idx_tile = ct.load(Indices, index=(block_id, 0),
                       shape=(BLOCK_S, HEAD_DIM), padding_mode=zero_pad)
    norm_tile = ct.load(Norms, index=(block_id,), shape=(BLOCK_S,),
                        padding_mode=zero_pad)
    pi = ct.load(Pi, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    idx_f32 = ct.astype(idx_tile, ct.float32)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx_f32 > 0.5, c1, y_hat)
    y_hat = ct.where(idx_f32 > 1.5, c2, y_hat)
    y_hat = ct.where(idx_f32 > 2.5, c3, y_hat)

    reconstructed = ct.mma(ct.astype(y_hat, ct.float16), pi,
                           ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    norm_f32 = ct.astype(norm_tile, ct.float32)
    output = reconstructed * ct.expand_dims(norm_f32, axis=1)

    ct.store(Output, index=(block_id, 0), tile=ct.astype(output, ct.float16))
