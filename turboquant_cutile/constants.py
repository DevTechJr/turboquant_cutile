"""Shared constants for TurboQuant cuTile kernels."""

# Tile sizes tuned for Blackwell B200 Tensor Cores.
# MMA needs multiples of 16; 64-wide KV blocks fit well in SRAM.
HEAD_DIM = 128

BLOCK_Q = 16
BLOCK_KV = 64
BLOCK_S = 64

SUPPORTED_MSE_BITS = {1, 2, 3, 4}
DEFAULT_TOTAL_BITS = 3
DEFAULT_SEED = 42
