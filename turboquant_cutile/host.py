"""Host-side engine: buffer allocation, kernel dispatch, PyTorch fallbacks."""

import math
import torch

from .codebook import LloydMaxCodebook
from .constants import ( BLOCK_Q, BLOCK_S, DEFAULT_SEED, DEFAULT_TOTAL_BITS, HEAD_DIM, )


def _generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).to(device)


def _generate_qjl_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Random Gaussian projection matrix for QJL."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 10000)
    return torch.randn(d, d, generator=gen).to(device)


class TurboQuantEngine:
    """Precomputed state (Pi, S, codebook) + kernel launch + PyTorch fallbacks."""

    def __init__( self, head_dim: int = HEAD_DIM, total_bits: int = DEFAULT_TOTAL_BITS, seed: int = DEFAULT_SEED, device: str = "cpu", ):
        self.head_dim = head_dim
        self.total_bits = total_bits
        self.mse_bits = max(total_bits - 1, 1)
        self.device = device

        self.Pi = _generate_rotation_matrix(head_dim, seed, device)
        self.PiT = self.Pi.T.contiguous()
        self.S = _generate_qjl_matrix(head_dim, seed, device)
        self.ST = self.S.T.contiguous()

        self.key_codebook = LloydMaxCodebook(head_dim, self.mse_bits)
        self.val_codebook = LloydMaxCodebook(head_dim, total_bits)

        self.scale = 1.0 / math.sqrt(head_dim)
        self.correction_scale = math.sqrt(math.pi / 2) / head_dim

    @torch.no_grad()
    def compress_keys_pytorch(self, K: torch.Tensor) -> dict:
        """K: (seq_k, head_dim) -> compressed dict."""
        K_f = K.float()
        vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
        K_normed = K_f / (vec_norms + 1e-8)

        rotated = K_normed @ self.PiT.float()

        centroids = self.key_codebook.centroids.to(K.device)
        diffs = rotated.unsqueeze(-1) - centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        y_hat = centroids[indices.long()]
        k_mse = (y_hat @ self.Pi.float()) * vec_norms

        residual = K_f - k_mse
        residual_norms = torch.norm(residual, dim=-1)

        projected = residual @ self.ST.float()
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1

        return {
            "indices": indices,
            "k_mse": k_mse.half(),
            "qjl_signs": signs,
            "vec_norms": vec_norms.squeeze(-1).half(),
            "residual_norms": residual_norms.half(),
        }

    @torch.no_grad()
    def compress_values_pytorch(self, V: torch.Tensor) -> dict:
        """V: (seq_v, head_dim) -> compressed dict (MSE only, no QJL)."""
        V_f = V.float()
        vec_norms = torch.norm(V_f, dim=-1, keepdim=True)
        V_normed = V_f / (vec_norms + 1e-8)

        rotated = V_normed @ self.PiT.float()

        centroids = self.val_codebook.centroids.to(V.device)
        diffs = rotated.unsqueeze(-1) - centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        return {
            "indices": indices,
            "vec_norms": vec_norms.squeeze(-1).half(),
        }

    @torch.no_grad()
    def decompress_values_pytorch(self, compressed_v: dict) -> torch.Tensor:
        centroids = self.val_codebook.centroids.to(compressed_v["indices"].device)
        y_hat = centroids[compressed_v["indices"].long()]
        reconstructed = (y_hat @ self.Pi.float()) * compressed_v["vec_norms"].float().unsqueeze(-1)
        return reconstructed.half()

    @torch.no_grad()
    def attention_scores_pytorch( self, Q: torch.Tensor, compressed_k: dict ) -> torch.Tensor:
        """Asymmetric estimator: term1 (MSE dot) + term2 (QJL correction)."""
        Q_f = Q.float()
        k_mse = compressed_k["k_mse"].float()
        signs = compressed_k["qjl_signs"].float()
        r_norms = compressed_k["residual_norms"].float()

        term1 = Q_f @ k_mse.T

        q_proj = Q_f @ self.S.T.float()
        qjl_ip = q_proj @ signs.T

        term2 = self.correction_scale * qjl_ip * r_norms.unsqueeze(0)

        return (term1 + term2) * self.scale

    @torch.no_grad()
    def fused_attention_pytorch( self, Q: torch.Tensor, compressed_k: dict, compressed_v: dict, ) -> torch.Tensor:
        """Full pipeline: scores -> softmax -> weighted V sum."""
        scores = self.attention_scores_pytorch(Q, compressed_k)
        weights = torch.softmax(scores, dim=-1)
        V_recon = self.decompress_values_pytorch(compressed_v).float()
        return (weights @ V_recon).half()

    def _cdiv(self, a: int, b: int) -> int:
        return (a + b - 1) // b

    @torch.no_grad()
    def launch_compress_keys(self, K: torch.Tensor) -> dict:
        try:
            from .compress import turboquant_compress_2bit, turboquant_compress_3bit
            import cuda.tile as ct
        except ImportError:
            return self.compress_keys_pytorch(K)

        seq_k, d = K.shape
        assert d == self.head_dim

        indices = torch.empty(seq_k, d, dtype=torch.uint8, device=K.device)
        signs = torch.empty(seq_k, d, dtype=torch.int8, device=K.device)
        norms = torch.empty(seq_k, dtype=torch.float16, device=K.device)
        r_norms = torch.empty(seq_k, dtype=torch.float16, device=K.device)

        grid = (self._cdiv(seq_k, BLOCK_S), 1, 1)
        centroids = self.key_codebook.centroids.tolist()
        boundaries = self.key_codebook.boundaries.tolist()
        stream = torch.cuda.current_stream()

        if self.mse_bits == 2:
            ct.launch(stream, grid, turboquant_compress_2bit, ( K, self.PiT.half(), self.Pi.half(), self.ST.half(), indices, signs, norms, r_norms, *centroids, *boundaries, seq_k, ))
        elif self.mse_bits == 3:
            ct.launch(stream, grid, turboquant_compress_3bit, ( K, self.PiT.half(), self.Pi.half(), self.ST.half(), indices, signs, norms, r_norms, *centroids, *boundaries, seq_k, ))
        else:
            return self.compress_keys_pytorch(K)

        k_mse = self._dequant_keys_from_indices(indices, norms)

        return {
            "indices": indices,
            "k_mse": k_mse,
            "qjl_signs": signs,
            "vec_norms": norms,
            "residual_norms": r_norms,
        }

    def _dequant_keys_from_indices( self, indices: torch.Tensor, norms: torch.Tensor ) -> torch.Tensor:
        """indices -> centroids -> un-rotate -> rescale."""
        centroids = self.key_codebook.centroids.to(indices.device)
        y_hat = centroids[indices.long()]
        k_mse = (y_hat.float() @ self.Pi.float()) * norms.float().unsqueeze(-1)
        return k_mse.half()

    @torch.no_grad()
    def launch_compress_values(self, V: torch.Tensor) -> dict:
        try:
            from .compress import ( turboquant_compress_values_3bit, turboquant_compress_values_2bit, )
            import cuda.tile as ct
        except ImportError:
            return self.compress_values_pytorch(V)

        seq_v, d = V.shape
        assert d == self.head_dim

        indices = torch.empty(seq_v, d, dtype=torch.uint8, device=V.device)
        norms = torch.empty(seq_v, dtype=torch.float16, device=V.device)

        grid = (self._cdiv(seq_v, BLOCK_S), 1, 1)
        centroids = self.val_codebook.centroids.tolist()
        boundaries = self.val_codebook.boundaries.tolist()
        stream = torch.cuda.current_stream()

        if self.total_bits == 3:
            ct.launch(stream, grid, turboquant_compress_values_3bit, ( V, self.PiT.half(), indices, norms, *centroids, *boundaries, seq_v, ))
        elif self.total_bits == 2:
            ct.launch(stream, grid, turboquant_compress_values_2bit, ( V, self.PiT.half(), indices, norms, *centroids, *boundaries, seq_v, ))
        else:
            return self.compress_values_pytorch(V)

        return {
            "indices": indices,
            "vec_norms": norms,
        }

    @torch.no_grad()
    def launch_decompress_values(self, compressed_v: dict) -> torch.Tensor:
        try:
            from .decompress import turboquant_decompress_3bit, turboquant_decompress_2bit
            import cuda.tile as ct
        except ImportError:
            return self.decompress_values_pytorch(compressed_v)

        indices = compressed_v["indices"]
        norms = compressed_v["vec_norms"]
        seq_v = indices.shape[0]
        output = torch.empty(seq_v, self.head_dim, dtype=torch.float16, device=indices.device)

        grid = (self._cdiv(seq_v, BLOCK_S), 1, 1)
        centroids = self.val_codebook.centroids.tolist()
        stream = torch.cuda.current_stream()

        if self.total_bits == 3:
            ct.launch(stream, grid, turboquant_decompress_3bit, ( indices, norms, self.Pi.half(), output, *centroids, seq_v, ))
        elif self.total_bits == 2:
            ct.launch(stream, grid, turboquant_decompress_2bit, ( indices, norms, self.Pi.half(), output, *centroids, seq_v, ))
        else:
            return self.decompress_values_pytorch(compressed_v)

        return output

    @torch.no_grad()
    def launch_attention_scores( self, Q: torch.Tensor, compressed_k: dict, use_swizzle: bool = False ) -> torch.Tensor:
        try:
            from .attention import turboquant_attention_scores
            import cuda.tile as ct
        except ImportError:
            return self.attention_scores_pytorch(Q, compressed_k)

        seq_q = Q.shape[0]
        seq_k = compressed_k["k_mse"].shape[0]

        Q_proj = (Q.float() @ self.S.T.float()).half()
        output = torch.empty(seq_q, seq_k, dtype=torch.float32, device=Q.device)

        grid = (self._cdiv(seq_q, BLOCK_Q), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, turboquant_attention_scores, ( Q, compressed_k["k_mse"], compressed_k["qjl_signs"], compressed_k["residual_norms"], Q_proj, output, self.scale, self.correction_scale, seq_k, use_swizzle, ))
        return output

    @torch.no_grad()
    def launch_fused_attention( self, Q: torch.Tensor, compressed_k: dict, compressed_v: dict, use_swizzle: bool = False, ) -> torch.Tensor:
        try:
            from .attention import ( turboquant_fused_attention, turboquant_fused_attention_vfused_3bit, turboquant_fused_attention_vfused_2bit, )
            import cuda.tile as ct
        except ImportError:
            return self.fused_attention_pytorch(Q, compressed_k, compressed_v)

        seq_q = Q.shape[0]
        seq_k = compressed_k["k_mse"].shape[0]

        Q_proj = (Q.float() @ self.S.T.float()).half()
        output = torch.empty(seq_q, self.head_dim, dtype=torch.float32, device=Q.device)
        grid = (self._cdiv(seq_q, BLOCK_Q), 1, 1)
        stream = torch.cuda.current_stream()

        val_centroids = self.val_codebook.centroids.tolist()

        if self.total_bits == 3:
            ct.launch(stream, grid, turboquant_fused_attention_vfused_3bit, ( Q, compressed_k["k_mse"], compressed_k["qjl_signs"], compressed_k["residual_norms"], Q_proj, compressed_v["indices"], compressed_v["vec_norms"], self.Pi.half(), output, self.scale, self.correction_scale, seq_k, *val_centroids, use_swizzle, ))
        elif self.total_bits == 2:
            ct.launch(stream, grid, turboquant_fused_attention_vfused_2bit, ( Q, compressed_k["k_mse"], compressed_k["qjl_signs"], compressed_k["residual_norms"], Q_proj, compressed_v["indices"], compressed_v["vec_norms"], self.Pi.half(), output, self.scale, self.correction_scale, seq_k, *val_centroids, use_swizzle, ))
        else:
            V_recon = self.launch_decompress_values(compressed_v)
            ct.launch(stream, grid, turboquant_fused_attention, ( Q, compressed_k["k_mse"], compressed_k["qjl_signs"], compressed_k["residual_norms"], Q_proj, V_recon, output, self.scale, self.correction_scale, seq_k, use_swizzle, ))

        return output.half()

    def compressed_size_bytes(self, seq_len: int) -> dict:
        d = self.head_dim

        key_mse_bits = seq_len * d * self.mse_bits
        key_qjl_bits = seq_len * d * 1
        key_norms_bits = seq_len * 16 * 2  # vec_norm + residual_norm
        key_total = (key_mse_bits + key_qjl_bits + key_norms_bits) / 8

        val_mse_bits = seq_len * d * self.total_bits
        val_norms_bits = seq_len * 16
        val_total = (val_mse_bits + val_norms_bits) / 8

        fp16_total = seq_len * d * 2 * 2  # K + V in FP16

        return {
            "key_bytes": key_total,
            "val_bytes": val_total,
            "total_bytes": key_total + val_total,
            "fp16_bytes": fp16_total,
            "compression_ratio": fp16_total / (key_total + val_total),
        }
