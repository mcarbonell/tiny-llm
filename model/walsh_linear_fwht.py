import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fwht_op import fwht_native


def _next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fwht_torch(x):
    """
    FWHT ortogonal (1/sqrt(n)) sobre el último eje usando el kernel nativo si
    está disponible, con fallback a una implementación vectorizada en PyTorch.
    x: (..., n) con n potencia de dos. Devuelve (..., n) normalizado 1/sqrt(n).
    """
    # El kernel FWHT muta el tensor in-place; clonamos para preservar el grafo
    # y no romper tensores que el llamador reuse (p.ej. el embedding original).
    lib = fwht_native(x.clone())
    if lib is not None:
        return lib
    # Fallback vectorizado (más lento, pero correcto y diferenciable)
    n = x.shape[-1]
    out = x.clone()
    h = 1
    while h < n:
        out = out.reshape(-1, n // (2 * h), 2, h).transpose(2, 3).reshape(-1, 2 * h)
        a = out[..., :h]
        b = out[..., h:]
        out = torch.cat([a + b, a - b], dim=-1)
        h <<= 1
    return out / (n ** 0.5)


def _raw_fwht(x_padded):
    """FWHT SIN normalización global (matriz Hadamard ±1 pura)."""
    n_pad = x_padded.shape[-1]
    return fwht_torch(x_padded) * (n_pad ** 0.5)


class WalshLinearFWHT(nn.Module):
    """
    WalshLinear sintetizado on-the-fly SIN matriz densa.

    W_synthesized = H_out[:, :k] @ core @ H_in[:k, :]   (d_out x d_in)

    Forward eficiente O(d*log d + k^2):
        y = ((x @ A.T) @ core.T) @ B.T
    donde A = H_in[:k] (k x d_in), B = H_out[:, :k] (d_out x k).
    x @ A.T  == FWHT(x) truncado a las k primeras coords (H_in es la matriz completa).
    (z @ B.T) == colocar z en las k coords de un vector d_out y aplicar FWHT inverso.

    Los buffers H_in / H_out NO se materializan (ahorra O(d^2) memoria); solo se
    usa su estructura via el kernel FWHT.
    """

    def __init__(self, in_features, out_features, k, normalized=True):
        super().__init__()
        self.k = min(k, in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.normalized = normalized

        self.core = nn.Parameter(torch.randn(self.k, self.k) / (self.k ** 0.5))
        if normalized:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = None

        # Padding a potencia de dos para el kernel FWHT
        self.n_in = _next_power_of_two(in_features)
        self.n_out = _next_power_of_two(out_features)

    def forward(self, x):
        # x: (..., in_features)
        # Síntesis: W = H_out[:, :k] @ core @ H_in[:k, :]   (d_out x d_in)
        # H_in, H_out son matrices Walsh ortogonales normalizadas 1/sqrt(d).
        # x @ A.T con A = H_in[:k]  == (1/sqrt(d_in)) * rawFWHT(x)[:k]
        x_pad = F.pad(x, (0, self.n_in - self.in_features))
        X = _raw_fwht(x_pad)[..., :self.k] / (self.in_features ** 0.5)   # (..., k)

        # 2) @ core.T
        Z = X @ self.core.t()                        # (..., k)

        # 3) @ B.T con B = H_out[:, :k]
        #    (z @ B.T)[j] = (1/sqrt(d_out)) * rawFWHT(z_ext)[j], z_ext = z en k primeras coords
        z_ext = F.pad(Z, (0, self.n_out - self.k))   # (..., n_out)
        Y = _raw_fwht(z_ext)[..., :self.out_features] / (self.out_features ** 0.5)  # (..., d_out)

        if self.normalized:
            return Y * self.scale
        return Y
