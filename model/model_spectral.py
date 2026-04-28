"""
model_spectral.py — SpectralThinker v2: DCT Attention + Walsh FFN

OPTIMIZACIONES vs v1:
  - Base truncada: almacena solo k filas (k×N) en lugar de la matriz completa (N×N)
  - Caché de dispositivo: 0 transferencias CPU->GPU después del primer forward
  - Sin torch.zeros: síntesis via 2 matmuls directas (D_out_k.T @ core @ D_in_k)
  - Resultado: ~4-5x speedup en DirectML

Arquitectura:
  - Atención Q,K,V,O: DCTLinear (cosenos ortogonales)
  - FFN w1,w2,w3:     WalshLinear (Walsh-Hadamard ±1)
  - RoPE + GQA + RMSNorm + SwiGLU del modelo base
  - Compatible con train.py via arch='spectral'
  - Compatible con DirectML (Radeon 780M)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. SpectralArgs
# =============================================================================

@dataclass
class SpectralArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    # k_dim_attn=64, dim=256 => 4x compresion en atencion
    k_dim_attn: int = 64
    # k_dim_ffn=64, dim=256 => 4x compresion en FFN entrada/salida
    k_dim_ffn: int = 64
    # k_hidden_ffn=128, hidden~512 => 4x compresion en FFN dim oculta
    k_hidden_ffn: int = 128


# =============================================================================
# 2. Generadores de bases (cache global — una sola generacion por dimension)
# =============================================================================

_DCT_CACHE: dict = {}
_WALSH_CACHE: dict = {}


def _get_dct_matrix(N: int) -> torch.Tensor:
    """Genera (o recupera del cache) la base DCT-II ortogonal N×N en CPU."""
    if N not in _DCT_CACHE:
        mat = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if i == 0:
                    mat[i, j] = math.sqrt(1.0 / N)
                else:
                    mat[i, j] = math.sqrt(2.0 / N) * math.cos(
                        math.pi * i * (2 * j + 1) / (2 * N)
                    )
        _DCT_CACHE[N] = mat
    return _DCT_CACHE[N]


def _get_walsh_matrix(N: int) -> torch.Tensor:
    """Genera (o recupera del cache) la base Walsh-Hadamard N×N en CPU."""
    assert N > 0 and (N & (N - 1)) == 0, f"Walsh requiere potencia de 2, got {N}"
    if N not in _WALSH_CACHE:
        H = torch.tensor([[1.0]])
        while H.shape[0] < N:
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        _WALSH_CACHE[N] = H / math.sqrt(N)
    return _WALSH_CACHE[N]


# =============================================================================
# 3. Capas espectrales optimizadas
# =============================================================================

class DCTLinear(nn.Module):
    """
    Proyeccion lineal sintetizada desde un nucleo DCT de baja frecuencia.

    Matematica exacta:
        W = D_out_k.T @ core @ D_in_k
    donde D_in_k son las primeras k_in filas de la base DCT de entrada,
    y D_out_k las primeras k_out filas de la base DCT de salida.

    OPTIMIZACIONES:
    - Almacena solo D_k (k×N) en lugar de D completa (N×N): 4x menos memoria
    - Cachea D_k en el dispositivo destino tras el primer forward: 0 transfers
    - Sin torch.zeros: sintesis via 2 matmuls directas
    """

    def __init__(self, in_features: int, out_features: int,
                 k_in: int, k_out: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k_in  = min(k_in,  in_features)
        self.k_out = min(k_out, out_features)

        # Solo las k primeras filas (vectores de baja frecuencia)
        D_in  = _get_dct_matrix(in_features)
        D_out = _get_dct_matrix(out_features)
        # Guardamos en CPU como atributos normales (no buffers de PyTorch)
        # para evitar que se serialicen en el state_dict
        self._D_in_k_cpu  = D_in[:self.k_in, :].clone()    # (k_in,  in_features)
        self._D_out_k_cpu = D_out[:self.k_out, :].clone()   # (k_out, out_features)
        self._D_in_k_dev  = None  # cache en dispositivo destino
        self._D_out_k_dev = None
        self._cached_dev  = None

        self.core = nn.Parameter(
            torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _get_bases(self, device):
        """Devuelve las bases en el dispositivo correcto, cacheando tras la primera llamada."""
        if self._cached_dev != device:
            self._D_in_k_dev  = self._D_in_k_cpu.to(device)
            self._D_out_k_dev = self._D_out_k_cpu.to(device)
            self._cached_dev  = device
        return self._D_in_k_dev, self._D_out_k_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D_in_k, D_out_k = self._get_bases(x.device)
        # W = D_out_k.T @ core @ D_in_k
        # (k_out, in_features) → (out_features, in_features)
        W = D_out_k.t() @ self.core @ D_in_k
        return F.linear(x, W, self.bias_param)


class WalshLinear(nn.Module):
    """
    Proyeccion lineal sintetizada desde un nucleo Walsh-Hadamard de baja frecuencia.

    Matematica exacta:
        W = H_out_k.T @ core @ H_in_k
    donde H_in_k y H_out_k son las primeras k filas de las bases Walsh.
    in_features y out_features deben ser potencias de 2.
    """

    def __init__(self, in_features: int, out_features: int,
                 k_in: int, k_out: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k_in  = min(k_in,  in_features)
        self.k_out = min(k_out, out_features)

        H_in  = _get_walsh_matrix(in_features)
        H_out = _get_walsh_matrix(out_features)
        self._H_in_k_cpu  = H_in[:self.k_in, :].clone()    # (k_in,  in_features)
        self._H_out_k_cpu = H_out[:self.k_out, :].clone()   # (k_out, out_features)
        self._H_in_k_dev  = None
        self._H_out_k_dev = None
        self._cached_dev  = None

        self.core = nn.Parameter(
            torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _get_bases(self, device):
        if self._cached_dev != device:
            self._H_in_k_dev  = self._H_in_k_cpu.to(device)
            self._H_out_k_dev = self._H_out_k_cpu.to(device)
            self._cached_dev  = device
        return self._H_in_k_dev, self._H_out_k_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in_k, H_out_k = self._get_bases(x.device)
        W = H_out_k.t() @ self.core @ H_in_k
        return F.linear(x, W, self.bias_param)


# =============================================================================
# 4. Utilidades estandar (RMSNorm, RoPE)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Workaround DirectML: aten::pow.Tensor_Scalar no disponible bajo autocast
        if x.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                x_f = x.float()
                var = (x_f * x_f).mean(-1, keepdim=True)
                return (x_f * torch.rsqrt(var + self.eps)) * self.weight
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return freqs


def _reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1]), f"{freqs.shape} vs {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs = _reshape_for_broadcast(freqs, xq_[..., 0])
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    xq_0, xq_1 = xq_.unbind(-1)
    xk_0, xk_1 = xk_.unbind(-1)
    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos
    xk_out_0 = xk_0 * cos - xk_1 * sin
    xk_out_1 = xk_0 * sin + xk_1 * cos
    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# 5. SpectralAttention (DCTLinear + RoPE + GQA)
# =============================================================================

class SpectralAttention(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.n_heads    = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_rep    = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        k = args.k_dim_attn

        self.wq = DCTLinear(args.dim, args.n_heads    * self.head_dim, k_in=k, k_out=k)
        self.wk = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, k_in=k, k_out=k)
        self.wv = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, k_in=k, k_out=k)
        self.wo = DCTLinear(args.n_heads * self.head_dim, args.dim,    k_in=k, k_out=k)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                past_key_value=None) -> Tuple[torch.Tensor, tuple]:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads,    self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs=freqs_cis)

        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(
                bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim
            ).flatten(2, 3)
            xv = xv[:, :, :, None, :].expand(
                bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim
            ).flatten(2, 3)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=2)
            xv = torch.cat([past_v, xv], dim=2)

        if past_key_value is None and seqlen > 1:
            out = F.scaled_dot_product_attention(xq, xk, xv,
                                                  attn_mask=None, dropout_p=0.0,
                                                  is_causal=True)
        else:
            out = F.scaled_dot_product_attention(xq, xk, xv,
                                                  attn_mask=mask, dropout_p=0.0,
                                                  is_causal=False)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out), (xk, xv)


# =============================================================================
# 6. SpectralFeedForward (WalshLinear + SwiGLU)
# =============================================================================

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class SpectralFeedForward(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        # Misma formula que model.py (SwiGLU)
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )
        # Walsh exige potencias de 2
        dim_w    = _next_pow2(args.dim)
        hidden_w = _next_pow2(hidden_dim)
        k_d, k_h = args.k_dim_ffn, args.k_hidden_ffn

        self.w1 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)
        self.w2 = WalshLinear(hidden_w, dim_w,  k_in=k_h, k_out=k_d)
        self.w3 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)

        self.in_dim   = args.dim
        self.out_dim  = args.dim
        self.dim_w    = dim_w
        self.hidden_w = hidden_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding si dim no es exactamente potencia de 2
        if self.in_dim != self.dim_w:
            x = F.pad(x, (0, self.dim_w - self.in_dim))
        h = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.out_dim != self.dim_w:
            h = h[..., :self.out_dim]
        return h


# =============================================================================
# 7. SpectralTransformerBlock
# =============================================================================

class SpectralTransformerBlock(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.attention      = SpectralAttention(args)
        self.feed_forward   = SpectralFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm       = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def _fwd(self, x, freqs_cis, mask, past_kv):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h)), new_kv

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._fwd, x, freqs_cis, mask, past_key_value, use_reentrant=False
            )
        return self._fwd(x, freqs_cis, mask, past_key_value)


# =============================================================================
# 8. SpectralThinker — modelo completo
# =============================================================================

class SpectralThinker(nn.Module):
    """
    LLM Espectral compatible con TinyThinker train.py:
        logits          = model(tokens)
        logits, loss    = model(tokens, targets)
        logits, new_kvs = model(tokens, use_cache=True)
    """

    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args       = args
        self.vocab_size = args.vocab_size
        self.n_layers   = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [SpectralTransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm   = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying + inicializacion correcta (std = 1/sqrt(dim) evita explosion
        # de logits con vocab grande). Se asigna DESPUES de crear ambos modulos.
        self.output.weight = self.tok_embeddings.weight
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                past_key_values=None,
                use_cache: bool = False):
        _bsz, seqlen = tokens.shape

        # Workaround DirectML: aten::embedding no soporta autocast
        if tokens.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                h = self.tok_embeddings(tokens)
                freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
                mask = _causal_mask(seqlen, tokens.device, h.dtype)
        else:
            h = self.tok_embeddings(tokens)
            freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
            mask = _causal_mask(seqlen, tokens.device, h.dtype)

        new_kvs: List = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, kv = layer(h, freqs_cis, mask, pkv)
            new_kvs.append(kv)

        logits = self.output(self.norm(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        if use_cache:
            return logits, new_kvs
        return logits


def _causal_mask(seqlen: int, device, dtype) -> Optional[torch.Tensor]:
    if seqlen <= 1:
        return None
    mask = torch.zeros(seqlen, seqlen, device=device, dtype=dtype)
    mask.masked_fill_(
        torch.ones(seqlen, seqlen, device=device, dtype=torch.bool)
        .tril().logical_not(), float('-inf')
    )
    return mask.view(1, 1, seqlen, seqlen)


# =============================================================================
# 9. Utilidad: desglose de parametros
# =============================================================================

def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'trainable': trainable, 'trainable_M': round(trainable / 1e6, 2)}
