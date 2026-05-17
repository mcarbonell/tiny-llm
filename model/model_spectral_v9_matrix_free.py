import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SpectralArgsV9:
    dim: int = 128
    n_layers: int = 6
    vocab_size: int = 16384
    max_seq_len: int = 1024
    k_walsh: int = 32  # Tamaño del núcleo de Walsh (e.g. 64 o 32)
    # Por defecto en Matrix-Free, las capas usan norm=True en las salidas
    # y la inicialización alfa de nGPT
    alpha_init: float = 0.05

# ══════════════════════════════════════════════════════════════════════
# nGPT UTILS
# ══════════════════════════════════════════════════════════════════════
def norm_sphere(x, eps=1e-8):
    """Proyección a la hiperesfera unitaria por token."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class NormalizedLinear(nn.Linear):
    def forward(self, x):
        w = F.normalize(self.weight, dim=-1)
        return F.linear(x, w, self.bias)

# ══════════════════════════════════════════════════════════════════════
# WALSH/HADAMARD UTILS
# ══════════════════════════════════════════════════════════════════════
def get_walsh_matrix_1d(dim):
    """Genera matriz de Walsh-Hadamard recursivamente (ortogonal, normalizada)."""
    if dim == 1:
        return torch.tensor([[1.]])
    H = get_walsh_matrix_1d(dim // 2)
    return torch.cat([
        torch.cat([H, H], dim=1),
        torch.cat([H, -H], dim=1)
    ], dim=0) / math.sqrt(2)

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k, normalized=True):
        super().__init__()
        # k debe ser <= min(in_features, out_features)
        self.k = min(k, in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.normalized = normalized
        
        # Núcleo denso de baja dimensionalidad (Matrix-Free core)
        self.core = nn.Parameter(torch.randn(self.k, self.k) / math.sqrt(self.k))
        # Escala escalar si se normaliza, para recuperar amplitud
        self.scale = nn.Parameter(torch.ones(1)) if normalized else None
        
        self.register_buffer('H_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('H_out', get_walsh_matrix_1d(out_features))

    def forward(self, x):
        # Síntesis on-the-fly de la matriz W usando el núcleo KxK
        # H_out_k: (out_features, k)
        # H_in_k:  (k, in_features)
        H_out_k = self.H_out[:, :self.k]
        H_in_k = self.H_in[:self.k, :]
        
        # W_synthesized (out_features, in_features)
        W_synthesized = H_out_k @ self.core @ H_in_k
        
        if self.normalized:
            w = F.normalize(W_synthesized, dim=-1)
            return F.linear(x, w) * self.scale
        else:
            return F.linear(x, W_synthesized)

# ══════════════════════════════════════════════════════════════════════
# MIXER & FFN
# ══════════════════════════════════════════════════════════════════════
class CausalComplexFFTMixer(nn.Module):
    """
    True Causal Complex FFT Mixer.
    Transforma la secuencia al dominio espectral, aplica modulación de fase causal,
    y vuelve al dominio del tiempo, proyectando el output mediante WalshLinear.
    """
    def __init__(self, T, D, k_walsh, normalized=True):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        
        # Modulación espectral (Fase como Positional Encoding)
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        
        # Proyección asintótica O(k^2)
        self.out_proj = WalshLinear(D, D, k_walsh, normalized=normalized)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1) # (B, D, T)
        
        # Padding para FFT (evitar wrap-around)
        pad = torch.zeros(B, D, self.pad_T - T, device=x.device, dtype=x.dtype)
        xt_pad = torch.cat([xt, pad], dim=-1)
        X = torch.fft.rfft(xt_pad, dim=-1)

        # Creación y proyección causal del gate
        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal  = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)

        out = torch.fft.irfft(X * gate_causal, n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1) # (B, T, D)
        return self.out_proj(out)

class WalshNarrowFFN(nn.Module):
    """
    Feed-Forward Network O(k^2).
    Elimina la expansión masiva y reemplaza la matriz dxd por un WalshLinear.
    """
    def __init__(self, D, k_walsh, normalized=True):
        super().__init__()
        self.proj = WalshLinear(D, D, k_walsh, normalized=normalized)
            
    def forward(self, x):
        return F.gelu(self.proj(x))

# ══════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class nGPTBlock(nn.Module):
    """
    Bloque de normalización hiper-esférica nGPT.
    Garantiza que la magnitud de los vectores latentes nunca estalle,
    crucial cuando las proyecciones son ortogonales (Hadamard).
    """
    def __init__(self, args: SpectralArgsV9):
        super().__init__()
        # Componentes Matrix-Free
        self.mixer = CausalComplexFFTMixer(args.max_seq_len, args.dim, args.k_walsh, normalized=True)
        self.ffn = WalshNarrowFFN(args.dim, args.k_walsh, normalized=True)
        
        # nGPT Eigen-learning rates
        self.alpha_mixer = nn.Parameter(torch.full((args.dim,), args.alpha_init))
        self.alpha_ffn   = nn.Parameter(torch.full((args.dim,), args.alpha_init))
        
        self.use_checkpoint = False

    def forward(self, x):
        def _inner_forward(x):
            m = norm_sphere(self.mixer(x))
            alpha_m = self.alpha_mixer.abs().unsqueeze(0).unsqueeze(0)
            x = norm_sphere(x + alpha_m * m)
            
            f = norm_sphere(self.ffn(x))
            alpha_f = self.alpha_ffn.abs().unsqueeze(0).unsqueeze(0)
            x = norm_sphere(x + alpha_f * f)
            return x

        if self.use_checkpoint and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            return _inner_forward(x)

class SpectralThinkerV9(nn.Module):
    def __init__(self, args: SpectralArgsV9):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([nGPTBlock(args) for _ in range(args.n_layers)])
        
        # Cabeza final. Para V9 la hacemos estándar (no Walsh), pero normalizada
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        # En nGPT, inyectamos en la hiperesfera directamente tras el embedding
        h = norm_sphere(self.embed(x))
        for layer in self.layers:
            h = layer(h)
            
        # Opcionalmente, proyector final normalizado, pero dejamos capa lineal pura
        # para predecir probabilidades del vocabulario
        return self.head(h)
