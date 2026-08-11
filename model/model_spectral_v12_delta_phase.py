"""
model_spectral_v12_delta_phase.py
==================================
TinyThinker Architecture V12 (Parallel Chunkwise Delta-Phase + V328 Spectral Lerp FFN):
  1. Matrix-Parallel Chunkwise Delta-Phase Memory in O(N) using Parallel Householder Inversion (v300/v305).
  2. Learnable Substrate Lerp FFN (FWHT + DCT-II + DWT Haar Wavelet).
  3. Short Causal Conv1D (k=4) + Factorized Embeddings/Weight Tying.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SpectralArgsV12:
    dim: int = 1024
    emb_dim: int = 256        # Factorized embedding dimension
    n_layers: int = 8
    n_heads: int = 8          # Number of phase memory heads
    vocab_size: int = 32768
    max_seq_len: int = 1024
    chunk_size: int = 64      # Parallel matrix chunk size (C=64)
    conv_kernel_size: int = 4
    num_banks: int = 4        # Multi-frequency banks per spectral FFN
    spherical_head: bool = False
    weight_tying: bool = True

def norm_sphere(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class SphericalHead(nn.Module):
    def __init__(self, in_features, out_features, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return F.linear(x_norm, w_norm) * self.tau

class ShortCausalConv1D(nn.Module):
    """Depthwise 1D Causal Convolution (kernel_size=4) for local token binding"""
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :L].transpose(1, 2)
        return x + self.act(conv_out)

# --- Constructores de Matrices Espectrales Ortogonales ---

def create_hadamard_matrix(n: int) -> torch.Tensor:
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)

def create_dct2_matrix(n: int) -> torch.Tensor:
    C = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = 1.0 / math.sqrt(n)
            else:
                C[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2.0 * n))
    return C

def create_haar_matrix(n: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    H_sub = create_haar_matrix(n // 2)
    low = torch.cat([H_sub, H_sub], dim=1) / math.sqrt(2)
    high = torch.zeros((n // 2, n), dtype=torch.float32)
    for i in range(n // 2):
        high[i, 2 * i] = 1.0 / math.sqrt(2)
        high[i, 2 * i + 1] = -1.0 / math.sqrt(2)
    return torch.cat([low, high], dim=0)

# --- FFN Espectral Lerp Router (V328 Breakthrough) ---

class LearnableSubstrateLerpFFN(nn.Module):
    """FFN con router Softmax Lerp aprendible entre FWHT, DCT-II y DWT Haar (Vectorizado)"""
    def __init__(self, d_model: int, num_banks: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        self.substrate_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        
        self.register_buffer('mat_fwht', create_hadamard_matrix(d_model))
        self.register_buffer('mat_dct', create_dct2_matrix(d_model))
        self.register_buffer('mat_haar', create_haar_matrix(d_model))
        
        self.phi1_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_dct = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_dct = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_haar = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_haar = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x):
        weights = F.softmax(self.substrate_logits, dim=0)
        
        # 1. Rama FWHT (Vectorizada con broadcasting)
        h_fwht = F.linear(x, self.mat_fwht).unsqueeze(-2)
        outs_fwht = (torch.cos(h_fwht + self.phi1_fwht) * self.w1_fwht + 
                     torch.sin(h_fwht + self.phi2_fwht) * self.w2_fwht).flatten(-2)
        out_fwht = F.linear(self.combine(outs_fwht), self.mat_fwht.t())
        
        # 2. Rama DCT-II (Vectorizada con broadcasting)
        h_dct = F.linear(x, self.mat_dct).unsqueeze(-2)
        outs_dct = (torch.cos(h_dct + self.phi1_dct) * self.w1_dct + 
                    torch.sin(h_dct + self.phi2_dct) * self.w2_dct).flatten(-2)
        out_dct = F.linear(self.combine(outs_dct), self.mat_dct.t())
        
        # 3. Rama DWT Haar (Vectorizada con broadcasting)
        h_haar = F.linear(x, self.mat_haar).unsqueeze(-2)
        outs_haar = (torch.cos(h_haar + self.phi1_haar) * self.w1_haar + 
                     torch.sin(h_haar + self.phi2_haar) * self.w2_haar).flatten(-2)
        out_haar = F.linear(self.combine(outs_haar), self.mat_haar.t())
        
        # Combinación Lerp Convexa
        return weights[0] * out_fwht + weights[1] * out_dct + weights[2] * out_haar

    def get_substrate_probabilities(self):
        probs = F.softmax(self.substrate_logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

class DeltaPhaseHolographicBlockV12(nn.Module):
    """
    Parallel Chunkwise Complex Delta-Phase Memory Layer (v300/v305).
    Computes intra-chunk transitions and outputs via parallel GPU matmuls.
    """
    def __init__(self, d_model, n_heads=8, conv_kernel_size=4, chunk_size=64, num_banks=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.inv_dk = 1.0 / float(self.d_k)
        self.chunk_size = chunk_size
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_retrieved = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_beta = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=num_banks)

    def forward(self, x, memory_state=None):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        C = self.chunk_size
        inv_dk = self.inv_dk

        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        theta_k = self.w_k(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.w_beta(conv_x)).transpose(1, 2)
        
        theta_k_f = theta_k.float()
        theta_q_f = theta_q.float()
        K = torch.complex(torch.cos(theta_k_f), torch.sin(theta_k_f))
        Q = torch.complex(torch.cos(theta_q_f), torch.sin(theta_q_f))

        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)

        # 1. Matmuls paralelos intra-chunk para Gram y Matriz de Transición Triangular T_mat
        Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))

        # 2. Inter-chunk scan (SOLO num_chunks iteraciones en lugar de L iteraciones token a token)
        if memory_state is None:
            M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        else:
            M_state = memory_state

        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old = torch.matmul(M_state, torch.conj(kc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, torch.conj(qc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1, -2)).real) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.to(torch.complex64).transpose(-1, -2), kc)

        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, D)
        retrieved_norm = self.norm_retrieved(retrieved)

        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M_state

class SpectralThinkerV12(nn.Module):
    def __init__(self, args: SpectralArgsV12):
        super().__init__()
        self.args = args
        
        if args.emb_dim and args.emb_dim > 0:
            self.embed = nn.Embedding(args.vocab_size, args.emb_dim)
            self.embed_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
            self.use_factorized = True
        else:
            self.embed = nn.Embedding(args.vocab_size, args.dim)
            self.embed_proj = None
            self.use_factorized = False
            
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlockV12(
                d_model=args.dim,
                n_heads=args.n_heads,
                conv_kernel_size=args.conv_kernel_size,
                chunk_size=args.chunk_size,
                num_banks=args.num_banks
            ) for _ in range(args.n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(args.dim)
        
        if args.spherical_head:
            if self.use_factorized:
                self.head_proj = nn.Linear(args.dim, args.emb_dim, bias=False)
                self.head = SphericalHead(args.emb_dim, args.vocab_size, init_tau=10.0)
            else:
                self.head_proj = None
                self.head = SphericalHead(args.dim, args.vocab_size, init_tau=10.0)
        else:
            if self.use_factorized:
                self.head_proj = nn.Linear(args.dim, args.emb_dim, bias=False)
                self.head = nn.Linear(args.emb_dim, args.vocab_size, bias=False)
            else:
                self.head_proj = None
                self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
                
        if args.weight_tying:
            self.head.weight = self.embed.weight
            
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_full, states=None):
        if self.use_factorized:
            e = self.embed(x_full)
            h = self.embed_proj(e)
        else:
            h = self.embed(x_full)
            
        if states is None:
            states = [None] * len(self.blocks)
            
        new_states = []
        for i, block in enumerate(self.blocks):
            h, next_state = block(h, states[i])
            new_states.append(next_state)
            
        h_norm = self.norm_f(h)
        if self.head_proj is not None:
            h_out = self.head_proj(h_norm)
        else:
            h_out = h_norm
            
        logits = self.head(h_out)
        return logits

    def print_substrate_report(self):
        """Imprime un reporte transparente de sustratos espectrales sintonizados por capa"""
        print("\n" + "="*85)
        print("REPORTE TRANSPARENTE DE SUSTRATOS ESPECTRALES ELEGIDOS (V12 LERP ROUTER)")
        print("="*85)
        print(f"{'Capa Residual':<15} | {'% FWHT (Binario)':<20} | {'% DCT-II (Cosenos)':<20} | {'% DWT Haar (Ondículas)':<22}")
        print("-" * 85)
        for idx, block in enumerate(self.blocks):
            p_fwht, p_dct, p_haar = block.ffn.get_substrate_probabilities()
            print(f"Capa {idx+1:<10} | {p_fwht*100:<20.2f}% | {p_dct*100:<20.2f}% | {p_haar*100:<22.2f}%")
        print("="*85)
