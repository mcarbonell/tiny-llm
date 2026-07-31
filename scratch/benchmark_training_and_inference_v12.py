"""
benchmark_training_and_inference_v12.py
=========================================
Comprehensive Benchmark of V12 Delta-Phase Memory:
  1. Training Throughput (Forward + Backward Pass, B=8, L=1024)
  2. TorchScript JIT Optimized Streaming Inference (B=1, O(1) RAM)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. TorchScript JIT Optimized Streaming Step ─────────────────────────
@torch.jit.script
def delta_phase_step_jit(
    M: torch.Tensor,
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    beta: torch.Tensor,
    inv_dk: float
) -> torch.Tensor:
    """
    M: [B, H, d_k, d_k] complex
    K, Q: [B, H, d_k] complex
    V: [B, H, d_k] real
    beta: [B, H, 1, 1] real
    """
    K_conj = torch.conj(K)
    Q_conj = torch.conj(Q)
    
    # 1. Readout v_old = Re( M * conj(K) ) * inv_dk
    v_old = torch.matmul(M, K_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
    
    # 2. Residual error
    err = V - v_old
    
    # 3. Outer product update: M = M + beta * (err (x) K)
    update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), K.unsqueeze(-2))
    M = M + beta * update
    
    # 4. Readout R = Re( M * conj(Q) ) * inv_dk
    R_out = torch.matmul(M, Q_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
    return R_out

# ── 2. Full V12 Layer Component for Training ────────────────────────────
class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
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

class V12DeltaPhaseLayer(nn.Module):
    def __init__(self, d_model=1024, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.inv_dk = 1.0 / float(self.d_k)
        
        self.norm = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        normed = self.norm(x)
        conv_x = self.conv(normed)
        B, L, D = conv_x.shape
        
        theta_k = self.w_k(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.w_q(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.w_v(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.w_beta(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_list = []
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            k_conj = torch.conj(k_t)
            q_conj = torch.conj(q_t)
            
            v_old = torch.matmul(M, k_conj.unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = v_t - v_old
            
            update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), k_t.unsqueeze(-2))
            M = M + beta_t * update
            
            ret = torch.matmul(M, q_conj.unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(ret)
            
        retrieved = torch.stack(out_list, dim=1).view(B, L, D)
        return x + self.out_proj(retrieved)

def run_benchmarks():
    device = torch.device("cpu")
    print("==================================================")
    print(" 1. TorchScript JIT Streaming Inference Benchmark")
    print("==================================================")
    
    B, H, d_k = 1, 8, 32
    inv_dk = 1.0 / float(d_k)
    seq_len = 5000
    
    theta_k = torch.randn(seq_len, B, H, d_k, device=device)
    theta_q = torch.randn(seq_len, B, H, d_k, device=device)
    V_seq   = torch.randn(seq_len, B, H, d_k, device=device)
    beta_seq= torch.sigmoid(torch.randn(seq_len, B, H, device=device)).view(seq_len, B, H, 1, 1)
    
    K_seq = torch.polar(torch.ones_like(theta_k), theta_k)
    Q_seq = torch.polar(torch.ones_like(theta_q), theta_q)
    
    M_jit = torch.zeros(B, H, d_k, d_k, dtype=torch.complex64, device=device)
    
    # Warmup JIT
    _ = delta_phase_step_jit(M_jit.clone(), K_seq[0], Q_seq[0], V_seq[0], beta_seq[0], inv_dk)
    
    t0 = time.time()
    for t in range(seq_len):
        _ = delta_phase_step_jit(M_jit, K_seq[t], Q_seq[t], V_seq[t], beta_seq[t], inv_dk)
    t_jit = time.time() - t0
    tok_sec_jit = seq_len / t_jit
    print(f"  TorchScript JIT Streaming Speed: {tok_sec_jit:8.1f} tokens/sec  ({t_jit:.3f} s for {seq_len} tokens)")
    
    print("\n==================================================")
    print(" 2. Training Forward + Backward Pass Benchmark")
    print("==================================================")
    
    layer = V12DeltaPhaseLayer(d_model=512, n_heads=8).to(device)
    x = torch.randn(4, 256, 512, requires_grad=True, device=device)
    
    # Warmup
    out = layer(x)
    out.sum().backward()
    
    t0 = time.time()
    for _ in range(5):
        layer.zero_grad()
        out = layer(x)
        loss = out.sum()
        loss.backward()
    t_train = (time.time() - t0) / 5.0
    print(f"  Single Layer Training Step (B=4, L=256, D=512): {t_train*1000:.1f} ms per step")

if __name__ == "__main__":
    run_benchmarks()
