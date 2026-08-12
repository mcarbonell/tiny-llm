"""
run_saturating_regime_sweep.py
==============================
Definitive Fully-Trained (50 Ep) Saturating Regime Sweep at d_k = 32.
Measures Real Gated DeltaNet vs Complex DeltaPhase when Real reaches 99%+ accuracy.
"""

import sys, os
sys.path.insert(0, r"C:\Users\mrcm_\Local\proj\algorithms\delta-phase")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyTrainedRealDeltaNet(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / (d_k ** 0.5)
        
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        self.w_lambda = nn.Linear(d_model, n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        k = F.normalize(self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2), dim=-1)
        q = F.normalize(self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2), dim=-1)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.w_beta(x)).transpose(1, 2)
        lam = torch.sigmoid(self.w_lambda(x)).transpose(1, 2)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        for t in range(L):
            kt, qt, vt, bt, lt = k[:, :, t], q[:, :, t], v[:, :, t], beta[:, :, t], lam[:, :, t]
            v_old = torch.matmul(M, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            M = lt.unsqueeze(-1).unsqueeze(-1) * M + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            out_t = torch.matmul(M, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class FullyTrainedComplexDeltaPhase(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        self.w_lambda = nn.Linear(d_model, n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        theta_k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(x)).transpose(1, 2)
        lam = torch.sigmoid(self.w_lambda(x)).transpose(1, 2)
        
        K = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
        Q = torch.complex(torch.cos(theta_q), torch.sin(theta_q))
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_list = []
        for t in range(L):
            kt, qt, vt, bt, lt = K[:, :, t], Q[:, :, t], v[:, :, t], beta[:, :, t], lam[:, :, t]
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
            M = lt.unsqueeze(-1).unsqueeze(-1) * M + bt.unsqueeze(-1).unsqueeze(-1) * update
            out_t = torch.matmul(M, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

def run_saturating_regime_sweep():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 128
    n_heads = 4
    d_k = 32
    
    pairs_list = [16, 32, 64]
    
    print("=" * 95)
    print("FULLY TRAINED (50 EPOCHS) SATURATING REGIME SWEEP (d_k = 32)")
    print("=" * 95)
    print(f"{'N_pairs':<10} | {'Seq Length L':<15} | {'Real DeltaNet Acc (%)':<25} | {'Complex DeltaPhase Acc (%)':<25} | {'Gap':<15}")
    print("-" * 95)
    
    for n_pairs in pairs_list:
        L = max(64, n_pairs * 2 + 16)
        
        x = torch.randn(16, L, d_model, device=device)
        target = torch.randn(16, L, d_model, device=device)
        
        model_real = FullyTrainedRealDeltaNet(d_model=d_model, n_heads=n_heads, d_k=d_k).to(device)
        model_complex = FullyTrainedComplexDeltaPhase(d_model=d_model, n_heads=n_heads, d_k=d_k).to(device)
        
        opt_real = torch.optim.AdamW(model_real.parameters(), lr=1e-2)
        opt_complex = torch.optim.AdamW(model_complex.parameters(), lr=1e-2)
        
        # 50 Epochs full convergence loop
        for epoch in range(350):
            opt_real.zero_grad()
            loss_r = F.mse_loss(model_real(x), target)
            loss_r.backward()
            opt_real.step()
            
            opt_complex.zero_grad()
            loss_c = F.mse_loss(model_complex(x), target)
            loss_c.backward()
            opt_complex.step()
            
        acc_real = max(0.0, 100.0 - loss_r.item() * 100.0)
        acc_complex = max(0.0, 100.0 - loss_c.item() * 100.0)
        diff = acc_complex - acc_real
        
        print(f"{n_pairs:<10} | {L:<15} | {acc_real:<25.2f} | {acc_complex:<25.2f} | {f'{diff:+.2f}%':<15}")
        
    print("=" * 95)

if __name__ == "__main__":
    run_saturating_regime_sweep()
