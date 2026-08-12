"""
test_gradcheck_fp64.py
======================
FP64 Double Precision Gradcheck and Global L2 Relative Gradient Audit for DeltaPhase.
Performs exact PyTorch gradcheck and per-tensor gradient breakdown.
"""

import sys, os
sys.path.insert(0, r"C:\Users\mrcm_\Local\proj\algorithms\delta-phase")

import torch
import torch.nn as nn
from delta_phase.layers import DeltaPhaseHolographicBlock

def test_fp64_gradcheck():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 32
    n_heads = 2
    chunk_size = 16
    
    print("=" * 105)
    print("FP64 DOUBLE PRECISION GRADCHECK & GLOBAL L2 RELATIVE GRADIENT AUDIT")
    print("=" * 105)
    
    for dtype, name in [(torch.float64, "FP64 Double"), (torch.float32, "FP32 Single")]:
        block = DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size).to(device, dtype=dtype)
        block.eval()
        
        L = 64
        x = torch.randn(2, L, d_model, device=device, dtype=dtype, requires_grad=True)
        init_M = torch.randn(2, n_heads, d_model // n_heads, d_model // n_heads, dtype=torch.complex128 if dtype==torch.float64 else torch.complex64, device=device)
        
        # 1. Parallel Chunkwise Forward & Backward
        out_chunk, state_chunk = block(x, memory_state=init_M.clone())
        loss_chunk = out_chunk.sum()
        loss_chunk.backward()
        grad_chunk = x.grad.clone()
        x.grad.zero_()
        
        # 2. Sequential Step-by-Step Scan Forward & Backward
        out_seq_list = []
        seq_state = (None, init_M.clone())
        for t in range(L):
            x_t = x[:, t:t+1, :]
            out_t, seq_state = block.step(x_t, state=seq_state)
            out_seq_list.append(out_t)
        out_seq = torch.cat(out_seq_list, dim=1)
        loss_seq = out_seq.sum()
        loss_seq.backward()
        grad_seq = x.grad.clone()
        x.grad.zero_()
        
        # Global L2 Norm Relative Error calculation
        diff_norm_l2 = (grad_chunk - grad_seq).norm(p=2).item()
        seq_norm_l2 = grad_seq.norm(p=2).item()
        rel_error_l2 = diff_norm_l2 / max(seq_norm_l2, 1e-12)
        
        max_abs_err = (grad_chunk - grad_seq).abs().max().item()
        
        print(f"[{name}] L2 Grad Norm: Chunk={grad_chunk.norm().item():.6f} | Seq={seq_norm_l2:.6f}")
        print(f"[{name}] Max Abs Grad Diff: {max_abs_err:.6e} | Global L2 Relative Grad Error: {rel_error_l2:.6e}")
        print("-" * 105)

    # PyTorch Gradcheck on Chunkwise Block in FP64
    print("Running PyTorch autograd.gradcheck in FP64 Double Precision...")
    block_64 = DeltaPhaseHolographicBlock(d_model=16, n_heads=2, chunk_size=8).to(device, dtype=torch.float64)
    x_check = torch.randn(1, 16, 16, device=device, dtype=torch.float64, requires_grad=True)
    
    def func(inputs):
        out, _ = block_64(inputs)
        return out
        
    try:
        passed = torch.autograd.gradcheck(func, (x_check,), eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f"[OK] PyTorch autograd.gradcheck PASSED in FP64 Double Precision: {passed}")
    except Exception as e:
        print(f"[FAIL] Gradcheck failed: {e}")
        
    print("=" * 105)

if __name__ == "__main__":
    test_fp64_gradcheck()
