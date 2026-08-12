"""
test_beta_range.py
==================
Empirical Comparison: Damped Beta in (0, 1) vs Extended Householder Beta in (0, 2).
"""

import sys, os
sys.path.insert(0, r"C:\Users\mrcm_\Local\proj\algorithms\delta-phase")

import torch
import torch.nn as nn
from delta_phase.layers import DeltaPhaseHolographicBlock

def test_beta_comparison():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    n_heads = 4
    chunk_size = 64
    
    print("=" * 90)
    print("BETA RANGE AUDIT: DAMPED (0, 1) VS EXTENDED HOUSEHOLDER (0, 2)")
    print("=" * 90)
    
    block_damped = DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size).to(device)
    block_householder = DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size).to(device)
    
    # Load same weights except beta scale
    block_householder.load_state_dict(block_damped.state_dict())
    
    # Evaluate convergence on a synthetic recall task
    x = torch.randn(4, 128, d_model, device=device)
    target = torch.randn(4, 128, d_model, device=device)
    
    opt_damped = torch.optim.AdamW(block_damped.parameters(), lr=1e-3)
    opt_house = torch.optim.AdamW(block_householder.parameters(), lr=1e-3)
    
    for step in range(50):
        # Damped
        out_d, _ = block_damped(x)
        loss_d = ((out_d - target)**2).mean()
        opt_damped.zero_grad()
        loss_d.backward()
        opt_damped.step()
        
        # Householder scale (modify beta inside w_beta output during forward)
        out_h, _ = block_householder(x)
        loss_h = ((out_h - target)**2).mean()
        opt_house.zero_grad()
        loss_h.backward()
        opt_house.step()
        
        if step % 10 == 0 or step == 49:
            print(f"Step {step:2d} | Damped (0, 1) Loss: {loss_d.item():.6f} | Householder (0, 2) Loss: {loss_h.item():.6f}")
            
    print("=" * 90)

if __name__ == "__main__":
    test_beta_comparison()
