"""
test_mini_v12.py
================
Ultra-lightweight mini test for V12 layer memory footprint.
Uses tiny dimensions (B=1, L=32, D=64, n_layers=2) to verify RAM safety.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_spectral_v12_delta_phase import SpectralArgsV12, SpectralThinkerV12

def test_mini():
    print("=== STARTING MINI V12 SAFETY TEST ===")
    args = SpectralArgsV12(
        dim=64,
        emb_dim=32,
        n_layers=2,
        n_heads=2,
        vocab_size=1000,
        max_seq_len=32,
        chunk_size=16,
        conv_kernel_size=4,
        weight_tying=True
    )
    
    model = SpectralThinkerV12(args)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mini Model Params: {params:,}")
    
    # Tiny batch: B=1, L=32
    X = torch.randint(0, args.vocab_size, (1, 32))
    Y = torch.randint(0, args.vocab_size, (1, 32))
    
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    print(f"Mini Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass finished safely!")
    print("=== MINI V12 TEST PASSED CLEANLY ===")

if __name__ == "__main__":
    test_mini()
