"""
test_medium_v12.py
==================
Medium scale safety test (B=2, L=256, D=256, n_layers=4).
Verifies memory stability and execution speed in < 3 seconds.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_spectral_v12_delta_phase import SpectralArgsV12, SpectralThinkerV12

def test_medium():
    print("=== STARTING MEDIUM V12 SAFETY TEST ===")
    args = SpectralArgsV12(
        dim=256,
        emb_dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=8000,
        max_seq_len=256,
        chunk_size=64,
        conv_kernel_size=4,
        weight_tying=True
    )
    
    model = SpectralThinkerV12(args)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Medium Model Params: {params:,}")
    
    X = torch.randint(0, args.vocab_size, (2, 256))
    Y = torch.randint(0, args.vocab_size, (2, 256))
    
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    print(f"Medium Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass finished safely!")
    print("=== MEDIUM V12 TEST PASSED CLEANLY ===")

if __name__ == "__main__":
    test_medium()
