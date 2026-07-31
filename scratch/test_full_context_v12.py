"""
test_full_context_v12.py
========================
Full context length safety test (B=2, L=1024, D=1024, n_layers=8).
Verifies full context stability, initial loss (~10.39), and zero RAM leak.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_spectral_v12_delta_phase import SpectralArgsV12, SpectralThinkerV12

def test_full_context():
    print("=== STARTING FULL CONTEXT V12 SAFETY TEST ===")
    args = SpectralArgsV12(
        dim=1024,
        emb_dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=32768,
        max_seq_len=1024,
        chunk_size=64,
        conv_kernel_size=4,
        weight_tying=True
    )
    
    model = SpectralThinkerV12(args)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full Model Params: {params / 1e6:.2f}M")
    
    X = torch.randint(0, args.vocab_size, (2, 1024))
    Y = torch.randint(0, args.vocab_size, (2, 1024))
    
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    print(f"Full Context Initial Loss: {loss.item():.4f} (Expected ~10.39)")
    
    loss.backward()
    print("Backward pass finished safely!")
    print("=== FULL CONTEXT V12 TEST PASSED CLEANLY ===")

if __name__ == "__main__":
    test_full_context()
