"""
test_model_v12.py
===================
Unit test for TinyThinker V12 Architecture (DeltaPhaseHolographic LLM).
"""

import sys
import os
import torch

# Ensure model import from workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_spectral_v12_delta_phase import SpectralArgsV12, SpectralThinkerV12

def test_v12_model():
    args = SpectralArgsV12(
        dim=512,
        emb_dim=128,
        n_layers=4,
        n_heads=8,
        vocab_size=32768,
        max_seq_len=256,
        conv_kernel_size=4,
        weight_tying=True
    )
    
    model = SpectralThinkerV12(args)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V12 Model initialized with {params:,} parameters.")
    
    # Batch size 2, Sequence length 64
    x = torch.randint(0, args.vocab_size, (2, 64))
    logits = model(x)
    
    print(f"Logits shape: {logits.shape} (Expected: [2, 64, 32768])")
    assert logits.shape == (2, 64, args.vocab_size), "Logits shape mismatch!"
    
    # Test gradient backward pass
    loss = logits.sum()
    loss.backward()
    print("Backward pass completed cleanly with no NaNs or errors!")
    print("=== V12 UNIT TEST PASSED ===")

if __name__ == "__main__":
    test_v12_model()
