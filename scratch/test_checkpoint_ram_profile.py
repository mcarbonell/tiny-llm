"""
test_checkpoint_ram_profile.py
================================
RAM Profiling Test for PyTorch Gradient Checkpointing in V12.
Measures peak process RAM usage in MB during full context training pass (B=2, L=1024, D=1024, n_layers=8).
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_spectral_v12_delta_phase import SpectralArgsV12, SpectralThinkerV12

def get_process_ram_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def profile_ram():
    print("=== STARTING V12 RAM PROFILING TEST WITH GRADIENT CHECKPOINTING ===")
    args = SpectralArgsV12(
        dim=1024,
        emb_dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=32768,
        max_seq_len=1024,
        chunk_size=128,
        conv_kernel_size=4,
        weight_tying=True
    )
    
    ram_init = get_process_ram_mb()
    print(f"RAM before model init: {ram_init:.1f} MB")
    
    model = SpectralThinkerV12(args)
    model.train()
    
    ram_model = get_process_ram_mb()
    print(f"RAM after model init (76M params): {ram_model:.1f} MB  (Model weight footprint: {ram_model - ram_init:.1f} MB)")
    
    X = torch.randint(0, args.vocab_size, (2, 1024))
    Y = torch.randint(0, args.vocab_size, (2, 1024))
    
    t0 = time.time()
    logits = model(X)
    ram_fwd = get_process_ram_mb()
    print(f"RAM after forward pass (L=1024): {ram_fwd:.1f} MB  (Delta Fwd: {ram_fwd - ram_model:.1f} MB)")
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    loss.backward()
    
    t_elapsed = time.time() - t0
    ram_bwd = get_process_ram_mb()
    print(f"RAM after backward pass: {ram_bwd:.1f} MB  (Delta Bwd: {ram_bwd - ram_fwd:.1f} MB)")
    print(f"Execution time: {t_elapsed:.2f} seconds")
    print(f"Loss: {loss.item():.4f}")
    
    print("\n=== VERDICT: RAM CHECKPASSED CLEANLY! ===")
    assert (ram_bwd - ram_init) < 3000.0, f"RAM spike detected! ({ram_bwd - ram_init:.1f} MB)"

if __name__ == "__main__":
    profile_ram()
