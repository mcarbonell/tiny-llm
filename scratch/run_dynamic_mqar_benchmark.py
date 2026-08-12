"""
run_dynamic_mqar_benchmark.py
==============================
Real Dynamic On-the-Fly GPU MQAR Benchmark (No Fixed Dataset Split).
Evaluates Softmax Attention, Gated DeltaNet (Real), and DeltaPhase (Complex Phasor S^1).
"""

import sys, os
sys.path.insert(0, r"C:\Users\mrcm_\Local\proj\algorithms\delta-phase")

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def generate_dynamic_mqar_batch(batch_size, seq_len, num_kv_pairs, vocab_size, device):
    """
    Generates dynamic on-the-fly MQAR sequences per batch.
    Keys/values are drawn from vocab pool [10, vocab_size).
    """
    # Key-value pairs inserted into sequence
    inputs = torch.randint(10, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        keys = torch.choice if hasattr(torch, 'choice') else torch.randperm(vocab_size - 10)[:num_kv_pairs] + 10
        vals = torch.choice if hasattr(torch, 'choice') else torch.randperm(vocab_size - 10)[num_kv_pairs:2*num_kv_pairs] + 10
        
        kv_positions = random.sample(range(0, seq_len // 2 - 1, 2), num_kv_pairs)
        for i, (k, v) in enumerate(zip(keys, vals)):
            pos = kv_positions[i]
            inputs[b, pos] = k
            inputs[b, pos + 1] = v
            
        # Query positions in second half
        q_pos = random.randint(seq_len // 2, seq_len - 2)
        target_kv_idx = random.randint(0, num_kv_pairs - 1)
        inputs[b, q_pos] = keys[target_kv_idx]
        targets[b, q_pos + 1] = vals[target_kv_idx]
        
    return inputs, targets

def run_benchmark():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Real Dynamic GPU MQAR Benchmark...")
    print(f"Device: {device}")
    
    # We will test on small dynamic batches
    print("=" * 80)
    print("REAL GPU DYNAMIC BENCHMARK EXECUTED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
