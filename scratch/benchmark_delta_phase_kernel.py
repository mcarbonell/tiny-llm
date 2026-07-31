"""
benchmark_delta_phase_kernel.py
===================================
Benchmarking and Kernel C++ Extension for Complex Phase Delta Memory Inference.

Goal:
  Implement a C++/PyTorch C++ inline extension (AVX-512 CPU / C++ fused loop) for the Complex Phase Delta Rule:
    State: M in C^{H x d_k x d_k}
    Per token:
      1. v_old_t = Re( M * conj(K_t) ) / d_k
      2. err_t   = V_t - v_old_t
      3. M      += beta_t * (err_t (x) K_t)
      4. R_t     = Re( M * conj(Q_t) ) / d_k

Measures:
  - Tokens / second throughput in single-token streaming mode (O(1) RAM)
  - Sequence lengths: L = 1,000, 5,000, 10,000 tokens
  - Compares Python loop vs Fused PyTorch C++ Extension
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. Fused C++ Kernel Source Code (Inline PyTorch C++ Extension) ──────
cpp_source = """
#include <torch/extension.h>
#include <vector>
#include <complex>

// Single-token streaming step in C++
// M: [B, H, d_k, d_k] complex float
// K: [B, H, d_k] complex float
// Q: [B, H, d_k] complex float
// V: [B, H, d_k] float
// beta: [B, H] float
// R_out: [B, H, d_k] float (output)
void delta_phase_step_cpp(
    torch::Tensor M,
    torch::Tensor K,
    torch::Tensor Q,
    torch::Tensor V,
    torch::Tensor beta,
    torch::Tensor R_out,
    float inv_dk
) {
    auto B = M.size(0);
    auto H = M.size(1);
    auto d_k = M.size(2);

    auto M_a = M.accessor<c10::complex<float>, 4>();
    auto K_a = K.accessor<c10::complex<float>, 3>();
    auto Q_a = Q.accessor<c10::complex<float>, 3>();
    auto V_a = V.accessor<float, 3>();
    auto b_a = beta.accessor<float, 2>();
    auto R_a = R_out.accessor<float, 3>();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            float b_val = b_a[b][h];
            
            // 1. Compute v_old_i = Re( sum_j M_ij * conj(K_j) ) * inv_dk
            for (int i = 0; i < d_k; ++i) {
                float v_old_i = 0.0f;
                for (int j = 0; j < d_k; ++j) {
                    c10::complex<float> m_ij = M_a[b][h][i][j];
                    c10::complex<float> k_j = K_a[b][h][j];
                    c10::complex<float> k_conj = std::conj(k_j);
                    c10::complex<float> prod = m_ij * k_conj;
                    v_old_i += prod.real();
                }
                v_old_i *= inv_dk;
                
                // 2. Compute error
                float err_i = V_a[b][h][i] - v_old_i;
                c10::complex<float> err_c(err_i * b_val, 0.0f);
                
                // 3. Update M_ij += beta * err_i * K_j
                for (int j = 0; j < d_k; ++j) {
                    M_a[b][h][i][j] += err_c * K_a[b][h][j];
                }
            }
            
            // 4. Compute readout R_i = Re( sum_j M_ij * conj(Q_j) ) * inv_dk
            for (int i = 0; i < d_k; ++i) {
                float r_i = 0.0f;
                for (int j = 0; j < d_k; ++j) {
                    c10::complex<float> m_ij = M_a[b][h][i][j];
                    c10::complex<float> q_j = Q_a[b][h][j];
                    c10::complex<float> q_conj = std::conj(q_j);
                    c10::complex<float> prod = m_ij * q_conj;
                    r_i += prod.real();
                }
                R_a[b][h][i] = r_i * inv_dk;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("delta_phase_step", &delta_phase_step_cpp, "Complex Phase Delta Step (C++)");
}
"""

def load_cpp_kernel():
    try:
        from torch.utils.cpp_extension import load_inline
        print("Compiling Fused C++ Extension for Complex Phase Delta Step...")
        cpp_module = load_inline(
            name="delta_phase_cpp",
            cpp_sources=cpp_source,
            extra_cflags=["-O3", "-fopenmp"] if not torch.cuda.is_available() else ["-O3"],
            verbose=False
        )
        print("C++ Kernel compiled successfully!")
        return cpp_module
    except Exception as e:
        print(f"Warning: C++ compilation failed ({e}). Falling back to pure PyTorch.")
        return None

# ── 2. Pure PyTorch Implementation ──────────────────────────────────────
def delta_phase_step_pytorch(M, K, Q, V, beta, inv_dk):
    """
    M: [B, H, d_k, d_k] complex
    K, Q: [B, H, d_k] complex
    V: [B, H, d_k] real
    beta: [B, H, 1, 1] real
    """
    K_conj = torch.conj(K)
    Q_conj = torch.conj(Q)
    
    # 1. Readout current prediction: v_old = Re( M * conj(K) ) * inv_dk
    v_old = torch.einsum('bhij,bhj->bhi', M, K_conj).real * inv_dk
    
    # 2. Residual error
    err = V - v_old
    
    # 3. Outer product update: M = M + beta * (err (x) K)
    update = torch.einsum('bhi,bhj->bhij', err.to(torch.complex64), K)
    M.add_(beta * update)
    
    # 4. Query readout: R = Re( M * conj(Q) ) * inv_dk
    R_out = torch.einsum('bhij,bhj->bhi', M, Q_conj).real * inv_dk
    return R_out

# ── 3. Streaming Inference Benchmark ───────────────────────────────────
def benchmark_inference(cpp_module=None, B=1, H=8, d_k=32, seq_len=2000):
    inv_dk = 1.0 / float(d_k)
    device = torch.device("cpu")
    
    print(f"\n==================================================")
    print(f" Benchmark: Streaming Inference Speed (O(1) RAM)")
    print(f" Batch Size: {B} | Heads: {H} | d_k: {d_k} | Tokens: {seq_len}")
    print(f"==================================================")
    
    # Generate random streaming inputs
    theta_k = torch.randn(seq_len, B, H, d_k, device=device)
    theta_q = torch.randn(seq_len, B, H, d_k, device=device)
    V_seq   = torch.randn(seq_len, B, H, d_k, device=device)
    beta_seq= torch.sigmoid(torch.randn(seq_len, B, H, device=device))
    
    K_seq = torch.polar(torch.ones_like(theta_k), theta_k)
    Q_seq = torch.polar(torch.ones_like(theta_q), theta_q)
    
    # ── Test 1: Pure PyTorch Streaming Loop ──
    M_pt = torch.zeros(B, H, d_k, d_k, dtype=torch.complex64, device=device)
    
    t0 = time.time()
    for t in range(seq_len):
        _ = delta_phase_step_pytorch(
            M_pt, K_seq[t], Q_seq[t], V_seq[t], beta_seq[t].view(B, H, 1, 1), inv_dk
        )
    t_pt = time.time() - t0
    tok_per_sec_pt = seq_len / t_pt
    print(f"  PyTorch Vectorized : {t_pt:.4f} s | {tok_per_sec_pt:8.1f} tokens/sec")
    
    # ── Test 2: Fused C++ Kernel ──
    if cpp_module is not None:
        M_cpp = torch.zeros(B, H, d_k, d_k, dtype=torch.complex64, device=device)
        R_out = torch.zeros(B, H, d_k, dtype=torch.float32, device=device)
        
        t0 = time.time()
        for t in range(seq_len):
            cpp_module.delta_phase_step(
                M_cpp, K_seq[t], Q_seq[t], V_seq[t], beta_seq[t], R_out, inv_dk
            )
        t_cpp = time.time() - t0
        tok_per_sec_cpp = seq_len / t_cpp
        speedup = t_pt / t_cpp
        print(f"  Fused C++ Kernel   : {t_cpp:.4f} s | {tok_per_sec_cpp:8.1f} tokens/sec  (Speedup: {speedup:.2f}x)")
        
        # Verify correctness (numerical agreement)
        diff = torch.max(torch.abs(M_pt - M_cpp)).item()
        print(f"  [Correctness Check] Max State Difference (PyTorch vs C++): {diff:.6e}")

def main():
    print("Initializing Complex Phase Delta Memory Kernel Benchmark...")
    cpp_module = load_cpp_kernel()
    
    # Run benchmark for context lengths
    benchmark_inference(cpp_module, B=1, H=8, d_k=32, seq_len=1000)
    benchmark_inference(cpp_module, B=1, H=8, d_k=32, seq_len=5000)

if __name__ == "__main__":
    main()
