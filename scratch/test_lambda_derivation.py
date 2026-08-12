"""
test_lambda_derivation.py
=========================
Rigorous Mathematical & Empirical Test of Memory Retention Factor (lambda)
Compares Case A (Error vs unattenuated v_old) and Case B (Error vs decayed lambda * v_old).
"""

import torch

def run_lambda_suite():
    d_k = 32
    inv_dk = 1.0 / float(d_k)
    
    lambdas = [0.85, 0.95, 1.00]
    betas = [0.5, 1.0]
    
    print("=" * 85)
    print("LAMBDA RETENTION FACTOR & ERROR DEFINITION SUITE")
    print("=" * 85)
    
    for lam in lambdas:
        for beta in betas:
            # Random initial memory state M_prev and key K, target V
            torch.manual_seed(42)
            M_prev = torch.randn(1, 1, d_k, d_k, dtype=torch.complex64)
            theta_k = torch.randn(1, 1, d_k)
            K = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
            V = torch.randn(1, 1, d_k)
            
            K_conj = torch.conj(K)
            v_old = torch.matmul(M_prev, K_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            
            # --- CASE A: e = V - v_old ---
            e_A = V - v_old
            update_A = torch.matmul(e_A.to(torch.complex64).unsqueeze(-1), K.unsqueeze(-2))
            M_A = lam * M_prev + beta * update_A
            v_hat_A = torch.matmul(M_A, K_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            
            # Formula A prediction: beta * V + (lam - beta) * v_old
            v_pred_A = beta * V + (lam - beta) * v_old
            diff_A = (v_hat_A - v_pred_A).abs().max().item()
            error_to_target_A = (v_hat_A - V).abs().max().item()
            
            # --- CASE B: e_att = V - lam * v_old ---
            e_B = V - lam * v_old
            update_B = torch.matmul(e_B.to(torch.complex64).unsqueeze(-1), K.unsqueeze(-2))
            M_B = lam * M_prev + beta * update_B
            v_hat_B = torch.matmul(M_B, K_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            
            # Formula B prediction: beta * V + (1 - beta) * lam * v_old
            v_pred_B = beta * V + (1.0 - beta) * lam * v_old
            diff_B = (v_hat_B - v_pred_B).abs().max().item()
            error_to_target_B = (v_hat_B - V).abs().max().item()
            
            print(f"Lambda={lam:.2f} | Beta={beta:.1f}")
            print(f"  Case A (e = V - v_old)     -> Formula Match Diff: {diff_A:.2e} | Target Error: {error_to_target_A:.4f}")
            print(f"  Case B (e = V - lam*v_old) -> Formula Match Diff: {diff_B:.2e} | Target Error: {error_to_target_B:.4f}")
            print("-" * 85)

if __name__ == "__main__":
    run_lambda_suite()
