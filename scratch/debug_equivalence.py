import torch

def verify():
    B, n_heads, d_k = 1, 1, 4
    inv_dk = 1.0 / float(d_k)
    
    # 1. Sequential step
    M = torch.zeros(B, n_heads, d_k, d_k, dtype=torch.complex64)
    
    k = torch.tensor([[[[1.0+0.0j, 0.0+1.0j, -1.0+0.0j, 0.0-1.0j]]]]) # B=1, L=1, H=1, D=4
    q = torch.tensor([[[[0.0+1.0j, 1.0+0.0j, 0.0-1.0j, -1.0+0.0j]]]])
    v = torch.tensor([[[[0.5, -0.2, 0.8, 0.1]]]])
    beta = torch.tensor([[[[0.9]]]])
    
    # Sequential Step execution
    k_t = k[:, :, 0]
    q_t = q[:, :, 0]
    v_t = v[:, :, 0]
    beta_t = beta[:, :, 0]
    
    v_old = torch.matmul(M, torch.conj(k_t).unsqueeze(-1)).squeeze(-1).real * inv_dk
    err = v_t - v_old
    update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), k_t.unsqueeze(-2))
    M_next = M + beta_t.unsqueeze(-1).unsqueeze(-1) * update
    
    # Readout with M_next vs M:
    ret_next = torch.matmul(M_next, torch.conj(q_t).unsqueeze(-1)).squeeze(-1).real * inv_dk
    ret_old = torch.matmul(M, torch.conj(q_t).unsqueeze(-1)).squeeze(-1).real * inv_dk
    
    print("M_next readout:", ret_next)
    print("M_old readout:", ret_old)
    
    # Self update contribution to readout:
    self_contrib = beta_t.unsqueeze(-1) * err * (torch.matmul(k_t, torch.conj(q_t).unsqueeze(-1)).squeeze(-1).real * inv_dk)
    print("Calculated ret from M_old + self_contrib:", ret_old + self_contrib)
    print("Match:", torch.allclose(ret_next, ret_old + self_contrib))

if __name__ == "__main__":
    verify()
