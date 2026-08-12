import torch

def test_raw_core_equivalence():
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 16
    C = 64
    inv_dk = 1.0 / float(D)
    
    # Generate random phasors K, Q and values V, beta
    theta_k = torch.randn(B, H, L, D)
    theta_q = torch.randn(B, H, L, D)
    K = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
    Q = torch.complex(torch.cos(theta_q), torch.sin(theta_q))
    V = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))
    
    # 1. Sequential Token-by-Token Recurrence
    M_seq = torch.zeros(B, H, D, D, dtype=torch.complex64)
    out_seq_list = []
    for t in range(L):
        kt = K[:, :, t]
        qt = Q[:, :, t]
        vt = V[:, :, t]
        bt = beta[:, :, t]
        
        k_conj = torch.conj(kt)
        q_conj = torch.conj(qt)
        
        v_old = torch.matmul(M_seq, k_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
        err = vt - v_old
        update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
        M_seq = M_seq + bt.unsqueeze(-1).unsqueeze(-1) * update
        
        ret = torch.matmul(M_seq, q_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
        out_seq_list.append(ret)
        
    out_seq = torch.stack(out_seq_list, dim=2) # [B, H, L, D]
    
    # 2. Parallel WY Matrix Chunkwise Algorithm
    num_chunks = L // C
    Q_c = Q.view(B, H, num_chunks, C, D)
    K_c = K.view(B, H, num_chunks, C, D)
    V_c = V.view(B, H, num_chunks, C, D)
    beta_c = beta.view(B, H, num_chunks, C)
    
    Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
    L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
    I_mat = torch.eye(C).view(1, 1, 1, C, C)
    T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
    
    M_chunk = torch.zeros(B, H, D, D, dtype=torch.complex64)
    out_chunk_list = []
    
    for c in range(num_chunks):
        qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
        v_old = torch.matmul(M_chunk, torch.conj(kc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
        E_c = torch.matmul(tc, vc - v_old)
        U_c = bc.unsqueeze(-1) * E_c
        o_inter = torch.matmul(M_chunk, torch.conj(qc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
        A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1, -2)).real) * inv_dk
        out_chunk_list.append(torch.matmul(A_intra, U_c) + o_inter)
        M_chunk = M_chunk + torch.matmul(U_c.to(torch.complex64).transpose(-1, -2), kc)
        
    out_chunk = torch.cat(out_chunk_list, dim=2) # [B, H, L, D]
    
    # Measure Difference
    out_diff = (out_seq - out_chunk).abs().max().item()
    state_diff = (M_seq - M_chunk).abs().max().item()
    
    print("=" * 70)
    print(f"Raw Core Recurrence vs Parallel Chunkwise Matrix Algorithm")
    print(f"Output Max Diff : {out_diff:.6e}")
    print(f"State Max Diff  : {state_diff:.6e}")
    print("=" * 70)

if __name__ == "__main__":
    test_raw_core_equivalence()
