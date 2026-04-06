import torch

def precompute_freqs_cis_orig(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def precompute_freqs_cis_new(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return freqs

def reshape_for_broadcast(freqs, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1]), f"{freqs.shape} vs {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)

def apply_rotary_emb_orig(xq: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)

def apply_rotary_emb_new(xq: torch.Tensor, freqs: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    freqs = reshape_for_broadcast(freqs, xq_[..., 0])
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    xq_0, xq_1 = xq_.unbind(-1)
    
    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos
    
    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).flatten(3)
    return xq_out.type_as(xq)

xq = torch.randn(2, 4, 8, 32) # batch, seq, heads, head_dim
freqs_cis = precompute_freqs_cis_orig(32, 8)
freqs = precompute_freqs_cis_new(32, 8)

out_orig = apply_rotary_emb_orig(xq, freqs_cis[:4])
out_new = apply_rotary_emb_new(xq, freqs[:4])

print("Difference:", (out_orig - out_new).abs().max().item())
