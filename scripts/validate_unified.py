"""
validate_unified.py — Verifica que model_spectral_unified carga, hace
forward+backward, y que WalshLinearFWHT (kernel) coincide numéricamente con
el WalshLinear denso original (V10).

Uso:
    python scripts/validate_unified.py
"""
import sys
import torch

sys.path.insert(0, '.')
from model.walsh_linear_fwht import WalshLinearFWHT, fwht_torch
from model.model_spectral_v10_hippocampus import WalshLinear as WalshLinearDense
from model.model_spectral_unified import UnifiedSpectral, UnifiedArgs


def test_walsh_equivalence():
    torch.manual_seed(0)
    din, dout, k = 64, 64, 16
    x = torch.randn(4, 8, din, dtype=torch.float32)

    dense = WalshLinearDense(din, dout, k, normalized=False)
    fast = WalshLinearFWHT(din, dout, k, normalized=False)

    with torch.no_grad():
        fast.core.copy_(dense.core)

    yd = dense(x)
    yf = fast(x)
    max_err = (yd - yf).abs().max().item()
    print(f"[Walsh equiv] max abs error vs dense (raw) = {max_err:.3e}")
    assert max_err < 1e-3, "WalshLinearFWHT NO coincide con el denso"
    print("  OK: síntesis idéntica (kernel == denso)")


def test_fwht_correctness():
    # FWHT de un vector ones debe dar sqrt(n) en la coord 0 y 0 en el resto
    n = 16
    x = torch.ones(1, n)
    y = fwht_torch(x)
    expected0 = (n ** 0.5)
    err = abs(y[0, 0].item() - expected0) + y[0, 1:].abs().sum().item()
    print(f"[FWHT] check ones -> coord0={y[0,0].item():.3f} (esperado {expected0:.3f}), resto sum={y[0,1:].abs().sum().item():.3e}")
    assert err < 1e-4
    print("  OK: FWHT correcto")


def test_unified_forward_backward():
    args = UnifiedArgs(
        dim=256, emb_dim=0, n_layers=3, vocab_size=1000,
        max_seq_len=128, k_walsh=32, use_hippocampus=True,
        k_mem=16, chunk_size=128, spherical_head=True, weight_tying=True,
    )
    model = UnifiedSpectral(args)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[Unified] params = {nparams:,}")

    x = torch.randint(0, args.vocab_size, (2, args.max_seq_len))
    out = model(x)
    print(f"[Unified] logits shape = {tuple(out.shape)}")
    loss = out.log_softmax(-1).mean() + model.get_aux_loss()
    loss.backward()
    print(f"[Unified] loss = {loss.item():.4f}, backward OK")

    # Chequeo de hipocampus: el estado se propaga y no es None
    h = norm_sphere_check(model, x)
    print("  OK: forward+backward con hippocampus")


def norm_sphere_check(model, x):
    # Pequeño forward manual para confirmar que los estados de memoria se generan
    h = model.embed(x)
    if hasattr(model, 'embed_proj') and model.embed_proj is not None:
        h = model.embed_proj(h)
    h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    states = [None] * len(model.blocks)
    for i, block in enumerate(model.blocks):
        h, states[i] = block(h, states[i])
    assert any(s is not None for s in states), "hippocampus no produjo estado"
    return h


if __name__ == '__main__':
    print("=== Validación model_spectral_unified ===")
    test_fwht_correctness()
    test_walsh_equivalence()
    test_unified_forward_backward()
    print("\nTODO OK")
