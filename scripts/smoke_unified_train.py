"""
smoke_unified_train.py — Construye el modelo unificado EXACTAMENTE como lo haria
train.py (mismo mapping de config -> UnifiedArgs) y hace un paso de optimizacion
falso. No entrena; solo valida que el path de train.py funciona end-to-end.

Uso: python scripts/smoke_unified_train.py
"""
import sys, os, yaml, torch
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from model.model_spectral_unified import UnifiedSpectral, UnifiedArgs

with open('configs/train_unified_v1.yaml') as f:
    cfg = yaml.safe_load(f)

# Mismo mapping que train.py (arch == 'unified')
ua = UnifiedArgs(
    dim=cfg['dim'], emb_dim=cfg.get('emb_dim', 0), n_layers=cfg['n_layers'],
    vocab_size=cfg['vocab_size'], max_seq_len=cfg['max_seq_len'],
    k_walsh=cfg['k_walsh'], use_hippocampus=cfg.get('use_hippocampus', True),
    k_mem=cfg.get('k_mem', 32), chunk_size=cfg.get('chunk_size', 1024),
    gamma=cfg.get('gamma', 0.9), lambda_phase=cfg.get('lambda_phase', 0.01),
    spherical_head=cfg.get('spherical_head', True), weight_tying=cfg.get('weight_tying', True),
)
model = UnifiedSpectral(ua)
nparams = sum(p.numel() for p in model.parameters())
print(f"[unified] params = {nparams:,}")

opt = torch.optim.AdamW(model.parameters(), lr=1.5e-2)
B, S = 16, 1024
x = torch.randint(0, cfg['vocab_size'], (B, S))
import time
t0 = time.time()
out = model(x)
loss = out.log_softmax(-1).mean() + model.get_aux_loss()
loss.backward()
opt.step()
print(f"[unified] 1 step (B={B},S={S}) = {time.time()-t0:.1f}s  loss={loss.item():.3f}")
print("OK: train.py path para 'unified' funcional")
