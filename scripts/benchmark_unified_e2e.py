"""
benchmark_unified_e2e.py — Mide forward (y backward) del modelo unificado a escala
real y reporta nº de parámetros + memoria pico. Sin entrenar.

Uso: python scripts/benchmark_unified_e2e.py
"""
import sys, time, torch
sys.path.insert(0, '.')
from model.model_spectral_unified import UnifiedSpectral, UnifiedArgs

torch.manual_seed(0)
args = UnifiedArgs(
    dim=2048, emb_dim=256, n_layers=8, vocab_size=32768,
    max_seq_len=1024, k_walsh=256, use_hippocampus=True,
    k_mem=32, chunk_size=1024, spherical_head=True, weight_tying=True,
)
model = UnifiedSpectral(args)
model.eval()
nparams = sum(p.numel() for p in model.parameters())
print(f"PARAMS = {nparams:,}")

B, S = 16, 1024
x = torch.randint(0, args.vocab_size, (B, S))

# warmup
with torch.no_grad():
    model(x)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

t0 = time.time()
with torch.no_grad():
    out = model(x)
tf = time.time() - t0
print(f"forward 1 batch (B={B}, S={S}): {tf:.2f}s  logits={tuple(out.shape)}")

# backward (mide el paso de entrenamiento real por micro-batch)
model.train()
x2 = torch.randint(0, args.vocab_size, (B, S))
t0 = time.time()
out2 = model(x2)
(out2.log_softmax(-1).mean() + model.get_aux_loss()).backward()
tb = time.time() - t0
print(f"forward+backward 1 batch: {tb:.2f}s")

# Estimacion por iter (grad_accum=4 => 4 micro-batches)
est = tb * 4
print(f"est. iter (grad_accum=4, batch=16): ~{est:.1f}s  (~{est/60:.1f} min)")

# Memoria de parametros
mb = nparams * 4 / 1e6
print(f"param memory (fp32): ~{mb:.1f} MB")
print("OK")
