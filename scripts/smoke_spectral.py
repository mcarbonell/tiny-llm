import sys, time
sys.path.insert(0, '.')
from model.model_spectral import SpectralThinker, SpectralArgs, count_params
import torch

args = SpectralArgs(
    dim=256, n_layers=6, n_heads=8, n_kv_heads=4,
    vocab_size=16384, max_seq_len=256,
    k_dim_attn=64, k_dim_ffn=64, k_hidden_ffn=128
)

model = SpectralThinker(args)
stats = count_params(model)
print("=== SpectralThinker v2 (Optimizado) ===")
print("  Entrenables: {:,} ({:.2f}M)".format(stats["trainable"], stats["trainable_M"]))

# Forward smoke test
tokens  = torch.randint(0, args.vocab_size, (2, 32))
targets = torch.randint(0, args.vocab_size, (2, 32))
logits, loss = model(tokens, targets)
print("  Forward OK: logits={}, loss={:.4f}".format(tuple(logits.shape), loss.item()))

# Medir velocidad: 10 pasos con batch/seq similar al entrenamiento real
print()
print("Benchmark velocidad (batch=4, seq=64, 10 pasos):")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
times = []
for i in range(12):
    x = torch.randint(0, args.vocab_size, (4, 64))
    y = torch.randint(0, args.vocab_size, (4, 64))
    t0 = time.time()
    lg, ls = model(x, y)
    opt.zero_grad()
    ls.backward()
    opt.step()
    dt = time.time() - t0
    if i >= 2:  # descartar warm-up
        times.append(dt)
    print("  step {:2d} | loss={:.4f} | {:.3f}s".format(i+1, ls.item(), dt))

avg = sum(times) / len(times)
print()
print("Tiempo medio por step (sin warmup): {:.3f}s".format(avg))
print("Estimacion 5000 iters en CPU: {:.1f}h".format(avg * 5000 / 3600))
print()
print("OK - modelo v2 funcional y listo para relanzar.")
