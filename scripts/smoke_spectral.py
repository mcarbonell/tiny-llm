import sys, time, torch
sys.path.insert(0, '.')
from model.model_spectral import SpectralThinker, SpectralArgs, count_params
import numpy as np

print('=== SpectralThinker Mega-Nano (dim=1024, k=64, 8 capas) ===')

args = SpectralArgs(
    dim=1024, n_layers=8, n_heads=8, n_kv_heads=4,
    vocab_size=16384, max_seq_len=256,
    k_dim_attn=64, k_dim_ffn=64, k_hidden_ffn=128,
    ffn_dim_multiplier=2.0, multiple_of=256
)

t0 = time.time()
model = SpectralThinker(args)
print('Construccion: {:.2f}s'.format(time.time()-t0))

stats = count_params(model)
print('Params entrenables: {:,} ({:.2f}M)'.format(stats['trainable'], stats['trainable_M']))

spectral_only = sum(p.numel() for n, p in model.named_parameters() if 'core' in n)
print('Nucleos espectrales: {:,} ({:.1f}K)'.format(spectral_only, spectral_only/1e3))

data = np.memmap('data/train_v1.bin', dtype=np.uint16, mode='r')
x = torch.from_numpy(data[:256*4].reshape(4,256).astype('int64'))
y = torch.from_numpy(data[1:256*4+1].reshape(4,256).astype('int64'))
with torch.no_grad():
    _, loss = model(x, y)
print('Loss inicial (datos reales): {:.4f} (esperado ~9.7)'.format(loss.item()))

print()
print('Benchmark velocidad (batch=8, seq=256, 7 steps):')
opt = torch.optim.AdamW(model.parameters(), lr=8e-4)
model.train()
times = []
for i in range(7):
    xb = torch.from_numpy(data[i*256*8:(i+1)*256*8].reshape(8,256).astype('int64'))
    yb = torch.from_numpy(data[i*256*8+1:(i+1)*256*8+1].reshape(8,256).astype('int64'))
    t0 = time.time()
    _, ls = model(xb, yb)
    opt.zero_grad()
    ls.backward()
    opt.step()
    dt = time.time() - t0
    times.append(dt)
    print('  step {} | loss={:.4f} | {:.2f}s'.format(i+1, ls.item(), dt))

avg = sum(times[2:]) / len(times[2:])
print()
print('Tiempo medio/step (sin warmup): {:.2f}s'.format(avg))
# En train.py: grad_accum_steps=8, asi que cada "iter" son 8 forwards
print('Estimacion 3000 iters (grad_accum=8): {:.1f}h'.format(avg*8*3000/3600))
print()
print('OK -- Mega-Nano listo para entrenar.')
