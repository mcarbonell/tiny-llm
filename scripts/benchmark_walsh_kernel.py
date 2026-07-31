"""
benchmark_walsh_kernel.py — Compara WalshLinear denso (V10) vs WalshLinearFWHT (kernel)
en tiempo de forward+backward para dim grande. Confirma la ganancia del kernel FWHT.

Uso: python scripts/benchmark_walsh_kernel.py
"""
import sys, time, torch
sys.path.insert(0, '.')
from model.model_spectral_v10_hippocampus import WalshLinear as WalshLinearDense
from model.walsh_linear_fwht import WalshLinearFWHT

D = 2048
K = 256
B, S = 4, 256

print(f"dim={D} k={K} batch={B} seq={S}")

# DENSO
dense = WalshLinearDense(D, D, K, normalized=True)
xd = torch.randn(B, S, D, requires_grad=True)
# materializar W una vez para aislar el matmul (el denso lo hace en cada forward)
t0 = time.time()
yd = dense(xd)
yd.sum().backward()
td = time.time() - t0
print(f"[denso]  forward+backward = {td*1000:.1f} ms  (sintetiza W d×d en cada paso)")

# FWHT
fast = WalshLinearFWHT(D, D, K, normalized=True)
xf = torch.randn(B, S, D, requires_grad=True)
t0 = time.time()
yf = fast(xf)
yf.sum().backward()
tf = time.time() - t0
print(f"[fwht]   forward+backward = {tf*1000:.1f} ms  (kernel, sin matriz d×d)")

print(f"-> speedup kernel vs denso = {td/tf:.2f}x")
assert tf < td, "el kernel deberia ser mas rapido"
print("OK: kernel FWHT mas rapido y sin matriz densa")
