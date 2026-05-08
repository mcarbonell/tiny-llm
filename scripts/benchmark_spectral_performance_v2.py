
import torch
import time
import os
import sys

# Añadir el directorio raíz al path para poder importar los modelos
sys.path.append(os.getcwd())

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Usando dispositivo: {device}")
except ImportError:
    device = torch.device("cpu")
    print("DirectML no disponible, usando CPU")

from kernels.fwht_op import fwht_native, fwht_native as fwht_cpu_native
from model.model_spectral_v8_6_universal import fwht_gpu_vectorized, FastUniversalFWHT

# Función de v8.4 (la más básica)
def fwht_v84(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.stack([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def benchmark():
    # Parámetros realistas de tus modelos
    B, T, D = 1, 128, 32768
    print(f"\nConfiguración: Batch={B}, SeqLen={T}, DimEspectral={D}")
    
    # Preparamos tensores
    x_gpu = torch.randn(B * T, D, device=device)
    x_cpu = torch.randn(B * T, D, device='cpu')
    
    iters = 10 # Menos iteraciones porque son lentos

    def measure(fn, name, input_tensor):
        # Warmup
        try:
            for _ in range(2):
                fn(input_tensor)
            
            if input_tensor.device.type != 'cpu':
                # Sincronización para DirectML (aproximada)
                torch.zeros(1, device=input_tensor.device) + 1
            
            start = time.perf_counter()
            for _ in range(iters):
                fn(input_tensor)
            
            if input_tensor.device.type != 'cpu':
                torch.zeros(1, device=input_tensor.device) + 1
            
            end = time.perf_counter()
            avg_ms = ((end - start) / iters) * 1000
            print(f"{name:35} | {avg_ms:8.2f} ms")
        except Exception as e:
            print(f"{name:35} | Error: {str(e)[:50]}...")

    print("\n--- FWHT Kernels en GPU (DirectML) ---")
    measure(fwht_v84, "v8.4 Iterativo (en GPU)", x_gpu)
    measure(fwht_gpu_vectorized, "v8.6 Vectorizado (en GPU)", x_gpu)
    measure(FastUniversalFWHT.apply, "v8.6 Universal (Auto-Offload)", x_gpu)

    print("\n--- FWHT Kernels en CPU ---")
    measure(fwht_v84, "v8.4 Iterativo (en CPU)", x_cpu)
    measure(fwht_cpu_native, "Kernel C++ Nativo (en CPU)", x_cpu)

if __name__ == "__main__":
    benchmark()
