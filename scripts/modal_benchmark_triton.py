
import modal
import time
import os
from pathlib import Path

# Configuración de Modal siguiendo tu patrón de 'supermario_optimizer'
# Definimos la imagen y añadimos los directorios locales directamente
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("triton", "torch", "numpy")
    # Añadimos los directorios necesarios del proyecto
    .add_local_dir("kernels", remote_path="/root/kernels")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App("triton-fwht-benchmark")

@app.function(
    image=image, 
    gpu="A100",
    timeout=600
)
def benchmark_triton():
    import sys
    import torch
    import time
    # Añadimos /root al path para que encuentre los módulos
    sys.path.append("/root")
    
    try:
        from kernels.fwht_triton import fwht_triton
    except ImportError as e:
        print(f"Error al importar el kernel: {e}")
        # Listar archivos para debug si falla
        print(f"Archivos en /root: {os.listdir('/root')}")
        return

    # 1. Definir carga de trabajo
    B, N = 128, 32768
    x = torch.randn(B, N, device="cuda")
    
    print(f"\n" + "="*40)
    print(f"🚀 BENCHMARK TRITON EN NVIDIA A100")
    print(f"Configuración: Batch={B}, Dimensión={N}")
    print("="*40)

    # --- VERSION TRADICIONAL (Iterativa en GPU) ---
    def fwht_iterative(x):
        b, n = x.shape
        res = x.clone()
        h = 1
        while h < n:
            res = res.view(b, n // (2 * h), 2, h)
            a, b_ = res[:, :, 0, :], res[:, :, 1, :]
            res = torch.stack([a + b_, a - b_], dim=2)
            h *= 2
        return res.view(b, n) / (n ** 0.5)

    # Warmup
    print("Compilando y calentando kernels...")
    for _ in range(5):
        fwht_iterative(x)
        fwht_triton(x.clone())
    
    torch.cuda.synchronize()
    
    # Test Iterativo
    print("Midiendo FWHT Iterativa (PyTorch)...")
    t0 = time.perf_counter()
    for _ in range(20):
        _ = fwht_iterative(x)
    torch.cuda.synchronize()
    t_iter = (time.perf_counter() - t0) / 20 * 1000
    print(f" -> Resultado: {t_iter:.4f} ms")

    # Test Triton
    print("Midiendo FWHT Triton (Fused Kernel)...")
    t0 = time.perf_counter()
    for _ in range(20):
        _ = fwht_triton(x.clone())
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 20 * 1000
    print(f" -> Resultado: {t_triton:.4f} ms")
    
    improvement = t_iter / t_triton
    print(f"\n🔥 MEJORA LOGRADA: {improvement:.2f}x")

    # Verificación
    res_iter = fwht_iterative(x)
    res_triton = fwht_triton(x.clone())
    diff = torch.abs(res_iter - res_triton).max()
    print(f"Diferencia máxima: {diff:.6e} (OK si < 1e-5)")
    print("="*40 + "\n")

@app.local_entrypoint()
def main():
    print("Iniciando benchmark en Modal...")
    benchmark_triton.remote()

if __name__ == "__main__":
    main()
