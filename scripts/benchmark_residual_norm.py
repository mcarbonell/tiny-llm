import torch
import time
import os
import sys

# Add root directory to path to import kernels
sys.path.append(os.getcwd())

# Try loading torch_directml
try:
    import torch_directml
    gpu_device = torch_directml.device()
    print(f"DirectML disponible. GPU Device: {gpu_device}")
except ImportError:
    gpu_device = None
    print("DirectML no disponible para GPU")

from kernels.fused_residual_norm_op import fused_residual_norm

def pytorch_native_residual_norm(x, y, alpha):
    """
    Operación de referencia en PyTorch puro (desfusionada).
    """
    # Manejar 2D o 3D
    if len(x.shape) == 3:
        abs_alpha = alpha.abs().unsqueeze(0).unsqueeze(0)
    else:
        abs_alpha = alpha.abs().unsqueeze(0)
    u = x + abs_alpha * y
    return u / (u.norm(dim=-1, keepdim=True) + 1e-8)

def test_correctness():
    print("\n=== VERIFICANDO EXTREMA PRECISIÓN NUMÉRICA (PARIDAD) ===")
    
    B, T, D = 4, 128, 256
    x = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
    y = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(D, dtype=torch.float32, requires_grad=True)
    
    # Clonamos para verificar de forma independiente
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)
    alpha_ref = alpha.clone().detach().requires_grad_(True)
    
    # 1. FORWARD PASS
    out_ref = pytorch_native_residual_norm(x_ref, y_ref, alpha_ref)
    out_fused = fused_residual_norm(x, y, alpha)
    
    diff_forward = torch.max(torch.abs(out_ref - out_fused)).item()
    print(f"Diferencia Máxima en Forward: {diff_forward:.1e}")
    if diff_forward > 1e-5:
        print("ALERTA: La discrepancia en forward supera el umbral tolerable.")
        return False
        
    # 2. BACKWARD PASS
    # Usamos un gradiente externo aleatorio
    grad_out = torch.randn_like(out_ref)
    
    out_ref.backward(grad_out)
    out_fused.backward(grad_out)
    
    diff_gx = torch.max(torch.abs(x_ref.grad - x.grad)).item()
    diff_gy = torch.max(torch.abs(y_ref.grad - y.grad)).item()
    diff_galpha = torch.max(torch.abs(alpha_ref.grad - alpha.grad)).item()
    
    print(f"Diferencia Máxima en Gradiente X:     {diff_gx:.1e}")
    print(f"Diferencia Máxima en Gradiente Y:     {diff_gy:.1e}")
    print(f"Diferencia Máxima en Gradiente Alpha: {diff_galpha:.1e}")
    
    if max(diff_gx, diff_gy, diff_galpha) > 1e-4:
        print("ALERTA: La discrepancia en backward supera el umbral tolerable.")
        return False
        
    print("¡VERIFICACIÓN EXITOSA! Los gradientes y resultados coinciden perfectamente.")
    return True

def run_benchmark(B, T, D, device_name, device, iters=100):
    print(f"\n--- Benchmark en {device_name} (B={B}, T={T}, Dim={D}, Iters={iters}) ---")
    
    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(B, T, D, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = pytorch_native_residual_norm(x, y, alpha)
        _ = fused_residual_norm(x, y, alpha)
        
    # 1. Benchmark PyTorch Puro (Forward + Backward)
    t_fwd_pt = 0.0
    t_bwd_pt = 0.0
    
    for _ in range(iters):
        x.grad = None
        y.grad = None
        alpha.grad = None
        
        # Medimos Forward
        t0 = time.perf_counter()
        out = pytorch_native_residual_norm(x, y, alpha)
        if device.type != 'cpu':
            torch.zeros(1, device=device) + 1 # Sincronización DirectML
        t1 = time.perf_counter()
        t_fwd_pt += (t1 - t0)
        
        # Medimos Backward
        t0 = time.perf_counter()
        out.backward(grad_out, retain_graph=True)
        if device.type != 'cpu':
            torch.zeros(1, device=device) + 1
        t1 = time.perf_counter()
        t_bwd_pt += (t1 - t0)
        
    # 2. Benchmark Fused Kernel (Forward + Backward)
    t_fwd_fused = 0.0
    t_bwd_fused = 0.0
    
    for _ in range(iters):
        x.grad = None
        y.grad = None
        alpha.grad = None
        
        # Medimos Forward
        t0 = time.perf_counter()
        out = fused_residual_norm(x, y, alpha)
        if device.type != 'cpu':
            torch.zeros(1, device=device) + 1
        t1 = time.perf_counter()
        t_fwd_fused += (t1 - t0)
        
        # Medimos Backward
        t0 = time.perf_counter()
        out.backward(grad_out, retain_graph=True)
        if device.type != 'cpu':
            torch.zeros(1, device=device) + 1
        t1 = time.perf_counter()
        t_bwd_fused += (t1 - t0)
        
    # Tiempos promedio en milisegundos
    ms_fwd_pt = (t_fwd_pt / iters) * 1000
    ms_bwd_pt = (t_bwd_pt / iters) * 1000
    ms_total_pt = ms_fwd_pt + ms_bwd_pt
    
    ms_fwd_fused = (t_fwd_fused / iters) * 1000
    ms_bwd_fused = (t_bwd_fused / iters) * 1000
    ms_total_fused = ms_fwd_fused + ms_bwd_fused
    
    speedup_fwd = ms_fwd_pt / ms_fwd_fused
    speedup_bwd = ms_bwd_pt / ms_bwd_fused
    speedup_total = ms_total_pt / ms_total_fused
    
    print(f"{"Operación":20} | {"Forward (ms)":15} | {"Backward (ms)":15} | {"Total (ms)":15}")
    print("-" * 73)
    print(f"{"PyTorch Puro":20} | {ms_fwd_pt:13.3f} | {ms_bwd_pt:13.3f} | {ms_total_pt:13.3f}")
    print(f"{"Kernel Fusionado":20} | {ms_fwd_fused:13.3f} | {ms_bwd_fused:13.3f} | {ms_total_fused:13.3f}")
    print("-" * 73)
    print(f"{"Aceleración":20} | {speedup_fwd:12.2f}x | {speedup_bwd:12.2f}x | {speedup_total:12.2f}x")

def main():
    print("=============================================================")
    print("BENCHMARK: FUSED RESIDUAL UPDATE + SPHERICAL NORMALIZATION (nGPT)")
    print("=============================================================")
    
    # 1. Verificar exactitud matemática
    correct = test_correctness()
    if not correct:
        print("Saliendo debido a fallos de precisión.")
        return
        
    # 2. Benchmarks de rendimiento en CPU (Ryzen OpenMP/AVX2)
    # Probamos dimensiones relevantes para tus experimentos
    dims = [256, 1024, 4096]
    for d in dims:
        run_benchmark(B=4, T=256, D=d, device_name="AMD CPU (Nativo AVX2/OpenMP)", device=torch.device("cpu"), iters=50)
        
    # 3. Benchmark en GPU con DirectML si está disponible
    if gpu_device is not None:
        print("\n=== EJECUTANDO BENCHMARK EN GPU (DirectML) ===")
        for d in dims:
            run_benchmark(B=4, T=256, D=d, device_name="Radeon GPU (DirectML / Fallback)", device=gpu_device, iters=50)

if __name__ == "__main__":
    main()
