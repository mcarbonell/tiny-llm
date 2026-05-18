import torch
import os
import subprocess
import ctypes

# Paths
kernel_dir = os.path.dirname(os.path.abspath(__file__))
cpp_source = os.path.join(kernel_dir, "fused_residual_norm_cpu.cpp")
dll_name = "fused_residual_norm_cpu.dll" if os.name == 'nt' else "fused_residual_norm_cpu.so"
dll_path = os.path.join(kernel_dir, dll_name)

_native_lib = None
_failed_to_load = False

def compile_and_load_residual_norm():
    global _native_lib, _failed_to_load
    if _native_lib is not None:
        return _native_lib
    if _failed_to_load:
        return None
    
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(kernel_dir)
        try:
            gcc_path = subprocess.check_output(["where", "g++"], text=True).split('\n')[0].strip()
            gcc_dir = os.path.dirname(gcc_path)
            if os.path.exists(gcc_dir):
                os.add_dll_directory(gcc_dir)
        except:
            pass

    needs_compile = not os.path.exists(dll_path)
    if not needs_compile:
        needs_compile = os.path.getmtime(cpp_source) > os.path.getmtime(dll_path)
        
    if needs_compile:
        print("Compilando kernel nativo FusedResidualNorm con g++...")
        try:
            cmd = [
                "g++", "-O3", "-shared", "-fopenmp", "-mavx2",
                cpp_source, "-o", dll_path
            ]
            if os.name == 'nt':
                cmd += ["-static-libgcc", "-static-libstdc++"]
                
            subprocess.run(cmd, check=True, capture_output=True)
            print("¡Compilación de FusedResidualNorm exitosa!")
        except Exception as e:
            print(f"Error compilando FusedResidualNorm: {e}")
            _failed_to_load = True
            return None
            
    try:
        lib = ctypes.CDLL(dll_path)
        # Forward argtypes: const float* x, const float* y, const float* alpha, float* out, float* norms, int batch_seq, int dim
        lib.fused_residual_norm_forward_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        lib.fused_residual_norm_forward_cpu.restype = None

        # Backward argtypes: const float* grad_out, const float* out, const float* y, const float* alpha, const float* norms, float* grad_x, float* grad_y, float* grad_alpha, int batch_seq, int dim
        lib.fused_residual_norm_backward_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        lib.fused_residual_norm_backward_cpu.restype = None
        
        _native_lib = lib
        return lib
    except Exception as e:
        print(f"Error cargando librería nativa {dll_name}: {e}")
        _failed_to_load = True
        return None

# PyTorch Autograd Function for CPU C++ Native
class FusedResidualNormCPUNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, alpha):
        # We assume input is contig and float32 for C++ ctypes
        orig_x_dtype = x.dtype
        orig_y_dtype = y.dtype
        orig_alpha_dtype = alpha.dtype
        
        x_f32 = x.float().contiguous()
        y_f32 = y.float().contiguous()
        alpha_f32 = alpha.float().contiguous()
        
        orig_shape = x_f32.shape
        dim = orig_shape[-1]
        x_flat = x_f32.view(-1, dim)
        y_flat = y_f32.view(-1, dim)
        batch_seq = x_flat.size(0)
        
        out_flat = torch.empty_like(x_flat)
        norms = torch.empty(batch_seq, dtype=torch.float32)
        
        lib = compile_and_load_residual_norm()
        if lib is None:
            # Fallback a PyTorch puro
            abs_alpha = alpha_f32.abs().unsqueeze(0)
            u = x_flat + abs_alpha * y_flat
            norms = u.norm(dim=-1, keepdim=True) + 1e-8
            out_flat = u / norms
            norms = norms.squeeze(-1)
        else:
            ptr_x = ctypes.cast(x_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_y = ctypes.cast(y_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_alpha = ctypes.cast(alpha_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_out = ctypes.cast(out_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_norms = ctypes.cast(norms.data_ptr(), ctypes.POINTER(ctypes.c_float))
            
            lib.fused_residual_norm_forward_cpu(
                ptr_x, ptr_y, ptr_alpha, ptr_out, ptr_norms, batch_seq, dim
            )
            
        out = out_flat.view(orig_shape).to(orig_x_dtype)
        
        ctx.save_for_backward(out_flat, y_flat, alpha_f32, norms)
        ctx.batch_seq = batch_seq
        ctx.dim = dim
        ctx.orig_shape = orig_shape
        ctx.orig_x_dtype = orig_x_dtype
        ctx.orig_y_dtype = orig_y_dtype
        ctx.orig_alpha_dtype = orig_alpha_dtype
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out_flat, y_flat, alpha_f32, norms = ctx.saved_tensors
        batch_seq = ctx.batch_seq
        dim = ctx.dim
        orig_shape = ctx.orig_shape
        
        grad_out_f32 = grad_out.float().contiguous().view(-1, dim)
        
        grad_x_flat = torch.empty_like(out_flat)
        grad_y_flat = torch.empty_like(out_flat)
        grad_alpha = torch.empty(dim, dtype=torch.float32)
        
        lib = compile_and_load_residual_norm()
        if lib is None:
            # Fallback PyTorch puro
            s = torch.sum(grad_out_f32 * out_flat, dim=-1, keepdim=True)
            grad_u = (grad_out_f32 - out_flat * s) / norms.unsqueeze(-1)
            grad_x_flat = grad_u
            grad_y_flat = grad_u * alpha_f32.abs().unsqueeze(0)
            
            sign_alpha = torch.sign(alpha_f32)
            grad_alpha = torch.sum(grad_u * y_flat, dim=0) * sign_alpha
        else:
            ptr_gout = ctypes.cast(grad_out_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_out = ctypes.cast(out_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_y = ctypes.cast(y_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_alpha = ctypes.cast(alpha_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_norms = ctypes.cast(norms.data_ptr(), ctypes.POINTER(ctypes.c_float))
            
            ptr_gx = ctypes.cast(grad_x_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_gy = ctypes.cast(grad_y_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
            ptr_galpha = ctypes.cast(grad_alpha.data_ptr(), ctypes.POINTER(ctypes.c_float))
            
            lib.fused_residual_norm_backward_cpu(
                ptr_gout, ptr_out, ptr_y, ptr_alpha, ptr_norms,
                ptr_gx, ptr_gy, ptr_galpha, batch_seq, dim
            )
            
        grad_x = grad_x_flat.view(orig_shape).to(ctx.orig_x_dtype)
        grad_y = grad_y_flat.view(orig_shape).to(ctx.orig_y_dtype)
        grad_alpha = grad_alpha.to(ctx.orig_alpha_dtype)
        
        return grad_x, grad_y, grad_alpha

# Intento de cargar la versión Triton si está disponible
try:
    from kernels.fused_residual_norm_triton import fused_residual_norm_triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def fused_residual_norm(x, y, alpha):
    """
    Función unificada que redirige automáticamente al kernel óptimo:
    - Si el tensor está en GPU CUDA y Triton está instalado: usa FusedResidualNormTriton.
    - Si está en CPU o DirectML (o Triton no está disponible): usa FusedResidualNormCPUNative (C++ optimizado con OpenMP).
    """
    if x.is_cuda and HAS_TRITON:
        return fused_residual_norm_triton(x, y, alpha)
    else:
        # En CPU o DirectML, llamamos a la versión C++ nativa (que tiene fallback automático a PyTorch)
        return FusedResidualNormCPUNative.apply(x, y, alpha)

