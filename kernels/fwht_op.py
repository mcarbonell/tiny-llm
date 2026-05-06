import torch
import os
import subprocess
import ctypes

# Rutas de archivos
kernel_dir = os.path.dirname(os.path.abspath(__file__))
cpp_source = os.path.join(kernel_dir, "fwht_cpu.cpp")
# En Windows usamos .dll
dll_name = "fwht_cpu.dll" if os.name == 'nt' else "fwht_cpu.so"
dll_path = os.path.join(kernel_dir, dll_name)

_native_lib = None
_failed_to_load = False

def compile_and_load():
    global _native_lib, _failed_to_load
    if _native_lib is not None:
        return _native_lib
    if _failed_to_load:
        return None
    
    # En Python 3.8+ en Windows, hay que añadir el directorio al search path de DLLs
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(kernel_dir)
        # También intentamos añadir la ruta de MinGW si la encontramos
        try:
            gcc_path = subprocess.check_output(["where", "g++"], text=True).split('\n')[0].strip()
            gcc_dir = os.path.dirname(gcc_path)
            if os.path.exists(gcc_dir):
                os.add_dll_directory(gcc_dir)
        except:
            pass

    # Si no existe la DLL o el fuente es más nuevo, compilamos
    needs_compile = not os.path.exists(dll_path)
    if not needs_compile:
        needs_compile = os.path.getmtime(cpp_source) > os.path.getmtime(dll_path)
        
    if needs_compile:
        print(f"Compilando kernel nativo FWHT con g++...")
        try:
            # Comando optimizado para el Ryzen 7 (AVX2 + OpenMP)
            cmd = [
                "g++", "-O3", "-shared", "-fopenmp", "-mavx2",
                cpp_source, "-o", dll_path
            ]
            # En Windows MinGW a veces necesita -static-libgcc -static-libstdc++
            if os.name == 'nt':
                cmd += ["-static-libgcc", "-static-libstdc++"]
                
            subprocess.run(cmd, check=True, capture_output=True)
            print("¡Compilación nativa exitosa!")
        except Exception as e:
            print(f"Error compilando con g++: {e}")
            return None
            
    try:
        # Cargamos la librería
        lib = ctypes.CDLL(dll_path)
        # Definimos tipos de argumentos: float* data, int batch_size, int n
        lib.fwht_float.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        lib.fwht_float.restype = None
        _native_lib = lib
        return lib
    except Exception as e:
        print(f"Error cargando librería nativa {dll_name}: {e}")
        _failed_to_load = True # No reintentar cada vez
        return None

def fwht_native(x):
    """
    Interface para el kernel nativo de FWHT vía ctypes.
    Maneja el casteo a float y la vuelta a half si es necesario.
    """
    lib = compile_and_load()
    if lib is None:
        return None
    
    # El kernel espera float32 (float en C++)
    orig_dtype = x.dtype
    if orig_dtype != torch.float32:
        x_float = x.float()
    else:
        x_float = x
        
    orig_shape = x_float.shape
    n = orig_shape[-1]
    # Asegurar que los datos están contiguos en memoria
    x_flat = x_float.reshape(-1, n).contiguous()
    batch_size = x_flat.size(0)
    
    # Obtener puntero a los datos (raw pointer)
    ptr = x_flat.data_ptr()
    float_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
    
    # Ejecutar kernel nativo
    lib.fwht_float(float_ptr, batch_size, n)
    
    # Volver a la forma y tipo original
    res = x_flat.view(orig_shape)
    if orig_dtype != torch.float32:
        res = res.to(orig_dtype)
        
    return res
