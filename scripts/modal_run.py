import modal
import os
import sys

# 1. Definir el entorno contenedor (Imagen de Docker)
local_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "tokenizers", "numpy", "pyyaml")
    .add_local_dir(
        local_project_dir, 
        remote_path="/workspace",
        ignore=[".git", "checkpoints", "logs", "__pycache__"]
    )
)

# 2. Definir la App
app = modal.App("tinythinker-cloud")

# 4. Definir la función que correrá en la GPU de la nube
@app.function(
    image=image,
    gpu="A10G",  # Puedes cambiar a "A100", "H100", "L4", "T4"
    timeout=3600
)
def test_gpu_connection():
    import torch
    
    print("\n" + "="*50)
    print("☁️  MODAL CLOUD ENVIRONMENT DETECTED ☁️")
    print("="*50)
    
    # Verificar hardware CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Asignada: {gpu_name}")
        print(f"✅ VRAM Total:   {vram_gb:.2f} GB")
    else:
        print("❌ No se detectó GPU CUDA.")
        return

    # Probar que las importaciones y la arquitectura funcionan en la nube
    os.chdir("/workspace")
    sys.path.append("/workspace")
    
    try:
        from model.model_spectral_v7 import SpectralArgs, SpectralThinker
        print("✅ Importación de librerías locales exitosa.")
        
        # Smoke test: Crear un modelo pequeño y mandarlo a la VRAM
        args = SpectralArgs(dim=512, vocab_size=32768, n_layers=2)
        model = SpectralThinker(args).to('cuda')
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Modelo Spectral instanciado en VRAM (Params físicos: {params/1e6:.2f}M)")
        
    except Exception as e:
        print(f"❌ Error al importar o instanciar el modelo: {e}")
        
    print("="*50)
    print("🚀 CONEXIÓN ESTABLECIDA. EL SISTEMA ESTÁ LISTO PARA ENTRENAR.")
    print("="*50 + "\n")

@app.local_entrypoint()
def main():
    print("Iniciando empaquetado y conexión con Modal...")
    # Llamamos a la función remota
    test_gpu_connection.remote()
