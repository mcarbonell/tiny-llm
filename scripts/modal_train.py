import modal
import os
import sys

# 1. Imagen optimizada para entrenamiento con GPU
# Usamos Debian con Python 3.11 y las dependencias base de torch
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "PyYAML>=6.0.2",
        "tqdm>=4.66.0"
    )
)

app = modal.App("tinythinker-remote")

# 2. Volumen persistente para datos y checkpoints
# Este volumen guarda los archivos .bin y los modelos .pt entre ejecuciones
volume = modal.Volume.from_name("tinythinker-storage", create_if_missing=True)

# 3. Función de entrenamiento remoto
@app.function(
    image=image,
    gpu="a10g",             # Nvidia A10G (24GB VRAM) - Balance ideal rendimiento/coste
    volumes={"/vol": volume},
    timeout=3600 * 4,       # Timeout de 4 horas
    mounts=[
        # Montamos las carpetas locales necesarias en el contenedor remoto
        modal.Mount.from_local_dir("model", remote_path="/root/model"),
        modal.Mount.from_local_dir("scripts", remote_path="/root/scripts"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ]
)
def train():
    # Aseguramos que el código en /root sea importable
    sys.path.append("/root")
    
    import torch
    import numpy as np
    from scripts.train import main as run_train
    from unittest.mock import patch

    # Definimos las rutas dentro del contenedor de Modal
    # /vol es el volumen persistente, /root es el código montado
    DATA_VOL_PATH = "/vol/data/train_combined.bin"
    LOCAL_DATA_DIR = "/root/data"
    LOCAL_DATA_FILE = os.path.join(LOCAL_DATA_DIR, "train_combined.bin")
    
    # Preparamos el entorno para que train.py encuentre lo que espera
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs("/root/checkpoints", exist_ok=True)
    os.makedirs("/root/logs", exist_ok=True)

    # Si el dataset está en el volumen, creamos un enlace simbólico en /root/data/
    if os.path.exists(DATA_VOL_PATH):
        print(f"✅ Dataset encontrado en volumen: {DATA_VOL_PATH}")
        if os.path.exists(LOCAL_DATA_FILE):
             os.remove(LOCAL_DATA_FILE)
        os.link(DATA_VOL_PATH, LOCAL_DATA_FILE)
    else:
        print(f"❌ ERROR: No se encontró el dataset en {DATA_VOL_PATH}")
        print("Sube el archivo primero con: modal volume put tinythinker-storage data/train_combined.bin data/train_combined.bin")
        return

    # Inyeccion de argumentos para train.py
    # Usamos batch_size 32 porque la A10G tiene memoria de sobra para este modelo
    test_args = [
        "scripts/train.py",
        "--device", "cuda",
        "--batch_size", "32",
        "--seq_len", "1024",
        "--max_iters", "2000"
    ]

    print("--- INICIANDO ENTRENAMIENTO REMOTO ---")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with patch("sys.argv", test_args):
        try:
            run_train()
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            raise e
        finally:
            # Intentamos salvar los checkpoints del disco local al Volumen persistente
            print("--- SINCRONIZANDO CHECKPOINTS AL VOLUMEN ---")
            CKPT_DIR = "/root/checkpoints"
            VOL_CKPT_DIR = "/vol/checkpoints"
            os.makedirs(VOL_CKPT_DIR, exist_ok=True)
            
            for f in os.listdir(CKPT_DIR):
                shutil_copy_path = os.path.join(CKPT_DIR, f)
                target_path = os.path.join(VOL_CKPT_DIR, f)
                import shutil
                shutil.copy(shutil_copy_path, target_path)
                print(f"Backup: {f} -> /vol/checkpoints/")
            
            # Commit final de los cambios en el volumen
            volume.commit()

@app.local_entrypoint()
def main():
    train.remote()
