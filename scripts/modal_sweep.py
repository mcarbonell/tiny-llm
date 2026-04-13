import modal
import os
import sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "PyYAML>=6.0.2",
        "tqdm>=4.66.0"
    )
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("tinythinker-sweep")

volume = modal.Volume.from_name("tinythinker-storage", create_if_missing=True)

@app.function(
    image=image,
    gpu="l4",               # Nvidia L4 (24GB VRAM)
    volumes={"/vol": volume},
    timeout=3600 * 2,       # Timeout de 2 horas (suficiente para 2000 iters del 10M)
)
def train_sweep(config_path: str):
    sys.path.append("/root")
    
    import torch
    import shutil
    from scripts.train import main as run_train
    from unittest.mock import patch

    DATA_VOL_PATH = "/vol/data/train_v1.bin"
    LOCAL_DATA_DIR = "/root/data"
    LOCAL_DATA_FILE = os.path.join(LOCAL_DATA_DIR, "train_v1.bin")
    
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs("/root/logs", exist_ok=True)

    if os.path.exists(DATA_VOL_PATH):
        if os.path.exists(LOCAL_DATA_FILE):
             os.remove(LOCAL_DATA_FILE)
        os.symlink(DATA_VOL_PATH, LOCAL_DATA_FILE)
    else:
        print(f"❌ ERROR: No se encontró el dataset en {DATA_VOL_PATH}")
        return

    config_name = os.path.basename(config_path).replace(".yaml", "")
    sweep_ckpt_dir = f"/root/checkpoints/{config_name}"
    os.makedirs(sweep_ckpt_dir, exist_ok=True)

    test_args = [
        "scripts/train.py",
        "--config", config_path,
        "--device", "cuda",
    ]

    print(f"--- INICIANDO SWEEP: {config_name} ---")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with patch("sys.argv", test_args):
        try:
            run_train()
        except Exception as e:
            print(f"❌ Error durante el entrenamiento {config_name}: {e}")
            raise e
        finally:
            print(f"--- SINCRONIZANDO CHECKPOINTS DE {config_name} AL VOLUMEN ---")
            VOL_CKPT_DIR = f"/vol/checkpoints/{config_name}"
            os.makedirs(VOL_CKPT_DIR, exist_ok=True)
            
            for f in os.listdir(sweep_ckpt_dir):
                shutil_copy_path = os.path.join(sweep_ckpt_dir, f)
                target_path = os.path.join(VOL_CKPT_DIR, f)
                shutil.copy(shutil_copy_path, target_path)
            
            volume.commit()

@app.local_entrypoint()
def main():
    configs = [
        "/root/configs/sweep_10M_lr3e4_wu500.yaml",
        "/root/configs/sweep_10M_lr6e4_wu500.yaml",
        "/root/configs/sweep_10M_lr1e3_wu1000.yaml",
        "/root/configs/sweep_10M_lr5e4_wu200.yaml"
    ]
    print(f"Lanzando {len(configs)} entrenamientos concurrentes (GPU L4) en Modal...")
    list(train_sweep.map(configs))
    print("Sweep completado y datos sincronizados en el volumen 'tinythinker-storage'.")
