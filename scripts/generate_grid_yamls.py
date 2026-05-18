import os
import yaml

def generate_grid():
    os.makedirs("configs/grid_search", exist_ok=True)
    
    # Base configuration template
    base_config = {
        "arch": "spectral_v10",
        "data_path": "data/train_v2_32k.bin",
        "tokenizer_path": "model/tokenizer_v2_32k.json",
        "batch_size": 16,
        "seq_len": 1024,
        "max_iters": 2000,     # Corto para grid search rápido
        "warmup_iters": 200,
        "learning_rate": 0.03,
        "min_lr": 0.03,        # <-- FUNDAMENTAL: LR Constante para comparar caídas puras
        "n_layers": 6,
        "vocab_size": 32768,
        "chunk_size": 256,
        "k_mem": 32,
        "gamma": 0.9,
        "lambda_phase": 0.01,
        "optimizer": "adamw",
    }
    
    # Grid 1: Variar 'dim' (Capacidad Semántica) con k_walsh fijo
    dims_to_test = [128, 256]
    for d in dims_to_test:
        config = base_config.copy()
        config["dim"] = d
        config["k_walsh"] = 64
        config["checkpoint_dir"] = f"checkpoints/grid_v10/dim{d}_k64"
        
        with open(f"configs/grid_search/v10_dim{d}_k64.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)

    # Grid 2: Variar 'k_walsh' (Ancho de Banda Lógico) con dim fijo
    k_to_test = [32, 128]
    for k in k_to_test:
        config = base_config.copy()
        config["dim"] = 256
        config["k_walsh"] = k
        config["checkpoint_dir"] = f"checkpoints/grid_v10/dim256_k{k}"
        
        with open(f"configs/grid_search/v10_dim256_k{k}.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)

    print("✅ Generados 4 YAMLs para Grid Search en 'configs/grid_search/'")
    print("Para ejecutarlos uno tras otro, podrías usar un pequeño script .bat o hacerlo a mano:")
    print("python scripts/train.py --config configs/grid_search/v10_dim128_k64.yaml")
    print("python scripts/train.py --config configs/grid_search/v10_dim256_k64.yaml")
    print("python scripts/train.py --config configs/grid_search/v10_dim256_k32.yaml")
    print("python scripts/train.py --config configs/grid_search/v10_dim256_k128.yaml")

if __name__ == "__main__":
    generate_grid()
