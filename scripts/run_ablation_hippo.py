"""
run_ablation_hippo.py - Lanza SECUENCIALMENTE los dos ablations del hippocampus
(on / off) con configs idénticas salvo el flag. No corre ambos a la vez para no
saturar la maquina. Escribe a logs/ablation_hippo_on.log y _off.log.
"""
import subprocess, sys, os, time

VENV = r".\venv_gpu\Scripts\python.exe"
CONFIGS = [
    ("configs/train_ablation_hippo_on.yaml",  "logs/ablation_hippo_on.log"),
    ("configs/train_ablation_hippo_off.yaml", "logs/ablation_hippo_off.log"),
]
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

for cfg, logpath in CONFIGS:
    print(f"[{time.strftime('%H:%M:%S')}] >>> Iniciando ablation: {cfg}", flush=True)
    with open(logpath, "w") as logf:
        p = subprocess.run(
            [VENV, "scripts/train.py", "--config", cfg, "--arch", "unified"],
            stdout=logf, stderr=subprocess.STDOUT,
        )
    print(f"[{time.strftime('%H:%M:%S')}] <<< Termino {cfg} (rc={p.returncode})", flush=True)
print("ABLATION COMPLETO", flush=True)
