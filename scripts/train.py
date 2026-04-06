import os
import math
import time
import datetime
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
import sys
import logging

# Agregar ruta base para resolver el import del modelo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs

# ----------------------------------
# Hiperparámetros de Entrenamiento
# ----------------------------------
batch_size = 16
seq_len = 256             # Contexto corto para pre-entrenamiento rápido
grad_accum_steps = 4      # Simular batch size real mayor: 16 * 4 = 64
max_iters = 5000          # Bucle global
learning_rate = 1e-3      # Para modelos mini solemos usar LRs algo más altos
min_lr = 1e-5
warmup_iters = 200
eval_interval = 250
eval_iters = 20

# [FIX BUG-C] out_dir definido en scope global ANTES del logging.basicConfig().
# Antes, el FileHandler fallaba con NameError porque out_dir no existía aún.
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(out_dir, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(out_dir, 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------
# Configuración Inicial y Hardware
# ----------------------------------
device = 'cpu'
try:
    import torch_directml
    device = torch_directml.device()   # Equivalente a 'dml:0'
    print(f"[device] DirectML activo: {device}")
except ImportError:
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    print(f"[device] DirectML no disponible, usando: {device}")

# AMP (Precisión Mixta Automática):
# BF16 en CPU (AMD 8845HS lo soporta nativamente) da un ~1.5-2x de speedup.
# En CUDA usamos bf16 si está soportado, si no fp32. GradScaler solo aplica para fp16.
if device == 'cuda':
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif device == 'cpu':
    ptdtype = torch.bfloat16  # AVX-512 BF16 nativo en Zen 4
else:
    ptdtype = torch.float32

ctx = torch.amp.autocast(device_type=device if device != 'mps' else 'cpu', dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16 and device == 'cuda'))

# Fracción del dataset reservada para validación (nunca vista en entrenamiento)
val_fraction = 0.05

# ----------------------------------
# Cargador de Datos (Memmap)
# ----------------------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"¡Oops! Falta {data_path}. Corre scripts/prepare_data.py")

full_data = np.memmap(data_path, dtype=np.uint16, mode='r')

# Split estático train / val (el val es el último 5% del array, nunca toca train)
_val_start = int(len(full_data) * (1.0 - val_fraction))
train_data = full_data[:_val_start]
val_data   = full_data[_val_start:]

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    # Toma de muestra aleatoria continua en el array uint16
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """Estima la loss en ambos splits (train y val) para ver si hay overfitting."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    # Linear Warmup + Cosine Decay
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="TinyThinker Pretrain", add_help=False)
    _parser.add_argument('--use_gradient_checkpointing', action='store_true', default=False,
                         help='Activa gradient checkpointing para reducir uso de VRAM/RAM.')
    _cli, _ = _parser.parse_known_args()

    # Setup de logs
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    start_date = datetime.datetime.now()
    run_id = start_date.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"train_{run_id}.log")

    global_start_time = time.time()
    def t_print(msg):
        elapsed = time.time() - global_start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        lines = str(msg).split('\n')
        for line in lines:
            full_msg = f"[{elapsed_str}] {line}"
            logger.info(full_msg)
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")

    t_print(f"👉 Tokens mapeados en RAM (Memmap): {len(train_data) / 1e6:.2f} Millones")

    args = ModelArgs(
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=16384,
        max_seq_len=seq_len
    )
    model = TinyThinker(args)
    model.to(device)

    # Activar gradient checkpointing en cada capa si se pide via CLI
    if _cli.use_gradient_checkpointing:
        for layer in model.layers:
            layer.use_checkpoint = True
        t_print("✅ Gradient checkpointing ACTIVADO (menor RAM, mayor tiempo de backward)")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    header = f"""========================================
DATE: {start_date.strftime('%Y-%m-%d %H:%M:%S')}
DEVICE: {device.upper()}
--------------- HYPERPARAMS -----------
batch_size: {batch_size}
seq_len: {seq_len}
grad_accum_steps: {grad_accum_steps}
max_iters: {max_iters}
learning_rate: {learning_rate}
--------------- MODEL PARAMS ----------
dim: {args.dim}
n_layers: {args.n_layers}
n_heads: {args.n_heads}
vocab_size: {args.vocab_size}
TOTAL PARAMS: {total_params / 1e6:.2f}M
========================================"""
    t_print(header)
    t_print(f"🔥 TinyThinker inicializado en {device.upper()} | Parámetros Totales: {total_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)
    
    t_print("🚀 Arrancando Bucle de Entrenamiento Autorregresivo...")
    t0 = time.time()
    
    for iter_num in range(max_iters):
        # 1. Update Learning Rate (Cosine decay)
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 2. Validación Periódica y Logging
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model)
            t_print(f"✨ [ITER {iter_num}] train_loss: {losses['train']:.4f} | val_loss: {losses['val']:.4f} | Guardando Checkpoint...")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'args': args,
                'val_loss': losses['val']
            }
            # Checkpoint rotativo con número de iteración
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num:05d}.pt'))
            # Checkpoint del mejor modelo según val_loss
            if not hasattr(estimate_loss, '_best_val') or losses['val'] < estimate_loss._best_val:
                estimate_loss._best_val = losses['val']
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_best.pt'))
                t_print(f"🏆 Nuevo mejor modelo guardado (val_loss={losses['val']:.4f})")

        # 3. Micro-Steps de Entrenamiento (Acumulación de Gradientes)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            X, Y = get_batch()
            
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            
            # Repartimos matemáticamente la contribución para normalizar
            loss = loss / grad_accum_steps
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
        # 4. Clipping para evitar explosión
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
        # 5. Volcado al modelo
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        # Trazas básicas en terminal
        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Multiplicamos por grad_accum_steps para ver el loss real del batch
            loss_unscaled = loss.item() * grad_accum_steps
            t_print(f"iter {iter_num:4d} | loss {loss_unscaled:.4f} | lr {lr:.2e} | loop {dt*1000:.2f}ms")

if __name__ == "__main__":
    main()
ter {iter_num:4d} | loss {loss_unscaled:.4f} | lr {lr:.2e} | loop {dt*1000:.2f}ms")

if __name__ == "__main__":
    main()
