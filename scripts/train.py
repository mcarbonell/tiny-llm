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
import argparse

# Agregar ruta base para resolver el import del modelo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs as DenseArgs
from model.model_moe import TinyThinkerMoE, ModelArgs as MoEArgs
from model.model_coga import TinyThinkerCOGA, ModelArgs as CogaArgs

# ----------------------------------
# Configuración por Defecto
# ----------------------------------
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEQ_LEN = 1024
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_ITERS = 10000
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_WARMUP = 200
DEFAULT_EVAL_INTERVAL = 250
DEFAULT_EVAL_ITERS = 20
DEFAULT_DATA_PATH = "data/train_combined.bin"

import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="TinyThinker Pretrain — Versión Optimizada")
    parser.add_argument('--config', type=str, default=None, help='Ruta al config YAML. Sobreescribe otros argumentos.')
    parser.add_argument('--arch', type=str, default='dense', choices=['dense', 'moe', 'coga'], help='Arquitectura a entrenar.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'dml', 'mps'], help='Dispositivo de entrenamiento.')
    parser.add_argument('--resume', action='store_true', help='Reanudar desde el último checkpoint.')
    parser.add_argument('--max_iters', type=int, default=DEFAULT_MAX_ITERS, help='Número total de iteraciones.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Tamaño de batch por micro-paso.')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN, help='Longitud de secuencia (ventana de contexto).')
    parser.add_argument('--grad_accum_steps', type=int, default=DEFAULT_GRAD_ACCUM, help='Pasos de acumulación de gradientes.')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate máximo.')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Activar ahorro de RAM.')
    return parser.parse_args()

def load_config(args):
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Fusionar yaml en args
    for k, v in config.items():
        setattr(args, k, v)
        
    return args

def main():
    args_cli = parse_args()
    args_cli = load_config(args_cli)
    
    if hasattr(args_cli, 'learning_rate'): 
        args_cli.lr = args_cli.learning_rate
    
    # ----------------------------------
    # 1. Configuración de Hardware
    # ----------------------------------
    device_name = args_cli.device
    device = 'cpu'
    
    if device_name == 'dml':
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"[Hardware] Usando DirectML (GPU AMD)")
        except ImportError:
            print("[Warning] DirectML no instalado, usando CPU.")
    elif device_name == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif device_name == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    
    # Setup de Precisión Mixta (AMP)
    _is_dml = str(device).startswith('dml') or 'privateuseone' in str(device)
    if _is_dml:
        ctx = contextlib.nullcontext()
        ptdtype = torch.float32
    elif device == 'cuda':
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    elif device == 'cpu':
        ptdtype = torch.bfloat16  # AVX-512 nativo en Zen 4
        ctx = torch.amp.autocast(device_type='cpu', dtype=ptdtype)
    else:
        ctx = contextlib.nullcontext()
        ptdtype = torch.float32

    scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16 and device == 'cuda'))

    # ----------------------------------
    # 2. Carga de Datos (Memmap)
    # ----------------------------------
    data_path = getattr(args_cli, 'data_path', DEFAULT_DATA_PATH)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Falta el dataset: {data_path}")
    
    full_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    val_fraction = 0.05
    val_start = int(len(full_data) * (1.0 - val_fraction))
    train_data = full_data[:val_start]
    val_data   = full_data[val_start:]

    def get_batch(split='train'):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - args_cli.seq_len, (args_cli.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+args_cli.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args_cli.seq_len]).astype(np.int64)) for i in ix])
        if device == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # ----------------------------------
    # 3. Inicialización del Modelo
    # ----------------------------------
    out_dir = getattr(args_cli, 'checkpoint_dir', "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    
    arch = getattr(args_cli, 'arch', 'dense')
    
    common_args = {
        'dim': getattr(args_cli, 'dim', 256),
        'n_layers': getattr(args_cli, 'n_layers', 6),
        'n_heads': getattr(args_cli, 'n_heads', 8),
        'n_kv_heads': getattr(args_cli, 'n_kv_heads', 4),
        'vocab_size': getattr(args_cli, 'vocab_size', 16384),
        'max_seq_len': getattr(args_cli, 'max_seq_len', getattr(args_cli, 'seq_len', 1024))
    }
    
    if arch == 'dense':
        model_args = DenseArgs(**common_args)
        model = TinyThinker(model_args)
    elif arch == 'moe':
        moe_args = common_args.copy()
        moe_args['n_experts'] = getattr(args_cli, 'n_experts', 8)
        moe_args['top_k'] = getattr(args_cli, 'top_k', 2)
        moe_args['n_reserved'] = getattr(args_cli, 'n_reserved', 4)
        model_args = MoEArgs(**moe_args)
        model = TinyThinkerMoE(model_args)
    elif arch == 'coga':
        coga_args = common_args.copy()
        coga_args.pop('n_layers', None) # COGA usa n_pre, n_core, n_post
        coga_args['n_pre_layers'] = getattr(args_cli, 'n_pre_layers', 1)
        coga_args['n_core_layers'] = getattr(args_cli, 'n_core_layers', 2)
        coga_args['n_post_layers'] = getattr(args_cli, 'n_post_layers', 1)
        coga_args['max_recurrence_steps'] = getattr(args_cli, 'max_recurrence_steps', 4)
        coga_args['n_scratch_slots'] = getattr(args_cli, 'n_scratch_slots', 32)
        coga_args['n_experts'] = getattr(args_cli, 'n_experts', 8)
        coga_args['top_k'] = getattr(args_cli, 'top_k', 2)
        coga_args['n_reserved'] = getattr(args_cli, 'n_reserved', 4)
        model_args = CogaArgs(**coga_args)
        model = TinyThinkerCOGA(model_args)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
        
    model.to(device)
    
    if args_cli.use_gradient_checkpointing:
        if arch == 'coga':
            for layer in model.pre_layers: layer.use_checkpoint = True
            for layer in model.core_layers: layer.use_checkpoint = True
            for layer in model.post_layers: layer.use_checkpoint = True
        else:
            for layer in model.layers: layer.use_checkpoint = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=1e-1)
    
    iter_num = 0
    best_val_loss = 1e9

    # ----------------------------------
    # 4. Lógica de Reanudación
    # ----------------------------------
    if args_cli.resume:
        ckpt_path = os.path.join(out_dir, 'ckpt_pretrain_best.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, 'ckpt_pretrain_latest.pt')
        
        if os.path.exists(ckpt_path):
            print(f"[Resume] Cargando progreso desde {ckpt_path}...")
            # En PyTorch 2.6 we need to allowlist our custom classes
            torch.serialization.add_safe_globals([DenseArgs, MoEArgs, CogaArgs])
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint.get('val_loss', 1e9)
            print(f"[Resume] Continuando desde la iteración {iter_num} (Pérdida previa: {best_val_loss:.4f})")
        else:
            print("[Resume] No se encontró ningún checkpoint. Empezando de cero.")

    # ----------------------------------
    # 5. Funciones Auxiliares
    # ----------------------------------
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(DEFAULT_EVAL_ITERS)
            for k in range(DEFAULT_EVAL_ITERS):
                X, Y = get_batch(split)
                with ctx:
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def get_lr(it):
        if it < DEFAULT_WARMUP:
            return args_cli.lr * it / DEFAULT_WARMUP
        if it > args_cli.max_iters:
            return DEFAULT_MIN_LR
        decay_ratio = (it - DEFAULT_WARMUP) / (args_cli.max_iters - DEFAULT_WARMUP)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return DEFAULT_MIN_LR + coeff * (args_cli.lr - DEFAULT_MIN_LR)

    # Setup de Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    start_date = datetime.datetime.now()
    log_file = os.path.join(log_dir, f"train_{start_date.strftime('%Y%m%d_%H%M%S')}.log")
    
    global_start_time = time.time()
    def t_print(msg):
        elapsed = time.time() - global_start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        full_msg = f"[{elapsed_str}] {msg}"
        print(full_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")

    total_params = sum(p.numel() for p in model.parameters())
    header = f"""========================================
DATE: {start_date.strftime('%Y-%m-%d %H:%M:%S')}
DEVICE: {str(device).upper()}
CPU THREADS: {torch.get_num_threads()}
--------------- HYPERPARAMS -----------
batch_size: {args_cli.batch_size}
seq_len: {args_cli.seq_len}
grad_accum_steps: {args_cli.grad_accum_steps}
max_iters: {args_cli.max_iters}
learning_rate: {args_cli.lr}
--------------- MODEL PARAMS ----------
dim: {model_args.dim}
n_layers: {model_args.n_layers}
n_heads: {model_args.n_heads}
vocab_size: {model_args.vocab_size}
TOTAL PARAMS: {total_params / 1e6:.2f}M
========================================"""
    t_print(header)

    # ----------------------------------
    # 6. Bucle Principal
    # ----------------------------------
    t_print(f"Entrenamiento activo en {str(device).upper()} | Iteraciones: {iter_num}/{args_cli.max_iters}")
    t0 = time.time()

    while iter_num < args_cli.max_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        # Evaluación
        if iter_num % DEFAULT_EVAL_INTERVAL == 0 and iter_num > 0:
            losses = estimate_loss()
            t_print(f"Iter {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'args': model_args,
                'arch': arch,
                'val_loss': losses['val']
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_pretrain_latest.pt'))
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_pretrain_best.pt'))
                t_print(f" -> Nuevo mejor modelo (val_loss: {best_val_loss:.4f})")

        # Paso de entrenamiento
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args_cli.grad_accum_steps):
            X, Y = get_batch()
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                loss = loss / args_cli.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled(): scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Logging de velocidad
        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Multiplicamos por accum para ver loss real
            loss_val = loss.item() * args_cli.grad_accum_steps
            t_print(f"iter {iter_num:5d} | loss {loss_val:.4f} | lr {lr:.2e} | time {dt:.2f}s")

        iter_num += 1

    t_print("Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    main()
