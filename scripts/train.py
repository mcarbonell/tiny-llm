import os
import math
import time
import datetime
import contextlib
import gc
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
from model.model_spectral import SpectralThinker, SpectralArgs
from model.model_spectral_v4 import SpectralThinker as SpectralThinkerV4, SpectralArgs as SpectralArgsV4
from model.model_spectral_v5 import SpectralThinker as SpectralThinkerV5, SpectralArgs as SpectralArgsV5
from model.model_spectral_v6 import SpectralThinker as SpectralThinkerV6, SpectralArgs as SpectralArgsV6
from model.model_spectral_v7 import SpectralThinker as SpectralThinkerV7, SpectralArgs as SpectralArgsV7
from model.model_spectral_v8 import SpectralThinkerV8, SpectralArgs as SpectralArgsV8
from model.model_spectral_v8_1 import SpectralThinkerV8_1, SpectralArgs as SpectralArgsV8_1
from model.model_spectral_v8_4_optimized import SpectralThinkerV8_4, SpectralArgs as SpectralArgsV8_4
from model.model_spectral_v8_5_native import SpectralThinkerV8_5, SpectralArgs as SpectralArgsV8_5
from model.model_spectral_v8_6_universal import SpectralThinkerV8_6, SpectralArgs as SpectralArgsV8_6
from model.model_spectral_v8_6b_gated import SpectralThinkerV8_6b
from model.model_coga_spectral import TinyThinkerCogaSpectral, CogaSpectralArgs
from model.model_analog import TinyThinkerAnalog, AnalogArgs
from model.model_auto_architect import TinyThinkerAutoArchitect, AutoArchitectArgs
from model.model_auto_analog import TinyThinkerAutoAnalog, AutoAnalogArgs
from optim_supermario import SuperMarioOptimizer

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
    parser.add_argument('--arch', type=str, default='dense', choices=['dense', 'moe', 'coga', 'spectral', 'spectral_v4', 'spectral_v5', 'spectral_v6', 'spectral_v7', 'spectral_v8', 'spectral_v8_1', 'spectral_v8_4', 'spectral_v8_5', 'spectral_v8_6', 'spectral_v8_6b', 'coga_spectral', 'analog', 'auto_architect', 'auto_analog'], help='Arquitectura a entrenar.')
    parser.add_argument('--phase1', action='store_true', help='Blueprint Fase 1: Solo entrena gates (requiere arch gated).')
    parser.add_argument('--phase2', action='store_true', help='Blueprint Fase 2: Entrena pesos capa por capa (Round-Robin).')
    parser.add_argument('--rotation_iters', type=int, default=100, help='Iteraciones por cada capa en Fase 2.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'swo'], help='Optimizador a utilizar.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'dml', 'mps'], help='Dispositivo de entrenamiento.')
    parser.add_argument('--resume', action='store_true', help='Reanudar desde el último checkpoint.')
    parser.add_argument('--max_iters', type=int, default=DEFAULT_MAX_ITERS, help='Número total de iteraciones.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Tamaño de batch por micro-paso.')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN, help='Longitud de secuencia (ventana de contexto).')
    parser.add_argument('--grad_accum_steps', type=int, default=DEFAULT_GRAD_ACCUM, help='Pasos de acumulación de gradientes.')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate máximo.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (por defecto 0.0).')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Activar ahorro de RAM.')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Ruta al dataset (.bin).')
    parser.add_argument('--tokenizer_path', type=str, default='model/tokenizer.json', help='Ruta al tokenizador (.json).')
    return parser.parse_args()

def load_config(args):
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Extraer argumentos pasados explícitamente por CLI para que tengan prioridad sobre YAML
    cli_args = [arg.lstrip('-').split('=')[0].replace('-', '_') for arg in sys.argv if arg.startswith('-')]
    
    # Fusionar yaml en args
    for k, v in config.items():
        if k not in cli_args:
            setattr(args, k, v)
        
    return args

class DMLAdamW(torch.optim.Optimizer):
    """
    Optimizador AdamW personalizado para DirectML.
    Evita la operación 'aten::lerp.Scalar_out' que causa fallbacks masivos a CPU y NaNs en AMD iGPUs.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay (AdamW)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Momentums sin usar lerp_
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)

def main():
    args_cli = parse_args()
    args_cli = load_config(args_cli)
    
    # Prioridad: CLI --lr > Config YAML learning_rate
    # Solo aplicamos el del config si el usuario no pasó uno por CLI o si es el default
    if hasattr(args_cli, 'learning_rate') and 'lr' not in sys.argv:
        args_cli.lr = args_cli.learning_rate
    
    # ----------------------------------
    # Setup de Logs y Función t_print (disponible temprano)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    start_date = datetime.datetime.now()
    log_file = os.path.join(log_dir, f"train_{start_date.strftime('%Y%m%d_%H%M%S')}.log")
    
    global_start_time = time.time()
    def t_print(msg):
        elapsed = time.time() - global_start_time
        days = int(elapsed // 86400)
        hours = int((elapsed % 86400) // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if days == 0 else f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        full_msg = f"[{elapsed_str}] {msg}"
        # Safe print para evitar UnicodeDecodeError en terminales Windows
        try:
            print(full_msg, flush=True)
        except UnicodeEncodeError:
            print(full_msg.encode('ascii', errors='replace').decode('ascii'), flush=True)
            
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except:
            pass

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
        # DirectML todavía falla con varias ops básicas bajo autocast
        # (pow, to, mul, linear). Priorizamos compatibilidad.
        ptdtype = torch.float32
        ctx = contextlib.nullcontext()
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        print("[Hardware] DirectML detectado: AMP/FP16 desactivado por compatibilidad")
    elif device == 'cuda':
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))
    elif device == 'cpu':
        ptdtype = torch.bfloat16  # AVX-512 nativo en Zen 4
        ctx = torch.amp.autocast(device_type='cpu', dtype=ptdtype)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    else:
        ctx = contextlib.nullcontext()
        ptdtype = torch.float32
        scaler = torch.amp.GradScaler('cpu', enabled=False)

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
        
        # Volvemos a un método más seguro pero optimizado
        x = torch.stack([torch.from_numpy(data[i:i+args_cli.seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+args_cli.seq_len].astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)

    # ----------------------------------
    # 3. Inicialización del Modelo
    # ----------------------------------
    arch = getattr(args_cli, 'arch', 'dense')
    
    # SEGURIDAD: Si no hay directorio específico, usamos uno basado en la arquitectura
    out_dir = getattr(args_cli, 'checkpoint_dir', os.path.join("checkpoints", arch))
    os.makedirs(out_dir, exist_ok=True)
    
    # PROTECCIÓN CONTRA SOBREESCRITURA
    if not args_cli.resume:
        latest_path = os.path.join(out_dir, 'ckpt_pretrain_latest.pt')
        best_path = os.path.join(out_dir, 'ckpt_pretrain_best.pt')
        if os.path.exists(latest_path) or os.path.exists(best_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n[!] AVISO: Se detectaron checkpoints existentes en {out_dir}")
            print(f"    Si querías continuar el entrenamiento, usa el flag --resume")
            print(f"    Para proteger tus datos, se renombrarán los archivos antiguos con el sufijo _{timestamp}")
            
            for path in [latest_path, best_path]:
                if os.path.exists(path):
                    new_path = path.replace('.pt', f'_{timestamp}.pt')
                    os.rename(path, new_path)
                    print(f"    Renombrado: {os.path.basename(path)} -> {os.path.basename(new_path)}")
            print("")
    
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
    elif arch == 'spectral':
        spectral_args = common_args.copy()
        spectral_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        spectral_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        spectral_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        model_args = SpectralArgs(**spectral_args)
        model = SpectralThinker(model_args)
    elif arch == 'spectral_v4':
        spectral_args = common_args.copy()
        spectral_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        spectral_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        spectral_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        model_args = SpectralArgsV4(**spectral_args)
        model = SpectralThinkerV4(model_args)
    elif arch == 'spectral_v5':
        spectral_args = common_args.copy()
        spectral_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        spectral_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        spectral_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        spectral_args['k_seq_len']    = getattr(args_cli, 'k_seq_len',    64)
        model_args = SpectralArgsV5(**spectral_args)
        model = SpectralThinkerV5(model_args)
    elif arch == 'spectral_v6':
        spectral_args = common_args.copy()
        spectral_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        spectral_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        spectral_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        spectral_args['k_seq_len']    = getattr(args_cli, 'k_seq_len',    64)
        model_args = SpectralArgsV6(**spectral_args)
        model = SpectralThinkerV6(model_args)
    elif arch == 'spectral_v7':
        spectral_args = common_args.copy()
        spectral_args['emb_dim']      = getattr(args_cli, 'emb_dim',     128)
        spectral_args['k_vocab']      = getattr(args_cli, 'k_vocab',     128)
        spectral_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   128)
        spectral_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    128)
        spectral_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 256)
        spectral_args['k_seq_len']    = getattr(args_cli, 'k_seq_len',    64)
        model_args = SpectralArgsV7(**spectral_args)
        model = SpectralThinkerV7(model_args)
    elif arch == 'spectral_v8':
        spectral_args = common_args.copy()
        spectral_args['emb_dim']      = getattr(args_cli, 'emb_dim',     128)
        spectral_args['num_experts']  = getattr(args_cli, 'num_experts', 131072)
        spectral_args['top_k']        = getattr(args_cli, 'top_k',       16)
        model_args = SpectralArgsV8(**spectral_args)
        model = SpectralThinkerV8(model_args)
    elif arch == 'spectral_v8_1':
        spectral_args = common_args.copy()
        spectral_args['emb_dim']      = getattr(args_cli, 'emb_dim',     128)
        spectral_args['num_experts']  = getattr(args_cli, 'num_experts', 131072)
        spectral_args['top_k']        = getattr(args_cli, 'top_k',       16)
        spectral_args['k_dim']        = getattr(args_cli, 'k_dim',       128)
        model_args = SpectralArgsV8_1(**spectral_args)
        model = SpectralThinkerV8_1(model_args)
    elif arch in ('spectral_v8_4', 'spectral_v8_5', 'spectral_v8_6', 'spectral_v8_6b'):
        spectral_args = common_args.copy()
        spectral_args.pop('n_heads', None)
        spectral_args.pop('n_kv_heads', None)
        spectral_args['emb_dim']      = getattr(args_cli, 'emb_dim',     128)
        spectral_args['num_experts']  = getattr(args_cli, 'num_experts', 128)
        spectral_args['top_k']        = getattr(args_cli, 'top_k',       8)
        
        if arch == 'spectral_v8_4':
            model_args = SpectralArgsV8_4(**spectral_args)
            model = SpectralThinkerV8_4(model_args)
        elif arch == 'spectral_v8_5':
            model_args = SpectralArgsV8_5(**spectral_args)
            model = SpectralThinkerV8_5(model_args)
        elif arch == 'spectral_v8_6':
            model_args = SpectralArgsV8_6(**spectral_args)
            model = SpectralThinkerV8_6(model_args)
        elif arch == 'spectral_v8_6b':
            model_args = SpectralArgsV8_6(**spectral_args)
            model = SpectralThinkerV8_6b(model_args)
    elif arch == 'coga_spectral':
        coga_spec_args = common_args.copy()
        coga_spec_args.pop('n_layers', None)
        coga_spec_args['n_pre_layers']  = getattr(args_cli, 'n_pre_layers', 2)
        coga_spec_args['n_core_layers'] = getattr(args_cli, 'n_core_layers', 4)
        coga_spec_args['n_post_layers'] = getattr(args_cli, 'n_post_layers', 2)
        coga_spec_args['max_recurrence_steps'] = getattr(args_cli, 'max_recurrence_steps', 4)
        coga_spec_args['n_scratch_slots'] = getattr(args_cli, 'n_scratch_slots', 32)
        coga_spec_args['n_experts'] = getattr(args_cli, 'n_experts', 8)
        coga_spec_args['top_k'] = getattr(args_cli, 'top_k', 2)
        coga_spec_args['n_reserved'] = getattr(args_cli, 'n_reserved', 4)
        coga_spec_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        coga_spec_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        coga_spec_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        model_args = CogaSpectralArgs(**coga_spec_args)
        model = TinyThinkerCogaSpectral(model_args)
    elif arch == 'analog':
        model_args = AnalogArgs(**common_args)
        model = TinyThinkerAnalog(model_args)
    elif arch == 'auto_architect':
        auto_args = common_args.copy()
        auto_args.pop('n_layers', None) # Auto-Architect empieza con 1
        auto_args['k_dim_attn']   = getattr(args_cli, 'k_dim_attn',   64)
        auto_args['k_dim_ffn']    = getattr(args_cli, 'k_dim_ffn',    64)
        auto_args['k_hidden_ffn'] = getattr(args_cli, 'k_hidden_ffn', 128)
        model_args = AutoArchitectArgs(**auto_args)
        model = TinyThinkerAutoArchitect(model_args)
    elif arch == 'auto_analog':
        auto_args = common_args.copy()
        auto_args.pop('n_layers', None) # Auto-Analog empieza con 1
        model_args = AutoAnalogArgs(**auto_args)
        model = TinyThinkerAutoAnalog(model_args)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
        
    model.to(device)
    
    # LÓGICA BLUEPRINT FASE 1: Congelación de Pesos
    if getattr(args_cli, 'phase1', False):
        t_print("⚠️ BLUEPRINT FASE 1 ACTIVADA: Congelando todo excepto GATES...")
        gated_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if any(x in name for x in ['gate', 'alpha', 'beta']):
                param.requires_grad = True
                gated_params += 1
            else:
                param.requires_grad = False
                frozen_params += 1
        t_print(f" -> Capas para Gating: {gated_params}")
        t_print(f" -> Capas Congeladas:  {frozen_params}")
        if gated_params == 0:
            t_print("❌ ERROR: No se encontraron parámetros de gating. ¿Estás usando una arquitectura 'b'?")
            sys.exit(1)
    
    if args_cli.use_gradient_checkpointing:
        if arch in ('coga', 'coga_spectral'):
            for layer in model.pre_layers: layer.use_checkpoint = True
            for layer in model.core_layers: layer.use_checkpoint = True
            for layer in model.post_layers: layer.use_checkpoint = True
        else:
            for layer in model.layers: layer.use_checkpoint = True

    # ----------------------------------
    # Helper: Optimizador con Weight Decay Selectivo
    # ----------------------------------
    def create_optimizer(model_obj, lr, weight_decay, opt_type):
        decay_params = []
        no_decay_params = []
        for n, p in model_obj.named_parameters():
            if not p.requires_grad:
                continue
            if not p.requires_grad: continue
            
            # Excluimos bias, normas y parámetros del dominio espectral (signatures, weights, basis)
            if p.dim() < 2 or any(x in n for x in ['signatures', 'weights', 'basis', 'norm']):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
                
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if opt_type == 'swo':
            return SuperMarioOptimizer(optim_groups, lr=lr, k_ratio=0.25)
        elif _is_dml:
            return DMLAdamW(optim_groups, lr=lr)
        else:
            return torch.optim.AdamW(optim_groups, lr=lr, foreach=False)

    weight_decay = getattr(args_cli, 'weight_decay', 0.0)
    if weight_decay > 0:
        print(f"⚠️ AVISO: Usando Weight Decay = {weight_decay} (No recomendado para redes espectrales)")
    
    if args_cli.optimizer == 'swo':
        print("[Optimizador] Usando SMO (SuperMarioOptimizer) - Compresión de estado al 93% (K=0.25)")
    elif _is_dml:
        print("[Optimizador] Usando DMLAdamW personalizado sin 'lerp_' para máxima compatibilidad con AMD")
        
    optimizer = create_optimizer(model, args_cli.lr, weight_decay, args_cli.optimizer)
    
    iter_num = 0
    best_val_loss = 1e9
    plateau_counter = 0

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
            torch.serialization.add_safe_globals([DenseArgs, MoEArgs, CogaArgs, SpectralArgs, SpectralArgsV4, SpectralArgsV5, CogaSpectralArgs, SpectralArgsV8_4, SpectralArgsV8_5, SpectralArgsV8_6])
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=False)

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception:
                print("[Resume] Aviso: Reiniciando estados del optimizador por cambio de configuración de grupos de parámetros.")
                
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
        # BLUEPRINT FASE 1: Forzamos LR alto y fijo para los gates
        if getattr(args_cli, 'phase1', False):
            return args_cli.lr
            
        if it < DEFAULT_WARMUP:
            return args_cli.lr * it / DEFAULT_WARMUP
        if it > args_cli.max_iters:
            return DEFAULT_MIN_LR
        decay_ratio = (it - DEFAULT_WARMUP) / (args_cli.max_iters - DEFAULT_WARMUP)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return DEFAULT_MIN_LR + coeff * (args_cli.lr - DEFAULT_MIN_LR)

    total_params = sum(p.numel() for p in model.parameters())
    model_file = f"model/model_{arch}.py" if arch != 'dense' else "model/model.py"
    
    # Resolver n_layers para el display
    if hasattr(model_args, 'n_layers'):
        n_layers_str = str(model_args.n_layers)
    elif arch in ('auto_architect', 'auto_analog'):
        n_layers_str = str(len(model.layers))
    else:
        n_layers_str = f"{getattr(model_args, 'n_pre_layers', 0)}+{getattr(model_args, 'n_core_layers', 0)}+{getattr(model_args, 'n_post_layers', 0)}"

    # Identificar Optimizador
    opt_name = getattr(args_cli, 'optimizer', 'adamw').upper()
    if opt_name == 'SWO': opt_name = "SuperMarioOptimizer (SMO)"
    elif _is_dml: opt_name = "DMLAdamW (DirectML AMD)"
    else: opt_name = "AdamW (Foreach=False)"

    header = f"""========================================
DATE: {start_date.strftime('%Y-%m-%d %H:%M:%S')}
DEVICE: {str(device).upper()}
CPU THREADS: {torch.get_num_threads()}
--------------- FILES -----------------
model_file: {model_file}
tokenizer:  {args_cli.tokenizer_path}
dataset:    {args_cli.data_path}
--------------- HYPERPARAMS -----------
batch_size: {args_cli.batch_size}
seq_len: {args_cli.seq_len}
grad_accum_steps: {args_cli.grad_accum_steps}
max_iters: {args_cli.max_iters}
learning_rate: {args_cli.lr} (min: {getattr(args_cli, 'min_lr', 'N/A')})
warmup_iters: {getattr(args_cli, 'warmup_iters', 0)}
weight_decay: {getattr(args_cli, 'weight_decay', 0.1)}
grad_clip: {getattr(args_cli, 'grad_clip', 1.0)}
--------------- OPTIMIZER -------------
name: {opt_name}
k_ratio: {getattr(args_cli, 'k_ratio', 0.25) if 'SMO' in opt_name else 'N/A'}
--------------- MODEL PARAMS ----------
dim: {model_args.dim}
n_layers: {n_layers_str}
n_heads: {getattr(model_args, 'n_heads', 'N/A')}
vocab_size: {model_args.vocab_size}
TOTAL PARAMS: {total_params / 1e6:.2f}M
========================================"""
    t_print(header)

    # ----------------------------------
    # 6. Bucle Principal
    # ----------------------------------
    t_print(f"Entrenamiento activo en {str(device).upper()} | Iteraciones: {iter_num}/{args_cli.max_iters}")
    t0 = time.time()

    while iter_num <= args_cli.max_iters:
        # LÓGICA BLUEPRINT FASE 2: Rotación de Capas (Round-Robin)
        if getattr(args_cli, 'phase2', False) and (iter_num % args_cli.rotation_iters == 0):
            n_layers = len(model.layers) if hasattr(model, 'layers') else 0
            if n_layers > 0:
                active_layer_idx = (iter_num // args_cli.rotation_iters) % (n_layers + 1)
                t_print(f"🔄 BLUEPRINT FASE 2: Rotando parámetros entrenables (Turno: {active_layer_idx})")
                
                # Congelar TODO (incluyendo gates de la fase anterior)
                for p in model.parameters(): p.requires_grad = False
                
                if active_layer_idx < n_layers:
                    # Turno de una capa específica
                    t_print(f" -> Activando Pesos de Capa {active_layer_idx}")
                    for name, p in model.layers[active_layer_idx].named_parameters():
                        if not any(x in name for x in ['gate', 'alpha', 'beta']):
                            p.requires_grad = True
                else:
                    # Turno de Embeddings y Bases (Foundation)
                    t_print(f" -> Activando Embeddings y Bases")
                    if hasattr(model, 'codes'): model.codes.requires_grad = True
                    if hasattr(model, 'basis'): model.basis.requires_grad = True
                
                # Reiniciar optimizador para la nueva configuración de parámetros
                optimizer = create_optimizer(model, args_cli.lr, weight_decay, args_cli.optimizer)

        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        # Evaluación y Checkpoint
        if iter_num % DEFAULT_EVAL_INTERVAL == 0 or iter_num == args_cli.max_iters:
            losses = estimate_loss()
            t_print(f"Iter {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")

            checkpoint = {
                'model': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'args': model_args,
                'arch': arch,
                'val_loss': losses['val']
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_pretrain_latest.pt'))
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                plateau_counter = 0
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_pretrain_best.pt'))
                t_print(f" -> Nuevo mejor modelo (val_loss: {best_val_loss:.4f})")
            else:
                plateau_counter += 1

            # Lógica Neurogénesis Residual (Auto-Architect V170)
            if arch in ('auto_architect', 'auto_analog') and plateau_counter >= 3:
                t_print(f"🌱 [{arch.upper()}] Estancamiento detectado (Paciencia 3). Aplicando Neurogénesis...")
                model.add_residual_layer()
                model.to(device)

                # Reinstanciar optimizador para capturar solo los nuevos parámetros (grad=True)
                optimizer = create_optimizer(model, args_cli.lr, weight_decay, args_cli.optimizer)

                plateau_counter = 0
                best_val_loss = losses['val'] # Reset baseline para la nueva capa especializados
                t_print(f"✅ Optimizador reiniciado. Parámetros totales: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # Salir si llegamos al límite
        if iter_num >= args_cli.max_iters:
            break

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
        if iter_num % 10 == 0 or iter_num < 10:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Multiplicamos por accum para ver loss real
            loss_val = loss.item() * args_cli.grad_accum_steps
            t_print(f"iter {iter_num:5d} | loss {loss_val:.4f} | lr {lr:.2e} | time {dt:.2f}s")

        # Gestión de Memoria Periódica (Evita el bloat de 60GB)
        if iter_num % 25 == 0:
            gc.collect()
            if _is_dml:
                try:
                    import torch_directml
                    torch_directml.empty_cache()
                except:
                    pass

        iter_num += 1
    t_print("Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    main()
