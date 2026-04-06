import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import contextlib
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs

# -----------------
# Configuración CPT (Continual Pre-Training / Fine-Tuning)
# -----------------
batch_size = 4
seq_len = 256
max_iters = 500  # Con pocos datos no hacen falta muchas iteraciones
eval_interval = 50
eval_iters = 10
learning_rate = 3e-5  # Muy bajo para no "destruir" la gramática recién aprendida
weight_decay = 1e-1
grad_clip = 1.0
grad_accum_steps = 4

# Ajuste automático del dispositivo (igual que tu mejora en train.py)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = 'mps'

if device == 'cuda':
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif device == 'cpu':
    ptdtype = torch.bfloat16  # AVX-512 BF16 nativo en Zen 4
else:
    ptdtype = torch.float32

ctx = torch.amp.autocast(device_type=device if device != 'mps' else 'cpu', dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16 and device == 'cuda'))

# ==========================================
# GESTIÓN SEGURA DE CHECKPOINTS
# ==========================================
BASE_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_best.pt")
OUT_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_finetuned.pt")
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "tool_dataset_real.json")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "logs", "finetune.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_tokenize_dataset():
    if not os.path.exists(DATA_FILE):
        logger.error(f"Error: No se encontró {DATA_FILE}")
        sys.exit(1)
        
    logger.info("Cargando tokenizador y transcribiendo dataset sintético a memoria...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Como son pocos datos (50-500 JSONs), los metemos en memoria de golpe (sin memmap)
    all_ids = []
    for example in dataset:
        tokens = tokenizer.encode(example["text"]).ids
        all_ids.extend(tokens)
        
    # Convertir a un solo tensor maestro
    data_tensor = torch.tensor(all_ids, dtype=torch.long)
    logger.info(f"Dataset de Fine-Tuning procesado: {len(data_tensor)} tokens listos para alinear a TinyThinker.")
    return data_tensor

def get_batch(data):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+1+seq_len] for i in ix])
    if device in ('cuda', 'cpu') and torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    elif device == 'mps':
        x, y = x.to(device), y.to(device)
    return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyThinker con soporte LoRA")
    parser.add_argument('--lora_r', type=int, default=0, help='Rank de LoRA. 0 = deshabilitado.')
    parser.add_argument('--lora_alpha', type=float, default=16.0, help='Escala de LoRA.')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='Dropout de LoRA.')
    parser.add_argument('--data_file', type=str, default=DATA_FILE, help='Ruta del dataset JSON para fine-tuning')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size para fine-tuning')
    parser.add_argument('--seq_len', type=int, default=seq_len, help='Longitud de secuencia para fine-tuning')
    parser.add_argument('--max_iters', type=int, default=max_iters, help='Número máximo de iteraciones')
    parser.add_argument('--eval_interval', type=int, default=eval_interval, help='Intervalo de evaluación')
    parser.add_argument('--eval_iters', type=int, default=eval_iters, help='Evaluación con cuántos pasos')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='LR para fine-tuning')
    return parser.parse_args()


def freeze_base_weights(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data)
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def main():
    global BASE_CKPT
    if not os.path.exists(BASE_CKPT):
        # Si no existe el _best (por usar el script antiguo), buscamos el ckpt genérico rotativo
        BASE_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt.pt")
        if not os.path.exists(BASE_CKPT):
            logger.error("❌ Error: No hay modelo base. Debes finalizar la Fase 1 primero.")
            sys.exit(1)
            
    cmd_args = parse_args()
    global batch_size, seq_len, max_iters, eval_interval, eval_iters, learning_rate
    batch_size = cmd_args.batch_size
    seq_len = cmd_args.seq_len
    max_iters = cmd_args.max_iters
    eval_interval = cmd_args.eval_interval
    eval_iters = cmd_args.eval_iters
    learning_rate = cmd_args.learning_rate

    print(f"🧠 Cargando Mente Base Mestra desde: {os.path.basename(BASE_CKPT)}")
    checkpoint = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    args = checkpoint['args']
    args.lora_r = cmd_args.lora_r
    args.lora_alpha = cmd_args.lora_alpha
    args.lora_dropout = cmd_args.lora_dropout
    
    global DATA_FILE
    DATA_FILE = cmd_args.data_file

    model = TinyThinker(args)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    if cmd_args.lora_r > 0:
        freeze_base_weights(model)
        print(f"✅ LoRA activado: r={cmd_args.lora_r}, alpha={cmd_args.lora_alpha}, dropout={cmd_args.lora_dropout}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔧 Parámetros entrenables LoRA: {trainable}")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    else:
        print(f"🔧 Preparando Optimizador (Low Learning Rate: {learning_rate})...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data = load_and_tokenize_dataset()
    
    print("\n🚀 Iniciando Fase 2/3: Inyección de Lógica y Tool-Calling...")
    
    for iter_num in range(1, max_iters + 1):
        if iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, data)
            print(f"✨ [ITER {iter_num}] Error de Formato Lógico: {val_loss:.4f} | Guardando Especialización...")
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'val_loss': val_loss
            }
            torch.save(ckpt, OUT_CKPT)
            
        # Acumulación de gradientes
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            X, Y = get_batch(data)
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                loss = loss / grad_accum_steps
                
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
        if grad_clip != 0.0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        if iter_num % 10 == 0:
            print(f"iter {iter_num:4d} | fine-tune loss {loss.item()*grad_accum_steps:.4f}")

    print(f"\n🎯 ¡Especialización Completada!")
    print(f"El modelo original sigue intacto en '{os.path.basename(BASE_CKPT)}'")
    print(f"El NUEVO modelo con 'Tool-Calling' ha sido guardado como: '{os.path.basename(OUT_CKPT)}'")
    print("\nPara interactuar con él, ve a 'scripts/chat.py' y asegúrate de cambiar la ruta de CKPT_PATH a 'ckpt_finetuned.pt'")

if __name__ == '__main__':
    main()
