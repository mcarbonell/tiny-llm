import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import contextlib
import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs

# -----------------
# Configuración SFT (Supervised Fine-Tuning)
# -----------------
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 1024 # Aumentamos para permitir búsquedas más ricas
DEFAULT_MAX_ITERS = 1000
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_EVAL_ITERS = 20
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_GRAD_ACCUM = 4

# Paths por defecto
BASE_CKPT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_base_305M_ctx1024.pt")
OUT_CKPT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_sft_ctx1024.pt")
DATA_FILE_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "tool_dataset_mixed.json")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")

def build_sft_example(full_text, tokenizer, max_seq_len):
    """Construye un ejemplo causal SFT con labels desplazadas una posición."""
    if "Assistant:" not in full_text:
        return None

    parts = full_text.split("Assistant:")
    prompt_text = parts[0] + "Assistant:"

    prompt_ids = tokenizer.encode(prompt_text).ids
    full_ids = tokenizer.encode(full_text).ids[:max_seq_len]

    input_ids = full_ids
    targets = full_ids[1:] + [-100]
    prompt_cutoff = min(max(len(prompt_ids) - 1, 0), len(targets))
    targets[:prompt_cutoff] = [-100] * prompt_cutoff

    padding_len = max_seq_len - len(input_ids)
    if padding_len > 0:
        input_ids += [0] * padding_len
        targets += [-100] * padding_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
    }

def load_sft_dataset(data_file, tokenizer, max_seq_len):
    """
    Carga el dataset y devuelve pares (input_ids, target_mask).
    La máscara será -100 para los tokens del prompt y el ID real para la respuesta.
    """
    if not os.path.exists(data_file):
        print(f"Error: No se encontró {data_file}")
        sys.exit(1)
        
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    processed_examples = []

    print(f"Procesando {len(dataset)} ejemplos para SFT...")
    
    for example in dataset:
        built = build_sft_example(example["text"], tokenizer, max_seq_len)
        if built is not None:
            processed_examples.append(built)
        
    return processed_examples

def get_batch_sft(examples, bsz, device):
    ix = torch.randint(0, len(examples), (bsz,))
    x = torch.stack([examples[i]["input_ids"] for i in ix])
    y = torch.stack([examples[i]["labels"] for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def parse_args():
    parser = argparse.ArgumentParser(description="TinyThinker SFT (Supervised Fine-Tuning)")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'dml', 'mps'], help='Dispositivo.')
    parser.add_argument('--lora_r', type=int, default=16, help='Rank de LoRA.')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='Escala de LoRA.')
    parser.add_argument('--max_iters', type=int, default=DEFAULT_MAX_ITERS, help='Iteraciones.')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LR, help='LR.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size.')
    parser.add_argument('--data_file', type=str, default=DATA_FILE_DEFAULT, help='Ruta al dataset SFT.')
    return parser.parse_args()

def freeze_base_weights(model):
    for name, param in model.named_parameters():
        # SOLO entrenamos adaptadores LoRA. 
        # Mantener embeddings y pesos base congelados es crucial en modelos pequeños 
        # para evitar el colapso lingüístico durante el SFT.
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def main():
    cmd_args = parse_args()
    
    # 1. Hardware
    device = 'cpu'
    if cmd_args.device == 'dml':
        import torch_directml
        device = torch_directml.device()
    elif cmd_args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    
    # 2. Modelo
    print(f"Cargando base: {os.path.basename(BASE_CKPT_DEFAULT)}")
    checkpoint = torch.load(BASE_CKPT_DEFAULT, map_location='cpu', weights_only=False)
    model_args = checkpoint['args']
    model_args.lora_r = cmd_args.lora_r
    model_args.lora_alpha = cmd_args.lora_alpha
    
    model = TinyThinker(model_args)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    freeze_base_weights(model)

    # 3. Dataset SFT con Máscara
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    dataset = load_sft_dataset(cmd_args.data_file, tokenizer, DEFAULT_SEQ_LEN)
    
    # 4. Logs
    log_file = os.path.join("logs", f"sft_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    start_time = time.time()
    def t_print(msg):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        full = f"[{elapsed}] {msg}"
        print(full)
        with open(log_file, "a") as f: f.write(full + "\n")

    metadata_block = f"""==================================================
METADATOS DE EJECUCION SFT
Fecha/Hora : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Hardware   : {device} (Hilos CPU: {os.cpu_count()})
Dataset    : {cmd_args.data_file}
Checkpoint : {BASE_CKPT_DEFAULT}
Salida     : {OUT_CKPT_DEFAULT}
Hiperparametros:
  - Max Iters : {cmd_args.max_iters}
  - Batch Size: {cmd_args.batch_size}
  - Seq Len   : {DEFAULT_SEQ_LEN}
  - Learn Rate: {cmd_args.learning_rate:.2e}
  - LoRA r/alp: {cmd_args.lora_r} / {cmd_args.lora_alpha}
  - Grad Accum: {DEFAULT_GRAD_ACCUM}
Parametros : Entrenables={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M
=================================================="""
    print(metadata_block)
    with open(log_file, "a") as f:
        f.write(metadata_block + "\n")

    t_print(f"SFT Iniciado | Device: {device} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cmd_args.learning_rate)

    # 5. Bucle
    t0 = time.time()
    best_val_loss = float('inf')
    
    for iter_num in range(1, cmd_args.max_iters + 1):
        if iter_num % DEFAULT_EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                X, Y = get_batch_sft(dataset, DEFAULT_EVAL_ITERS, device)
                logits = model(X)
                # CrossEntropy ignora automáticamente el índice -100
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                val_loss_val = loss.item()
                
                # Guardar siempre un checkpoint histórico
                ckpt_dict = {'model': model.state_dict(), 'args': model_args, 'iter_num': iter_num, 'val_loss': val_loss_val}
                history_ckpt = OUT_CKPT_DEFAULT.replace('.pt', f'_iter{iter_num}.pt')
                torch.save(ckpt_dict, history_ckpt)
                
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    t_print(f"Iter {iter_num} | val_loss {val_loss_val:.4f} | ¡Nuevo Mejor! Guardando...")
                    torch.save(ckpt_dict, OUT_CKPT_DEFAULT)
                else:
                    t_print(f"Iter {iter_num} | val_loss {val_loss_val:.4f} | (No mejora, best: {best_val_loss:.4f})")
            model.train()

        optimizer.zero_grad()
        for _ in range(DEFAULT_GRAD_ACCUM):
            X, Y = get_batch_sft(dataset, cmd_args.batch_size, device)
            logits = model(X)
            # Solo se calcula el error en los tokens de la respuesta (donde Y != -100)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss = loss / DEFAULT_GRAD_ACCUM
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter_num % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            t_print(f"iter {iter_num:4d} | loss {loss.item()*DEFAULT_GRAD_ACCUM:.4f} | time {dt:.2f}s")

    t_print("SFT Completado!")

if __name__ == '__main__':
    main()
