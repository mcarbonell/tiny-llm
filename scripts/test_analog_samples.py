
import os
import torch
import sys
from tokenizers import Tokenizer
import torch.nn.functional as F
import time
import argparse

# Añadir ruta base
sys.path.append(os.getcwd())
from model.model_analog import TinyThinkerAnalog, AnalogArgs

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # 1. Prefill
    with torch.no_grad():
        logits, pkv = model(x, use_cache=True)
    
    generated = list(tokens)
    curr_x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
    
    t0 = time.time()
    tokens_gen = 0
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # El modelo Analog soporta KV-Cache nativo según el forward
            logits, pkv = model(curr_x, past_key_values=pkv, use_cache=True)
            
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            curr_x = next_token
            generated.append(next_token.item())
            tokens_gen += 1
            
            if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
                break
                
    dt = time.time() - t0
    tps = tokens_gen / dt if dt > 0 else 0
    return tokenizer.decode(generated), tps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="checkpoints/analog_nano/ckpt_pretrain_best.pt")
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--temp', type=float, default=0.7)
    args_cli = parser.parse_args()

    checkpoint_path = args_cli.checkpoint
    tokenizer_path = "model/tokenizer_v1.json"
    device = 'cpu'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint no encontrado: {checkpoint_path}")
        return

    print(f"LOAD: Tokenizer {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"LOAD: Modelo {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_args' in checkpoint:
        args = checkpoint['model_args']
    elif 'args' in checkpoint:
        args = checkpoint['args']
    else:
        print("INFO: Usando parámetros manuales (Analog 256 dim, 6 layers)")
        args = AnalogArgs(dim=256, n_layers=6, n_heads=8, vocab_size=16384)

    model = TinyThinkerAnalog(args)
    
    # Manejar posibles variaciones en el state_dict
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    
    prompts = [
        "El secreto de la inteligencia es",
        "En un lugar de la Mancha",
        "La red neuronal predijo que",
        "User: Hola! Assistant:"
    ]
    
    print("\n" + "="*50)
    print(f"SAMPLES FROM ANALOG MODEL (Log: 20260430_215919)")
    print("="*50)
    
    for p in prompts:
        safe_print(f"\nPROMPT: {p}")
        completion, tps = generate(model, tokenizer, p, max_new_tokens=args_cli.tokens, temperature=args_cli.temp, device=device)
        safe_print(f"GEN: {completion}")
        safe_print(f"SPEED: {tps:.2f} tokens/sec")
        safe_print("-" * 30)

if __name__ == "__main__":
    main()
