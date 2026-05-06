import os
import torch
import sys
from tokenizers import Tokenizer
import torch.nn.functional as F

# Añadir ruta base
sys.path.append(os.getcwd())
from model.model_spectral_v7 import SpectralThinker, SpectralArgs

import time
import argparse

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

def generate_recurrent(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # 1. Prefill (Cargar el contexto inicial)
    with torch.no_grad():
        # En v7 el forward devuelve (logits, new_kvs) si use_cache=True
        logits, pkv = model(x, use_cache=True)
    
    generated = list(tokens)
    curr_x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
    
    t0 = time.time()
    tokens_gen = 0
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            # Inferencia O(1): procesamos solo el último token
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
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'dml'])
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--temp', type=float, default=0.8)
    args_cli = parser.parse_args()

    checkpoint_path = "checkpoints/spectral_v7_hd_pure/ckpt_pretrain_latest.pt"
    tokenizer_path = "model/tokenizer_v2_32k.json"
    device = args_cli.device
    
    if device == 'dml':
        import torch_directml
        device = torch_directml.device()
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint no encontrado: {checkpoint_path}")
        return

    print(f"LOAD: Tokenizer {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"LOAD: Modelo {checkpoint_path} en {device}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    model = SpectralThinker(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    prompts = [
        "Once upon a time",
        "The secret of the universe is",
        "User: Hello! Assistant:",
        "In a faraway kingdom, there was a"
    ]
    
    print("\n" + "="*50)
    print(f"SAMPLES AT ITERATION {checkpoint['iter_num']} (V7 - 5M Params)")
    print("="*50)
    
    for p in prompts:
        safe_print(f"\nPROMPT: {p}")
        completion, tps = generate_recurrent(model, tokenizer, p, max_new_tokens=args_cli.tokens, temperature=args_cli.temp, device=device)
        safe_print(f"GEN: {completion}")
        safe_print(f"SPEED: {tps:.2f} tokens/sec")
        safe_print("-" * 30)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
