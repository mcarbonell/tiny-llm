import os
import torch
import sys
from tokenizers import Tokenizer
import torch.nn.functional as F

# Añadir ruta base
sys.path.append(os.getcwd())
from model.model_spectral_v7 import SpectralThinker, SpectralArgs

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Simple sampling loop (no KV-cache for simplicity in this helper)
    generated = list(tokens)
    for _ in range(max_new_tokens):
        logits = model(x[:, -512:]) # Respect max seq len
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
            break
            
    return tokenizer.decode(generated)

def main():
    checkpoint_path = "checkpoints/spectral_v7_hd_pure/ckpt_pretrain_latest.pt"
    tokenizer_path = "model/tokenizer_v2_32k.json"
    device = 'cpu'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return

    print(f"LOAD: Tokenizer {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"LOAD: Modelo {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    print(f"SAMPLES AT ITERATION {checkpoint['iter_num']}")
    print("="*50)
    
    for p in prompts:
        safe_print(f"\nPROMPT: {p}")
        completion = generate(model, tokenizer, p, max_new_tokens=40)
        safe_print(f"GEN: {completion}")
        safe_print("-" * 30)

if __name__ == "__main__":
    main()
