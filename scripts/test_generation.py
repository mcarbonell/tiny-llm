import os
import sys
import torch
import argparse
from tokenizers import Tokenizer
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs as DenseArgs
from model.model_moe import TinyThinkerMoE, ModelArgs as MoEArgs
from model.model_coga import TinyThinkerCOGA, ModelArgs as CogaArgs
from model.model_spectral import SpectralThinker, SpectralArgs
from model.model_spectral_v4 import SpectralThinker as SpectralThinkerV4, SpectralArgs as SpectralArgsV4
from model.model_spectral_v5 import SpectralThinker as SpectralThinkerV5, SpectralArgs as SpectralArgsV5
from model.model_coga_spectral import TinyThinkerCogaSpectral, CogaSpectralArgs
from model.model_analog import TinyThinkerAnalog, AnalogArgs

def load_checkpoint(ckpt_path, device='cpu'):
    torch.serialization.add_safe_globals([DenseArgs, MoEArgs, CogaArgs, SpectralArgs, SpectralArgsV4, SpectralArgsV5, CogaSpectralArgs, AnalogArgs])
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    arch = checkpoint.get('arch', 'dense')
    
    if arch == 'dense':
        model = TinyThinker(args)
    elif arch == 'moe':
        model = TinyThinkerMoE(args)
    elif arch == 'coga':
        model = TinyThinkerCOGA(args)
    elif arch == 'spectral':
        model = SpectralThinker(args)
    elif arch == 'spectral_v4':
        model = SpectralThinkerV4(args)
    elif arch == 'spectral_v5':
        model = SpectralThinkerV5(args)
    elif arch == 'coga_spectral':
        model = TinyThinkerCogaSpectral(args)
    elif arch == 'analog':
        model = TinyThinkerAnalog(args)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(device)
    return model, arch, args, checkpoint.get('iter_num', 0), checkpoint.get('val_loss', 0.0)

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=40, device='cpu'):
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print(f"\n[PROMPT] {prompt}")
    print("[GENERANDO] ", end="", flush=True)
    
    generated_text = ""
    past_key_values = None
    
    with torch.no_grad():
        logits, past_key_values = None, None
        for step in range(max_tokens):
            if logits is None:
                logits, past_key_values = model(x, use_cache=True)
            else:
                last_token = x[:, -1:]
                logits, past_key_values = model(last_token, past_key_values=past_key_values, use_cache=True)

            logits_step = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits_step, dim=-1)
            probs_cpu = probs.cpu()
            if torch.isnan(probs_cpu).any() or torch.isinf(probs_cpu).any():
                 probs_cpu = torch.nan_to_num(probs_cpu, nan=0.0, posinf=1.0, neginf=0.0)
            if probs_cpu.sum() == 0:
                 probs_cpu = torch.ones_like(probs_cpu) / probs_cpu.size(-1)

            next_token = torch.multinomial(probs_cpu, num_samples=1).to(device)
            x = torch.cat((x, next_token), dim=1)
            
            token_id = next_token.item()
            if token_id == tokenizer.token_to_id("<eos>") or token_id == tokenizer.token_to_id("<pad>"):
                break
                
            new_word = tokenizer.decode([token_id], skip_special_tokens=False)
            generated_text += new_word
            print(new_word, end="", flush=True)
            
    print("\n")
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint a evaluar")
    parser.add_argument("--tokenizer", type=str, default="model/tokenizer_v1.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint no encontrado en {args.checkpoint}")
        return

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    print(f"Cargando modelo desde {args.checkpoint}...")
    model, arch, model_args, iter_num, val_loss = load_checkpoint(args.checkpoint, device)
    
    print(f"=====================================")
    print(f"Arquitectura: {arch}")
    print(f"Iteración: {iter_num} | Val Loss: {val_loss:.4f}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"=====================================")

    test_prompts = [
        "Para calcular la hipotenusa de un triángulo rectángulo",
        "El presidente de los Estados Unidos",
        "def fibonacci(n):",
        "La capital de Francia es",
        "[SYSTEM] You are TinyThinker. [/SYSTEM]\nUser: Escribe un poema sobre el mar.\nAssistant:"
    ]

    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, max_tokens=args.max_tokens, device=device)

if __name__ == "__main__":
    main()
