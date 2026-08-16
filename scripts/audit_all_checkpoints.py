import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs as DenseArgs
from model.model_moe import TinyThinkerMoE, ModelArgs as MoEArgs
from model.model_coga import TinyThinkerCOGA, ModelArgs as CogaArgs
from model.model_spectral import SpectralThinker, SpectralArgs
from model.model_spectral_v4 import SpectralThinker as SpectralThinkerV4, SpectralArgs as SpectralArgsV4
from model.model_spectral_v5 import SpectralThinker as SpectralThinkerV5, SpectralArgs as SpectralArgsV5
from model.model_spectral_v6 import SpectralThinker as SpectralThinkerV6, SpectralArgs as SpectralArgsV6
from model.model_spectral_v7 import SpectralThinker as SpectralThinkerV7, SpectralArgs as SpectralArgsV7
from model.model_coga_spectral import TinyThinkerCogaSpectral, CogaSpectralArgs
from model.model_analog import TinyThinkerAnalog, AnalogArgs
from model.model_auto_analog import TinyThinkerAutoAnalog, AutoAnalogArgs
from model.model_spectral_v10_hippocampus import SpectralThinkerV10, SpectralArgsV10
from model.model_spectral_v11_albert import SpectralThinkerV11, SpectralArgsV11
from model.model_spectral_v12_delta_phase import SpectralThinkerV12, SpectralArgsV12

def audit_checkpoint(ckpt_path):
    print("\n" + "=" * 90)
    print(f"AUDITANDO CHECKPOINT: {ckpt_path}")
    print("=" * 90)
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} no existe.")
        return
        
    torch.serialization.add_safe_globals([
        DenseArgs, MoEArgs, CogaArgs, SpectralArgs, SpectralArgsV4, SpectralArgsV5, 
        SpectralArgsV6, SpectralArgsV7, CogaSpectralArgs, AnalogArgs, AutoAnalogArgs, 
        SpectralArgsV10, SpectralArgsV11, SpectralArgsV12
    ])
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    arch = ckpt.get('arch', 'dense')
    iter_num = ckpt.get('iter_num', 'N/A')
    val_loss = ckpt.get('val_loss', 'N/A')
    
    vocab_size = getattr(args, 'vocab_size', 16384)
    print(f"Arquitectura: {arch} | Iteración: {iter_num} | Val Loss: {val_loss} | Vocab Size: {vocab_size}")
    
    # Seleccionar tokenizador según vocab_size
    if vocab_size == 32768:
        tok_path = "model/tokenizer_v2_32k.json"
    elif os.path.exists("model/tokenizer_v1.json"):
        tok_path = "model/tokenizer_v1.json"
    else:
        tok_path = "model/tokenizer.json"
        
    print(f"Tokenizador seleccionado: {tok_path}")
    tokenizer = Tokenizer.from_file(tok_path)
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()
        
    # Instanciar modelo
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
    elif arch == 'spectral_v6':
        model = SpectralThinkerV6(args)
    elif arch == 'spectral_v7':
        model = SpectralThinkerV7(args)
    elif arch == 'coga_spectral':
        model = TinyThinkerCogaSpectral(args)
    elif arch == 'analog':
        model = TinyThinkerAnalog(args)
    elif arch == 'auto_analog':
        model = TinyThinkerAutoAnalog(args)
    elif arch == 'spectral_v10':
        model = SpectralThinkerV10(args)
    elif arch == 'spectral_v11':
        model = SpectralThinkerV11(args)
    elif arch == 'spectral_v12':
        model = SpectralThinkerV12(args)
    else:
        print(f"Arquitectura desconocida: {arch}")
        return
        
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parámetros: {total_params:.2f}M")
    
    prompts = [
        "Once upon a time, there was a little girl named Lily who",
        "Sara wanted to bake a big chocolate cake, so she",
        "The quick brown fox jumps over the lazy dog and"
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long)
        
        print(f"\n--- PROMPT: \"{prompt}\" ---")
        print("[GENERADO]: ", end="", flush=True)
        
        decoded_so_far = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        with torch.no_grad():
            for step in range(50):
                out = model(x)
                logits_out = out[0] if isinstance(out, tuple) else out
                logits_step = logits_out[:, -1, :] / 0.7
                
                # top-k
                v, _ = torch.topk(logits_step, min(40, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits_step, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                if probs.sum() == 0:
                    probs = torch.ones_like(probs) / probs.size(-1)
                    
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
                
                token_id = next_token.item()
                if token_id in (tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<pad>")):
                    break
                    
                current_ids = x[0].tolist()
                decoded_current = tokenizer.decode(current_ids, skip_special_tokens=True)
                new_text = decoded_current[len(decoded_so_far):]
                decoded_so_far = decoded_current
                print(new_text, end="", flush=True)
        print("\n")

def main():
    checkpoints_to_test = [
        "checkpoints/test_v11_e256_d1024_k512_l8/ckpt_pretrain_best.pt",
        "checkpoints/test_v11_e256_d2048_k512_l8/ckpt_pretrain_best.pt",
        "checkpoints/spectral_v10/ckpt_pretrain_best.pt",
        "checkpoints/spectral_v8_6_universal/ckpt_pretrain_best.pt",
        "checkpoints/spectral_v7_hd_pure/ckpt_pretrain_best.pt",
        "checkpoints/ckpt_pretrain_best.pt"
    ]
    
    for ckpt in checkpoints_to_test:
        if os.path.exists(ckpt):
            try:
                audit_checkpoint(ckpt)
            except Exception as e:
                print(f"Error evaluando {ckpt}: {e}")

if __name__ == "__main__":
    main()
