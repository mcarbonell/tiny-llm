import os
import sys
import json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import datetime

# Añadir ruta base
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs as DenseArgs
from model.model_moe import TinyThinkerMoE, ModelArgs as MoEArgs
from model.model_coga import TinyThinkerCOGA, ModelArgs as CogaArgs
from model.model_spectral import SpectralThinker, SpectralArgs
from model.model_spectral_v4 import SpectralThinker as SpectralThinkerV4, SpectralArgs as SpectralArgsV4
from model.model_spectral_v5 import SpectralThinker as SpectralThinkerV5, SpectralArgs as SpectralArgsV5
from model.model_coga_spectral import TinyThinkerCogaSpectral, CogaSpectralArgs
from model.model_analog import TinyThinkerAnalog, AnalogArgs
from model.model_auto_analog import TinyThinkerAutoAnalog, AutoAnalogArgs


def validate_dataset(data: object, path: str) -> list:
    """Valida la estructura del dataset antes de procesarlo.
    Devuelve la lista validada o lanza ValueError con mensaje descriptivo.
    """
    if not isinstance(data, list):
        raise ValueError(
            f"Dataset '{path}' debe ser una lista JSON de objetos, "
            f"pero se encontró: {type(data).__name__}"
        )
    if len(data) == 0:
        raise ValueError(f"Dataset '{path}' está vacío (lista de longitud 0).")

    # Validar estructura de los primeros N items (rápido, no bloquea con datasets grandes)
    sample_size = min(len(data), 10)
    for i, item in enumerate(data[:sample_size]):
        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset '{path}': el elemento [{i}] debe ser un objeto JSON, "
                f"pero se encontró: {type(item).__name__}"
            )
        if 'text' not in item:
            raise ValueError(
                f"Dataset '{path}': el elemento [{i}] no tiene campo 'text'. "
                f"Claves encontradas: {list(item.keys())}"
            )
        if not isinstance(item['text'], str):
            raise ValueError(
                f"Dataset '{path}': el campo 'text' del elemento [{i}] debe ser str, "
                f"pero se encontró: {type(item['text']).__name__}"
            )
    return data


def load_model_and_tokenizer(checkpoint_path, device='cpu'):
    """Carga el modelo y tokenizador (Soporta .pt y .tiny V198 High-Fidelity)."""
    tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer_v1.json"))
    
    # 1. Manejo de archivos comprimidos .tiny (V198)
    if checkpoint_path.endswith('.tiny'):
        import zlib
        import io
        import math
        print(f"📂 Descomprimiendo paquete V198 Hi-Fi: {os.path.basename(checkpoint_path)}...")
        with open(checkpoint_path, "rb") as f:
            compressed_data = f.read()
        raw_bytes = zlib.decompress(compressed_data)
        buffer = io.BytesIO(raw_bytes)
        
        torch.serialization.add_safe_globals([DenseArgs, MoEArgs, CogaArgs, SpectralArgs, SpectralArgsV4, SpectralArgsV5, CogaSpectralArgs, AnalogArgs, AutoAnalogArgs])
        checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
        
        # 2. Reconstrucción Espectral Inversa
        from scripts.compress_model import get_dct_matrix
        state_dict = checkpoint['model']
        reconstructed_state = {}
        
        for name, entry in state_dict.items():
            if isinstance(entry, dict) and entry.get('type') == 'spectral_dct_v198':
                # W_spatial = D_out.T @ W_spectral @ D_in
                spectral_coeffs = entry['data']
                out_f, in_f = spectral_coeffs.shape
                d_out = get_dct_matrix(out_f).to(spectral_coeffs.dtype)
                d_in = get_dct_matrix(in_f).to(spectral_coeffs.dtype)
                
                # Invertir transformada
                reconstructed_state[name] = d_out.t() @ spectral_coeffs @ d_in
            else:
                reconstructed_state[name] = entry
        checkpoint['model'] = reconstructed_state
    else:
        # Carga estándar .pt
        torch.serialization.add_safe_globals([DenseArgs, MoEArgs, CogaArgs, SpectralArgs, SpectralArgsV4, SpectralArgsV5, CogaSpectralArgs, AnalogArgs, AutoAnalogArgs])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Usar config del checkpoint
    config = checkpoint['args']
    arch = checkpoint.get('arch', 'dense')

    if arch == 'dense':
        model = TinyThinker(config)
    elif arch == 'moe':
        model = TinyThinkerMoE(config)
    elif arch == 'coga':
        model = TinyThinkerCOGA(config)
    elif arch == 'spectral':
        model = SpectralThinker(config)
    elif arch == 'spectral_v4':
        model = SpectralThinkerV4(config)
    elif arch == 'spectral_v5':
        model = SpectralThinkerV5(config)
    elif arch == 'coga_spectral':
        model = TinyThinkerCogaSpectral(config)
    elif arch == 'analog':
        model = TinyThinkerAnalog(config)
    elif arch == 'auto_analog':
        model = TinyThinkerAutoAnalog(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Lógica para modelos evolutivos (Neurogénesis)
    if arch in ('auto_architect', 'auto_analog'):
        # Contar cuántas capas hay realmente en el state_dict
        layer_indices = set([int(k.split('.')[1]) for k in checkpoint['model'].keys() if k.startswith('layers.')])
        target_num_layers = max(layer_indices) + 1 if layer_indices else 1
        
        # Empezamos con la capa 0 (que ya existe), y añadimos las demás
        while len(model.layers) < target_num_layers:
            curr_idx = len(model.layers)
            # Detectar tipo de capa en el checkpoint para esta posición
            is_lateral = any(f"layers.{curr_idx}.lateral" in k for k in checkpoint['model'].keys())
            
            if is_lateral:
                from model.model_lateral_v197 import ResidualLateralBlock
                model.layers.append(ResidualLateralBlock(config.dim))
            else:
                from model.model_auto_analog import ResidualAnalogLayer
                model.layers.append(ResidualAnalogLayer(config))
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, tokenizer, config

def calculate_perplexity(model, tokenizer, dataset_path, device='cpu', seq_len=None, batch_size=4, num_batches=10):
    """Calcula la perplexity en un dataset de validación."""
    if seq_len is None:
        seq_len = model.args.max_seq_len
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} no encontrado.")
        return None

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    try:
        data = validate_dataset(raw, dataset_path)
    except ValueError as e:
        print(f"Error de validación del dataset: {e}")
        return None

    # Tokenizar todo el dataset
    all_tokens = []
    for example in data:
        tokens = tokenizer.encode(example['text']).ids
        all_tokens.extend(tokens)

    data_tensor = torch.tensor(all_tokens, dtype=torch.long)

    model.eval()
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for _ in range(num_batches):
            # Obtener batch aleatorio
            if len(data_tensor) <= seq_len:
                continue
            start_idx = torch.randint(0, len(data_tensor) - seq_len - 1, (1,)).item()
            x = data_tensor[start_idx:start_idx + seq_len].unsqueeze(0).to(device)
            y = data_tensor[start_idx + 1:start_idx + seq_len + 1].unsqueeze(0).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            num_tokens += y.numel()

    if num_tokens == 0:
        return None

    avg_loss = total_loss / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def resolve_checkpoint_path(checkpoints_dir):
    priority = [
        "ckpt_sft_latest.pt",
        "ckpt_sft_best.pt",
        "ckpt_pretrain_best.pt",
        "ckpt_pretrain_latest.pt",
        "ckpt_finetuned.pt",
        "ckpt_best.pt",
        "ckpt.pt",
    ]
    for ckpt in priority:
        path = os.path.join(checkpoints_dir, ckpt)
        if os.path.exists(path):
            return path
    return None

def generate_text(model, tokenizer, input_ids, max_new_tokens=50, temperature=1.0, device='cpu', top_k=40):
    """Genera texto con KV-cache para eficiencia."""
    model.eval()
    prompt_tokens = torch.as_tensor(input_ids, dtype=torch.long, device=device).view(-1).tolist()
    if len(prompt_tokens) == 0:
        return torch.empty((1, 0), dtype=torch.long, device=device)

    generated_tokens = list(prompt_tokens)
    x = torch.tensor([generated_tokens], dtype=torch.long, device=device)
    past_key_values = None
    
    with torch.no_grad():
        x_cond = x[:, -model.args.max_seq_len:] if x.size(1) > model.args.max_seq_len else x
        logits, past_key_values = model(x_cond, use_cache=True)
        next_token_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        for _ in range(max_new_tokens):
            probs_cpu = F.softmax(next_token_logits, dim=-1).cpu()
            next_token = torch.multinomial(probs_cpu, 1).to(device).item()
            generated_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            outputs = model(input_tensor, past_key_values=past_key_values, use_cache=True)
            if isinstance(outputs, tuple):
                logits, past_key_values = outputs
            else:
                logits = outputs
                past_key_values = None
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
    
    return torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # Devolver (1, seq)

def evaluate_tool_calling_accuracy(model, tokenizer, dataset_path, device='cpu', max_length=512):
    """Evalúa la accuracy en tool-calling: si el modelo genera <TOOL_CALL> cuando es apropiado."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} no encontrado.")
        return None

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    try:
        data = validate_dataset(raw, dataset_path)
    except ValueError as e:
        print(f"Error de validación del dataset: {e}")
        return None

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for example in data[:50]:  # Limitar a 50 ejemplos para evaluación rápida
            prompt = example['text'].split('<TOOL_CALL>')[0]  # Usar todo hasta antes del tool call
            expected_tool_call = '<TOOL_CALL>' in example['text']

            # Generar respuesta
            input_ids = tokenizer.encode(prompt).ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # (1, len)

            generated_ids = generate_text(model, tokenizer, input_tensor, max_new_tokens=150, temperature=0.7, device=device, top_k=40)
            generated_text = tokenizer.decode(generated_ids[0].tolist())

            # Verificar si generó <TOOL_CALL>
            predicted_tool_call = '<TOOL_CALL>' in generated_text

            if predicted_tool_call == expected_tool_call:
                correct += 1
            else:
                if total < 3:  # Imprimir primeros 3 fallos para debug
                    print(f"Fallo en ejemplo {total+1}: Prompt: {prompt[:50]}...")
                    print(f"Generado: {generated_text[:100]}...")
                    print(f"Esperado tool_call: {expected_tool_call}, Predicho: {predicted_tool_call}")
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluar el modelo TinyThinker")
    parser.add_argument('--checkpoint', type=str, default=None, help='Ruta al checkpoint')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_golden_v1.json'), help='Ruta al dataset de evaluación')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'dml'], help='Dispositivo (cpu/cuda/dml)')
    args = parser.parse_args()

    device = 'cpu'
    if args.device == 'dml':
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            pass
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'

    # Resolver checkpoint
    if not args.checkpoint:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        args.checkpoint = resolve_checkpoint_path(checkpoints_dir)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("Error: No se encontró un checkpoint válido.")
        return

    print(f"Cargando modelo desde {args.checkpoint}...")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)

    total_params = sum(p.numel() for p in model.parameters())
    # Obtener n_layers de forma segura
    if hasattr(config, 'n_layers'):
        n_layers_str = str(config.n_layers)
    elif hasattr(config, 'n_pre_layers'):
        n_layers_str = f"{config.n_pre_layers}+{config.n_core_layers}+{config.n_post_layers}"
    else:
        n_layers_str = "N/A"

    print(f"""========================================
EVALUATION SESSION
DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DEVICE: {str(device).upper()}
CHECKPOINT: {os.path.basename(args.checkpoint)}
--------------- MODEL PARAMS ----------
dim: {config.dim}
n_layers: {n_layers_str}
n_heads: {config.n_heads}
vocab_size: {config.vocab_size}
TOTAL PARAMS: {total_params / 1e6:.2f}M
========================================""")

    # Calcular perplexity
    print("Calculando perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, args.dataset, device)
    if perplexity:
        print(f"Perplexity: {perplexity:.2f}")
    else:
        print("No se pudo calcular perplexity (dataset insuficiente).")

    # Evaluar tool-calling accuracy
    print("Evaluando accuracy en tool-calling...")
    accuracy = evaluate_tool_calling_accuracy(model, tokenizer, args.dataset, device)
    if accuracy is not None:
        print(f"Tool-calling Accuracy: {accuracy:.2%}")
    else:
        print("No se pudo evaluar tool-calling accuracy.")

if __name__ == "__main__":
    main()
