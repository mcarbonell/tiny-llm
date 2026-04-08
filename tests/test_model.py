import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import json
from tokenizers import Tokenizer
from model.model import TinyThinker, ModelArgs, precompute_freqs_cis, LoRALinear

def test_model_forward():
    # Instanciamos una versión de juguete de la red solo para comprobar cálculos de tensores
    args = ModelArgs(
        vocab_size=1000, 
        dim=128, 
        n_layers=2, 
        n_heads=4, 
        n_kv_heads=2,
        max_seq_len=64
    )
    model = TinyThinker(args)
    
    # Creamos un lote falso: Batch de 2, Secuencia de 16 tokens
    bsz = 2
    seqlen = 16
    tokens = torch.randint(0, args.vocab_size, (bsz, seqlen))
    
    # Forward Pass
    logits = model(tokens)
    
    # Tiene que devolver (Batch Size, Seq Len, Vocab Size)
    assert logits.shape == (bsz, seqlen, args.vocab_size), f"Error en las dimensiones devueltas: {logits.shape}"
    assert not torch.isnan(logits).any(), "Los logits contienen valores NaN (inestabilidad matemática)"
    print("Todo correcto: Las dimensiones del Forward Pass coinciden con la teoria.")

def test_rotary_embeddings():
    """Verificar que RoPE genera embeddings con shapes correctas."""
    dim = 64
    end = 10
    freqs_cis = precompute_freqs_cis(dim, end)
    assert freqs_cis.shape == (end, dim // 2), f"Shape incorrecta: {freqs_cis.shape}"
    assert not torch.isnan(freqs_cis).any(), "RoPE contiene NaN"
    print("RoPE embeddings verificados correctamente.")

def test_gqa_shapes():
    """Verificar que GQA reduce correctamente las heads de KV."""
    args = ModelArgs(dim=128, n_heads=8, n_kv_heads=4, vocab_size=1000, max_seq_len=64)
    model = TinyThinker(args)
    
    # Verificar shapes en attention
    bsz, seqlen = 2, 16
    x = torch.randn(bsz, seqlen, args.dim)
    freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, seqlen)
    mask = torch.ones(seqlen, seqlen)
    
    # Forward attention
    output, kv = model.layers[0].attention(x, freqs_cis, mask)
    expected_shape = (bsz, seqlen, args.dim)
    assert output.shape == expected_shape, f"Attention output shape: {output.shape} != {expected_shape}"
    assert kv[0].shape[1] == args.n_heads, f"KV heads after expansion: {kv[0].shape[1]} != {args.n_heads}"
    print("GQA shapes verificadas correctamente.")

def test_lora_adapter():
    args = ModelArgs(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=1000, max_seq_len=64, lora_r=4, lora_alpha=32.0, lora_dropout=0.1)
    model = TinyThinker(args)
    assert isinstance(model.layers[0].attention.wq, LoRALinear), "LoRA no se aplicó a la proyección WQ"
    assert model.layers[0].attention.wq.r == 4, "Rank de LoRA incorrecto"

    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, "No hay parámetros entrenables para LoRA"

    tokens = torch.randint(0, args.vocab_size, (1, 8))
    logits = model(tokens)
    assert logits.shape == (1, 8, args.vocab_size)
    print("LoRA adapter verificado correctamente.")

def test_tokenizer_roundtrip():
    """Verificar encode -> decode ≈ original (ByteLevel BPE puede añadir espacio inicial)."""
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Tokenizer no encontrado, saltando test.")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    test_texts = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Python is great for ML",
    ]
    for test_text in test_texts:
        tokens = tokenizer.encode(test_text).ids
        decoded = tokenizer.decode(tokens).strip()
        assert decoded == test_text, f"Roundtrip failed: '{decoded}' != '{test_text}'"
    print("Tokenizer roundtrip verificado correctamente.")

def test_data_loading():
    """Verificar que get_batch devuelve shapes correctos."""
    # Simular data loading (sin archivo real)
    seq_len = 16
    batch_size = 4
    vocab_size = 1000
    data = torch.randint(0, vocab_size, (1000,))  # Simular datos tokenizados
    
    def get_batch(data):
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+1+seq_len] for i in ix])
        return x, y
    
    x, y = get_batch(data)
    assert x.shape == (batch_size, seq_len), f"X shape: {x.shape}"
    assert y.shape == (batch_size, seq_len), f"Y shape: {y.shape}"
    print("Data loading shapes verificadas correctamente.")

def test_kv_cache():
    """Verificar que KV-cache incremental produce el mismo último logit que el forward completo."""
    args = ModelArgs(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=1000, max_seq_len=64)
    model = TinyThinker(args)
    
    bsz, seqlen = 1, 16
    tokens = torch.randint(0, args.vocab_size, (bsz, seqlen))
    
    # Sin cache
    logits_no_cache = model(tokens)
    
    # Con cache, procesamos un prefijo y luego continuamos token a token.
    prefix_len = 8
    prefix = tokens[:, :prefix_len]
    suffix = tokens[:, prefix_len:]

    with torch.no_grad():
        _, past_key_values = model(prefix, use_cache=True)
        last_logits = None
        for i in range(suffix.size(1)):
            token = suffix[:, i:i+1]
            outputs = model(token, past_key_values=past_key_values, use_cache=True)
            if isinstance(outputs, tuple):
                logit, past_key_values = outputs
            else:
                logit = outputs
            last_logits = logit

    assert last_logits is not None, "No se generaron logits con KV-cache incremental"
    assert last_logits.shape == (bsz, 1, args.vocab_size), f"Logits finales con cache: {last_logits.shape}"
    diff = (logits_no_cache[:, -1, :] - last_logits[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"KV-cache incremental no coincide con forward completo: diff={diff}"
    assert not torch.isnan(last_logits).any(), "Incremental logits contain NaN"
    print("KV-cache verificado correctamente (incremental vs completo).")


def test_generate_text_prefills_prompt(monkeypatch):
    from scripts.eval import generate_text

    calls = []

    class DummyModel:
        def eval(self):
            return self

        def __call__(self, tokens, past_key_values=None, use_cache=False):
            calls.append((tuple(tokens.shape), past_key_values is not None, use_cache))
            vocab_size = 8
            logits = torch.zeros(tokens.size(0), tokens.size(1), vocab_size)
            logits[:, -1, 3] = 10.0
            next_past = ("cached",)
            if use_cache:
                return logits, next_past
            return logits

    def fake_multinomial(probs, num_samples=1):
        return torch.zeros((probs.size(0), num_samples), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    model = DummyModel()
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    output = generate_text(model, None, input_ids, max_new_tokens=1, temperature=1.0, device="cpu", top_k=None)

    assert output.shape[1] == 4
    assert calls[0] == ((1, 3), False, True)
    assert calls[1] == ((1, 1), True, True)
    assert output[0, -1].item() == 0

if __name__ == "__main__":
    test_model_forward()
    test_rotary_embeddings()
    test_gqa_shapes()
    test_tokenizer_roundtrip()
    test_data_loading()
    test_kv_cache()
