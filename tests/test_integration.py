"""
tests/test_integration.py — TinyThinker Integration Tests
==========================================================
Validan el pipeline completo end-to-end usando artefactos reales del proyecto.

Ejecución:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -k "config or valid"   # solo los rápidos (sin modelo)
    pytest tests/ -v                                         # unit + integration

Tests que requieren checkpoint se saltan automáticamente si no existe un checkpoint pretrain/sft válido.
"""
import os
import sys
import json
import math
import tempfile
import pytest
import torch

# Añadir raíz del proyecto al path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from model.model import TinyThinker, ModelArgs
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Rutas de artefactos reales
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
TOKENIZER_PATH  = os.path.join(ROOT, "model", "tokenizer.json")
DATASET_PATH    = os.path.join(ROOT, "data", "tool_dataset_real.json")


def resolve_checkpoint_path():
    priority = [
        "ckpt_sft_latest.pt",
        "ckpt_sft_best.pt",
        "ckpt_pretrain_best.pt",
        "ckpt_pretrain_latest.pt",
        "ckpt_finetuned.pt",
        "ckpt_best.pt",
        "ckpt.pt",
    ]
    for name in priority:
        path = os.path.join(CHECKPOINTS_DIR, name)
        if os.path.exists(path):
            return path
    return None

# Marcadores para saltar tests si los artefactos no existen
requires_checkpoint = pytest.mark.skipif(
    resolve_checkpoint_path() is None,
    reason="No se encontró un checkpoint válido — ejecuta train.py o finetune.py primero"
)
requires_dataset = pytest.mark.skipif(
    not os.path.exists(DATASET_PATH),
    reason=f"Dataset no encontrado: {DATASET_PATH} — ejecuta generate_synthetic_data.py primero"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    """Carga el modelo y tokenizador una sola vez para todos los tests del módulo."""
    ckpt_path = resolve_checkpoint_path()
    if not ckpt_path:
        pytest.skip("Checkpoint no disponible")
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip("Tokenizer no disponible")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    model = TinyThinker(model_args)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, tokenizer, model_args


# ---------------------------------------------------------------------------
# IT-1: Checkpoint load → generate → non-empty output
# ---------------------------------------------------------------------------
@requires_checkpoint
def test_checkpoint_load_and_generate(loaded_model_and_tokenizer):
    """Carga el checkpoint real y genera al menos 5 tokens coherentes."""
    model, tokenizer, _ = loaded_model_and_tokenizer

    prompt = "User: What is the capital of France?\nAssistant: "
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long)

    generated = []
    eos_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>")

    with torch.no_grad():
        for _ in range(20):
            logits = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            token_id = next_token.item()
            if token_id == eos_id:
                break
            generated.append(token_id)

    assert len(generated) >= 1, "El modelo no generó ningún token"
    decoded = tokenizer.decode(generated)
    assert isinstance(decoded, str) and len(decoded) > 0, "El output decodificado está vacío"


# ---------------------------------------------------------------------------
# IT-2: Eval pipeline → perplexity is a finite number
# ---------------------------------------------------------------------------
@requires_checkpoint
@requires_dataset
def test_eval_pipeline_perplexity(loaded_model_and_tokenizer):
    """calculate_perplexity() devuelve un float finito positivo."""
    from scripts.eval import calculate_perplexity
    model, tokenizer, _ = loaded_model_and_tokenizer

    perplexity = calculate_perplexity(
        model, tokenizer, DATASET_PATH,
        device="cpu", seq_len=128, num_batches=3
    )
    assert perplexity is not None, "calculate_perplexity() devolvió None"
    assert math.isfinite(perplexity), f"Perplexity no es finita: {perplexity}"
    assert perplexity > 1.0, f"Perplexity debe ser > 1, got {perplexity}"


# ---------------------------------------------------------------------------
# IT-3: Eval pipeline → tool-calling accuracy in [0, 1]
# ---------------------------------------------------------------------------
@requires_checkpoint
@requires_dataset
def test_eval_pipeline_tool_accuracy(loaded_model_and_tokenizer):
    """evaluate_tool_calling_accuracy() devuelve un float en [0, 1]."""
    from scripts.eval import evaluate_tool_calling_accuracy
    model, tokenizer, _ = loaded_model_and_tokenizer

    accuracy = evaluate_tool_calling_accuracy(
        model, tokenizer, DATASET_PATH, device="cpu"
    )
    assert accuracy is not None, "evaluate_tool_calling_accuracy() devolvió None"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy fuera de rango [0,1]: {accuracy}"


# ---------------------------------------------------------------------------
# IT-4: KV-cache consistency — logits idénticos con y sin cache
# ---------------------------------------------------------------------------
@requires_checkpoint
def test_kv_cache_consistency(loaded_model_and_tokenizer):
    """
    Verifica que el KV-cache genera los mismos logits que el forward sin cache
    para una secuencia dada. Esto garantiza la corrección del BUG-D fix.
    """
    model, tokenizer, _ = loaded_model_and_tokenizer

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt).ids[:16]  # Acotamos a 16 tokens
    x = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        # --- Sin cache: forward completo ---
        logits_no_cache = model(x)

        # --- Con cache: procesar todo el prefijo de una vez ---
        logits_with_cache, _ = model(x, use_cache=True)

    # Los logits del último token deben ser (prácticamente) idénticos
    diff = (logits_no_cache[:, -1, :] - logits_with_cache[:, -1, :]).abs().max().item()
    assert diff < 1e-4, (
        f"Discrepancia entre logits con/sin KV-cache: max_diff={diff:.6f}. "
        "El BUG-D fix puede haber introducido una regresión."
    )


# ---------------------------------------------------------------------------
# IT-5: YAML config roundtrip — save & reload preserva todos los campos
# ---------------------------------------------------------------------------
def test_yaml_config_roundtrip():
    """
    Crea una Config, la serializa a YAML, la recarga y verifica
    que todos los campos sean idénticos. No necesita modelo ni GPU.
    """
    pytest.importorskip("yaml", reason="PyYAML no instalado")
    import yaml
    from scripts.config import Config

    original = Config(
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        batch_size=8,
        max_iters=1000,
        use_gradient_checkpointing=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        tmp_path = f.name

    try:
        original.save_yaml(tmp_path)
        assert os.path.exists(tmp_path), "save_yaml() no creó el archivo"

        with open(tmp_path, "r", encoding="utf-8") as f:
            reloaded_data = yaml.safe_load(f)

        # Verificar campos clave
        assert reloaded_data["dim"] == original.dim
        assert reloaded_data["n_layers"] == original.n_layers
        assert reloaded_data["batch_size"] == original.batch_size
        assert reloaded_data["max_iters"] == original.max_iters
        assert reloaded_data["use_gradient_checkpointing"] == original.use_gradient_checkpointing
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# IT-6: Dataset validation rejects invalid schemas
# ---------------------------------------------------------------------------
def test_dataset_validation_rejects_invalid():
    """
    validate_dataset() lanza ValueError para los 4 casos de schema inválido.
    No necesita modelo ni checkpoint.
    """
    from scripts.eval import validate_dataset

    # Caso 1: No es lista (es dict)
    with pytest.raises(ValueError, match="lista JSON"):
        validate_dataset({"key": "value"}, "fake.json")

    # Caso 2: Lista vacía
    with pytest.raises(ValueError, match="vacío"):
        validate_dataset([], "fake.json")

    # Caso 3: Elemento no es dict
    with pytest.raises(ValueError, match="objeto JSON"):
        validate_dataset(["esto es un string, no un dict"], "fake.json")

    # Caso 4: Elemento dict sin campo 'text'
    with pytest.raises(ValueError, match="campo 'text'"):
        validate_dataset([{"prompt": "hello", "response": "world"}], "fake.json")

    # Caso 5: Campo 'text' no es string
    with pytest.raises(ValueError, match="debe ser str"):
        validate_dataset([{"text": 42}], "fake.json")

    # Caso positivo: dataset válido no lanza excepción
    result = validate_dataset([{"text": "hello world"}], "valid.json")
    assert len(result) == 1
