"""
scripts/train_master_tokenizer_v2.py — The Birth of the 32k Dict

Entrena un nuevo tokenizador BPE de 32,768 tokens usando el corpus maestro.
Incluye todos los tokens especiales de la arquitectura COGA y V197.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

def train_tokenizer():
    corpus_path = "data/raw/master_corpus_v2.txt"
    output_path = "model/tokenizer_v2_32k.json"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: No se encuentra el corpus en {corpus_path}. Ejecuta build_master_corpus_v2.py primero.")
        return

    # Definir tokens especiales
    special_tokens = [
        "<|endoftext|>",
        "[SYSTEM]",
        "[/SYSTEM]",
        "User:",
        "Assistant:",
        "<think>",
        "</think>",
        "[Law:",
        "WRITE(",
        "READ(",
        "EDIT(",
        "COMMIT(",
        "DELETE(",
        "recall(",
        "remember(",
        "lookup(",
        "verify(",
        "execute_code(",
        "<calc>",
        "</calc>",
        "<TOOL_CALL>",
        "</TOOL_CALL>",
    ]

    print(f"🛠️ Entrenando Tokenizador BPE (Vocab=32,768)...")
    
    # Inicializar modelo BPE
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    # Configurar entrenador
    trainer = BpeTrainer(
        vocab_size=32768,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )

    # Entrenar
    files = [corpus_path]
    tokenizer.train(files, trainer)

    # Añadir post-procesamiento para manejar el byte-level de forma limpia
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    # Guardar
    os.makedirs("model", exist_ok=True)
    tokenizer.save(output_path)
    
    print(f"\n✅ Tokenizador Maestro V2 guardado en: {output_path}")
    print(f"📊 Tamaño final del vocabulario: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_tokenizer()
