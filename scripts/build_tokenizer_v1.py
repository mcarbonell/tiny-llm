import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def build_tokenizer():
    # 1. Definir los datasets para el entrenamiento del vocabulario
    # Usamos una mezcla de los nuevos archivos para asegurar cobertura
    files = [
        "data/raw/fineweb_edu.txt",
        "data/raw/cosmopedia.txt",
        "data/raw/tinystories_v2.txt"
    ]
    
    # Verificar que existen
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Error: No se encuentra {f}. Asegúrate de haber descargado los datasets.")
            return

    # 2. Configurar el Tokenizador BPE (Byte-Pair Encoding)
    # Usamos ByteLevel para evitar problemas con caracteres desconocidos
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Definir Tokens Especiales
    # Estos tokens se reservan y no se dividen
    special_tokens = [
        "<|endoftext|>", # Separador base
        "<pad>",         # Padding para batches
        "<eos>",         # Fin de secuencia
        "[SYSTEM]",      # Inicio de prompt de sistema
        "[/SYSTEM]",     # Fin de prompt de sistema
        "User:",         # Marcador de usuario
        "Assistant:",    # Marcador de asistente
        "<TOOL_CALL>",   # Inicio de llamada a herramienta
        "</TOOL_CALL>",  # Fin de llamada a herramienta
        "<TOOL_RESULT>", # Inicio de resultado de herramienta
        "</TOOL_RESULT>",# Fin de resultado de herramienta
        "<think>",       # Inicio de bloque de pensamiento (CoT)
        "</think>",      # Fin de bloque de pensamiento (CoT)
        # Tokens de Curriculum (Niveles)
        "<level_0>", "<level_1>", "<level_2>", "<level_3>", 
        "<level_4>", "<level_5>", "<level_6>"
    ]

    # 4. Configurar el Entrenador
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. Entrenar
    print(f"--- Entrenando tokenizer_v1 con {len(files)} archivos ---")
    tokenizer.train(files, trainer)

    # 6. Post-procesamiento
    # Configura cómo se añaden los tokens especiales automáticamente si fuera necesario
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 7. Guardar con versionado
    output_path = "model/tokenizer_v1.json"
    tokenizer.save(output_path)
    
    print(f"\n✅ Tokenizador guardado exitosamente en: {output_path}")
    print(f"Vocabulario final: {tokenizer.get_vocab_size()} tokens")
    
    # Prueba rápida
    test_text = "[SYSTEM] Eres TinyThinker. [/SYSTEM]\nUser: <level_2> ¿Cuánto es 2+2? <think> Cálculo simple </think> Assistant:"
    encoded = tokenizer.encode(test_text)
    print(f"\nPrueba de codificación:")
    print(f"Texto: {test_text}")
    print(f"Tokens: {encoded.tokens}")

if __name__ == "__main__":
    build_tokenizer()
