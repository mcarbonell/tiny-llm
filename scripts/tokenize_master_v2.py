"""
scripts/tokenize_master_v2.py — High-Performance Tokenizer for 32k Vocab

Tokeniza el Master Corpus V2 usando el nuevo Tokenizador Maestro de 32k.
Genera el archivo binario final para el entrenamiento de los modelos Mega-Midi.
"""

import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse

def tokenize_corpus():
    input_path = "data/raw/master_corpus_v2.txt"
    output_path = "data/train_v2_32k.bin"
    tokenizer_path = "model/tokenizer_v2_32k.json"

    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: No se encuentra el tokenizador en {tokenizer_path}")
        return

    print(f"📂 Cargando tokenizador desde {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Buscamos el token de fin de texto
    eos_token = "<|endoftext|>"
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        print(f"⚠️ Warning: Token {eos_token} no encontrado, usando ID 0")
        eos_id = 0

    print(f"⚡ Tokenizando {input_path} -> {output_path}...")
    
    # Usamos un buffer para no saturar la RAM y escribir en bloques
    buffer = []
    chunk_size = 500_000 # Número de tokens antes de escribir a disco
    total_tokens = 0
    
    # Procesar línea a línea para ser eficientes con archivos grandes (>400MB)
    with open(input_path, "r", encoding="utf-8") as f, open(output_path, "wb") as bin_f:
        # Estimación de líneas para la barra de progreso
        num_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
        f.seek(0) # Volver al inicio
        
        for line in tqdm(f, total=num_lines, desc="Procesando líneas"):
            line = line.strip()
            if not line: continue
            
            # Tokenizar línea
            ids = tokenizer.encode(line).ids
            buffer.extend(ids)
            buffer.append(eos_id) # Añadir separador
            
            if len(buffer) >= chunk_size:
                # Escribir bloque en formato uint16 (soporta hasta 65k vocab)
                bin_f.write(np.array(buffer, dtype=np.uint16).tobytes())
                total_tokens += len(buffer)
                buffer = []
        
        # Escribir lo que quede en el buffer
        if buffer:
            bin_f.write(np.array(buffer, dtype=np.uint16).tobytes())
            total_tokens += len(buffer)

    print(f"\n✅ DATASET MAESTRO V2 COMPLETADO")
    print(f"----------------------------------------")
    print(f"Archivo: {output_path}")
    print(f"Tokens totales: {total_tokens / 1e6:.2f} Millones")
    print(f"Tamaño final: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    print(f"----------------------------------------")

if __name__ == "__main__":
    tokenize_corpus()
