import sys
import os
from tokenizers import Tokenizer, decoders

# Reconfigurar stdout para evitar crasheos de encoding en Windows
sys.stdout.reconfigure(encoding='utf-8')

tokenizer_path = "model/tokenizer_v2_32k.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

print("Tokenizer decoder:", tokenizer.decoder)

# Si el decodificador es None, podemos acoplar el decodificador de ByteLevel
if tokenizer.decoder is None:
    tokenizer.decoder = decoders.ByteLevel()
    print("Attached ByteLevel decoder successfully!")

text = "Para calcular la hipotenusa de un triángulo rectángulo"
ids = tokenizer.encode(text).ids
print("IDs:", ids)
print("Normal decode (with ByteLevel decoder):", tokenizer.decode(ids))
