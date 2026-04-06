import numpy as np
import os

files = ["data/train.bin", "data/wiki.bin"]
out   = "data/train_combined.bin"

with open(out, "wb") as fout:
    for path in files:
        arr = np.memmap(path, dtype=np.uint16, mode='r')
        fout.write(arr.tobytes())
        print(f"  Añadido: {path} ({len(arr)/1e6:.1f}M tokens)")

print(f"\nCorpus combinado guardado en: {out}")
