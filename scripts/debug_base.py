import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker
from tokenizers import Tokenizer

def resolve_checkpoint():
    candidates = [
        "checkpoints/ckpt_base_corpus305M_v2.pt",
        "checkpoints/ckpt_base_300M_v2.pt",
        "checkpoints/old/ckpt_first_pretraining.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No se encontró un checkpoint base válido.")

CKPT_PATH = resolve_checkpoint()
TOKENIZER_PATH = "model/tokenizer.json"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
model = TinyThinker(checkpoint['args'])
model.load_state_dict(checkpoint['model'])
model.eval()

prompt = "Once upon a time"
ids = tokenizer.encode(prompt).ids
x = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    logits = model(x)
    logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 5)

print(f"Prompt: {prompt}")
print("\nTop 5 predictions:")
for i in range(5):
    token_id = top_ids[0, i].item()
    prob = top_probs[0, i].item()
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"ID {token_id:5d} | Prob {prob:.4f} | Token: {repr(token_text)}")
