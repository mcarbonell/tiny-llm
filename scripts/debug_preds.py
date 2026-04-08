import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker
from tokenizers import Tokenizer

CKPT_PATH = "checkpoints/ckpt_finetuned.pt"
TOKENIZER_PATH = "model/tokenizer.json"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
model = TinyThinker(checkpoint['args'])
model.load_state_dict(checkpoint['model'])
model.eval()

prompt = "[SYSTEM] You are TinyThinker [/SYSTEM]\nUser: Hello\nAssistant: "
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
