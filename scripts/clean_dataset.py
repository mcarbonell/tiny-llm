import json
import re

DATA_PATH = "data/tool_dataset_real.json"
CLEAN_PATH = "data/tool_dataset_clean.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

clean_data = []
for item in data:
    text = item['text']
    # Remove <thought>...</thought> blocks (they are not special tokens)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    # Fix spacing issues if any
    text = re.sub(r'\n+', '\n', text).strip()
    clean_data.append({"text": text})

with open(CLEAN_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=4, ensure_ascii=False)

print(f"Cleaned {len(clean_data)} examples. Saved to {CLEAN_PATH}")
print("\nExample 1:")
print(clean_data[0]['text'][:500])
