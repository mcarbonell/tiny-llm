import json
import os

DATASET_PATH = 'data/tool_dataset_real.json'
SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]\n"

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

count = 0
for item in data:
    if not item['text'].startswith('[SYSTEM]'):
        item['text'] = SYSTEM_TEXT + item['text']
        count += 1

with open(DATASET_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated {count} items in {DATASET_PATH}")
