import os
import requests
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemma-4-31b-it:free" 

OUTPUT_PATH_JSONL = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logic_rich.jsonl")
RAW_LOG_PATH = "logs/api_raw.log"

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

os.makedirs("logs", exist_ok=True)

# English Topics for TinyLogic Level 1-2
TOPICS = [
    "colors of farm animals",
    "who arrived first at the snack time",
    "the mystery of the lost toy in the garden",
    "who has the red soccer ball",
    "fruits inside a basket",
    "guessing a friendly animal by clues",
    "ordering three toys by size",
    "who lives in the blue house",
    "the shortest path to the playground",
    "sorting clothes by weather (sunny vs rainy)"
]

SYSTEM_PROMPT = """You are a teacher creating simple logic riddles for children in English.
Your goal is to provide high-quality training data for a reasoning model.

REQUIRED FORMAT:
Question: [The riddle here]
<think> [Step-by-step simple reasoning explaining why the answer is correct] </think>
Answer: [Final solution]

RULES:
1. Use ONLY English.
2. ALWAYS use <think> and </think> tags.
3. Make the riddle suitable for children ages 5 to 8.
4. Use short sentences and simple words.
5. The reasoning must be clear, short, and correct.
6. Avoid scary, sad, or dangerous situations.
7. No meta-talk. Start directly with 'Question:'.
"""

def get_sample(topic):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a simple logic riddle about: {topic}. Remember to use <think> tags."}
        ],
        "max_tokens": 500,
        "temperature": 0.5
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        
        # Save raw log for debugging
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.ctime()} ---\n{response.text}\n")
            
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
                # Validate presence of required tags
                if "Question:" in content and "<think>" in content and "</think>" in content:
                    return content
        return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH_JSONL), exist_ok=True)
    
    # Checkpoint (resume from existing lines)
    start_count = 0
    if os.path.exists(OUTPUT_PATH_JSONL):
        with open(OUTPUT_PATH_JSONL, "r", encoding="utf-8") as f:
            start_count = sum(1 for _ in f)
            
    success_count = start_count
    target = 500 # Target increased for English session
    print(f"Starting 'TinyLogic' generation in ENGLISH. Target: {target} samples.")
    
    pbar = tqdm(total=target, initial=start_count)
    
    while success_count < target:
        topic = TOPICS[success_count % len(TOPICS)]
        res = get_sample(topic)
        
        if res:
            with open(OUTPUT_PATH_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": res, "topic": topic}, ensure_ascii=False) + "\n")
            success_count += 1
            pbar.update(1)
            time.sleep(3)
        else:
            print("\nFormat error or API issue. retrying...")
            time.sleep(5)

if __name__ == "__main__":
    main()
