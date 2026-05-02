import os
import requests
import json
import time
import argparse
import random
import re
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise ValueError("❌ NVIDIA_API_KEY not found in .env file")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = API_KEY
)

MODEL_NAME = "minimaxai/minimax-m2.7"
RATE_LIMIT_DELAY = 1.7 # ~35 RPM safety margin

# Variety Flavors
FLAVORS = [
    "forest animals", "outer space", "ocean creatures", "kitchen tools", 
    "magical kingdom", "school classroom", "garden bugs", "city traffic",
    "stuffed toys", "family dinner", "superheroes", "dinosaurs",
    "pirate treasure", "robots and gadgets", "sports game", "farm life"
]

# Presentation Formats
FORMATS = [
    "narrative-based story", 
    "direct list of facts/clues", 
    "dialogue between two characters", 
    "set of explicit if-then rules", 
    "short situational mystery"
]

# Reasoning Skills targeting specific cognitive paths
REASONING_SKILLS_BY_LEVEL = {
    0: ["classification", "ordering", "matching"],
    1: ["classification", "ordering", "matching", "arithmetic", "spatial"],
    2: ["ordering", "logic", "temporal", "causal", "arithmetic"],
    3: ["logic", "temporal", "causal", "probabilistic", "strategic"],
    4: ["logic", "probabilistic", "strategic", "causal"],
    5: ["strategic", "probabilistic", "logic", "causal"],
    6: ["strategic", "probabilistic", "logic"]
}

# Level Configuration
LEVEL_SPEC = {
    0: {"name": "Foundation", "audience": "children ages 3 to 4", "focus": "Basic categorization, colors, sizes, and counting up to 5."},
    1: {"name": "Concrete Early", "audience": "children ages 5 to 7", "focus": "1-step cause and effect, simple arithmetic (<20), and ordering."},
    2: {"name": "Concrete Advanced", "audience": "children ages 8 to 10", "focus": "Multi-step logic (3-4 entities), transitive relations, and simple negation."},
    3: {"name": "Pre-teen Structured", "audience": "children ages 11 to 13", "focus": "Proportional reasoning, complex schedules, and inconsistency detection."},
    4: {"name": "Formal Operations", "audience": "teenagers ages 14 to 16", "focus": "Abstraction, counterfactual reasoning, simple optimization, and systems."},
    5: {"name": "Adult Abstract", "audience": "young adults ages 17 to 20", "focus": "Multi-agent reasoning, Bayesian inference, complex optimization, and trade-off analysis."},
    6: {"name": "Expert Meta-Reasoning", "audience": "adults 21+", "focus": "Meta-cognition, counterfactual analysis, epistemology, and reasoning about reasoning."}
}

# Topics extracted from previous script
TOPICS = {
    0: ["identifying the color of common things", "sorting big vs small", "counting up to 5"],
    2: ["transitive logic (A > B, B > C)", "tracking items movement", "simple schedules", "math word problems"],
    # ... more can be added if needed, using random choice for now
}

def get_system_prompt(level):
    spec = LEVEL_SPEC[level]
    return f"""You are creating logic puzzles for {spec['audience']} to develop reasoning skills.

CRITICAL REQUIREMENTS:
1. Question must require genuine logical thinking, not just recall.
2. The <think> section must show the complete reasoning process.
3. IMPORTANT: You must explicitly state the mathematical or logical laws used in the thinking section using the format [Law: Name of the Law]. 
   Examples: [Law: Transitivity], [Law: Modulus], [Law: Substitution], [Law: Syllogism], [Law: Addition], [Law: Pattern Recognition].
4. Start immediately with 'Question:' without any introductory phrases.

REQUIRED FORMAT:
Question: [The puzzle or riddle]
<think> [Step-by-step reasoning. Explicitly include [Law: ...] tags.] </think>
Answer: [Final answer]

RULES:
- No markdown, no bold, no lists in the Question.
- ALWAYS include <think> and </think> tags.
- For level {level}, focus on: {spec['focus']}
- Age-appropriate simple words.
"""

def validate_sample(text, level):
    if not ("Question:" in text and "<think>" in text and "</think>" in text and "Answer:" in text):
        return False
    try:
        thought = re.search(r"<think>(.*?)</think>", text, re.DOTALL).group(1).strip()
    except AttributeError:
        return False
    if len(thought) < 20: 
        return False
    if level > 1 and "[Law:" not in thought:
        return False
    return True

def get_sample(level, topic, flavor, skill, format_style):
    prompt = get_system_prompt(level)
    user_instruction = (
        f"Create a Level {level} logic riddle about: {topic}.\n"
        f"Context: {flavor}. Skill: {skill}. Style: {format_style}."
    )
    
    try:
        completion = client.chat.completions.create(
          model=MODEL_NAME,
          messages=[
              {"role": "system", "content": prompt},
              {"role": "user", "content": user_instruction}
          ],
          temperature=0.8,
          max_tokens=1000
        )
        content = completion.choices[0].message.content
        if validate_sample(content, level):
            return content
        return None
    except Exception as e:
        print(f"\n[API Error] {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=range(0, 7), default=2)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    level_name = LEVEL_SPEC[args.level]['name'].lower().replace(" ", "_")
    output_path = args.output or f"data/raw/synthetic_logic_nim_{level_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🚀 Generating Level {args.level} via NVIDIA NIM ({MODEL_NAME}). Target: {args.target}")
    
    success_count = 0
    pbar = tqdm(total=args.target)
    
    # Predefined topics pool if not provided in LEVEL_SPEC
    topics_pool = [
        "transitive relations", "temporal order", "arithmetic word problems", 
        "negation and exclusion", "pattern completion", "causal chains"
    ]

    while success_count < args.target:
        topic = random.choice(topics_pool)
        flavor = random.choice(FLAVORS)
        format_style = random.choice(FORMATS)
        skill = random.choice(REASONING_SKILLS_BY_LEVEL[args.level])
        
        res = get_sample(args.level, topic, flavor, skill, format_style)
        if res:
            with open(output_path, "a", encoding="utf-8") as f:
                data = {"text": res, "level": args.level, "model": MODEL_NAME}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            success_count += 1
            pbar.update(1)
            time.sleep(RATE_LIMIT_DELAY)
        else:
            time.sleep(2)

if __name__ == "__main__":
    main()
