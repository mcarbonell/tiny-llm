import os
import requests
import json
import time
import argparse
import random
import re
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemma-4-31b-it:free" 

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

# Variety Flavors to inject randomness into themes
FLAVORS = [
    "forest animals", "outer space", "ocean creatures", "kitchen tools", 
    "magical kingdom", "school classroom", "garden bugs", "city traffic",
    "stuffed toys", "family dinner", "superheroes", "dinosaurs",
    "pirate treasure", "robots and gadgets", "sports game", "farm life"
]

# Presentation Formats to vary linguistic surface
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
    6: ["strategic", "probabilistic", "logic"]  # Meta-cognition
}

# Level Configuration
LEVEL_SPEC = {
    0: {
        "name": "Foundation",
        "audience": "children ages 3 to 4",
        "focus": "Basic categorization, colors, sizes, and counting up to 5.",
        "topics": [
            "identifying the color of common things or objects",
            "identifying objects or animals",
            "sorting things or animals by size (big vs small)",
            "counting items (up to 5)",
            "identifying which item doesn't belong",
            "basic relationships: inside or outside, up or down, far/near, before/after",
            "family relationships"
        ]
    },
    1: {
        "name": "Concrete Early",
        "audience": "children ages 5 to 7",
        "focus": "1-step cause and effect, simple arithmetic (<20), and ordering.",
        "topics": [
            "who arrived first",
            "who has more items in their collection",
            "simple addition or subtraction",
            "matching items by multiple attributes",
            "ordering three objects by height or weight",
            "predicting simple consequences"
        ]
    },
    2: {
        "name": "Concrete Advanced",
        "audience": "children ages 8 to 10",
        "focus": "Multi-step logic (3-4 entities), transitive relations, and simple negation.",
        "topics": [
            "transitive logic (who is taller: A > B, B > C)",
            "tracking items through movement (A gave to B, B gave to C)",
            "simple schedules (if music is before art, and art is after gym...)",
            "math word problems with 2 steps (buying and spending change)",
            "detecting simple contradictions in a set of facts",
            "finding the odd-one-out with abstract reasoning"
        ]
    },
    3: {
        "name": "Pre-teen Structured",
        "audience": "children ages 11 to 13",
        "focus": "Proportional reasoning, complex schedules, and inconsistency detection.",
        "topics": [
            "unit rates (if 3 items cost $X, how much for Y?)",
            "complex schedules with overlapping constraints",
            "basic combinations (how many outfits with X tops and Y pants)",
            "detecting logic flaws in a short paragraph",
            "logical sequences with non-obvious patterns",
            "spatial reasoning: reconstructing a scene from clues"
        ]
    },
    4: {
        "name": "Formal Operations",
        "audience": "teenagers ages 14 to 16",
        "focus": "Abstraction, counterfactual reasoning, simple optimization, and systems.",
        "topics": [
            "counterfactual thinking (what if a premise was false?)",
            "simple optimization (shortest path through 4 points)",
            "systems of equations in word form",
            "evaluating the strength of a logical argument",
            "basic probability and risk assessment",
            "understanding basic trade-offs: cost vs speed"
        ]
    },
    5: {
        "name": "Adult Abstract",
        "audience": "young adults ages 17 to 20",
        "focus": "Multi-agent reasoning, Bayesian inference, complex optimization, and trade-off analysis.",
        "topics": [
            "multi-person negotiation with incomplete information",
            "Bayesian updating with new evidence",
            "resource allocation under multiple constraints",
            "identifying causal vs. correlational relationships",
            "analyzing incentives in game theory scenarios",
            "system failure diagnosis with multiple potential causes",
            "prisoner's dilemma and similar strategic situations",
            "evaluating arguments with statistical evidence"
        ]
    },
    6: {
        "name": "Expert Meta-Reasoning",
        "audience": "adults 21+",
        "focus": "Meta-cognition, counterfactual analysis, epistemology, and reasoning about reasoning.",
        "topics": [
            "identifying missing assumptions in an argument",
            "evaluating if sufficient information exists to solve a problem",
            "comparing strengths of competing explanations",
            "recognizing cognitive biases in reasoning patterns",
            "designing experiments to test hypotheses",
            "counterfactual impact analysis in complex systems",
            "reasoning under radical uncertainty (unknown unknowns)",
            "epistemic status assessment (how confident should we be?)"
        ]
    }
}

def get_system_prompt(level):
    spec = LEVEL_SPEC[level]

    # Configuración por nivel
    level_config = {
        0: {"think_length": "2-3 sentences", "vocab": "very simple words"},
        1: {"think_length": "3-4 sentences", "vocab": "simple everyday language"},
        2: {"think_length": "4-5 sentences", "vocab": "clear explanatory language"},
        3: {"think_length": "5-6 sentences", "vocab": "precise technical terms when needed"},
        4: {"think_length": "6-8 sentences", "vocab": "formal but accessible language"},
        5: {"think_length": "8-12 sentences", "vocab": "advanced analytical language"},
        6: {"think_length": "10-15 sentences", "vocab": "epistemic and meta-cognitive language"},
    }
    
    config = level_config.get(level, {"think_length": "6-10 sentences", "vocab": "formal but accessible language"})

    return f"""You are creating logic puzzles for {spec['audience']} to develop reasoning skills.

CRITICAL REQUIREMENTS:
1. Question must require genuine logical thinking, not just recall.
2. The <think> section must show the complete reasoning process that leads to the answer.
3. Use {config['vocab']}.
4. Make the thinking section about {config['think_length']}.

REQUIRED FORMAT:
Question: [The puzzle or riddle]
<think> [Step-by-step reasoning. Show how you combine clues, eliminate possibilities, or calculate.] </think>
Answer: [Final answer]

ADDITIONAL RULES:
- No markdown, no bold, no lists in the Question.
- ALWAYS include <think> and </think> tags.
- The thinking must be self-contained; don't refer to external knowledge.
- If the puzzle involves multiple steps, show each step clearly.
- The answer should be concise but complete.
- Age-appropriate simple words and short sentences.
- For level {level}, focus on: {spec['focus']}
- Start immediately with 'Question:' without any introductory phrases.
- Make it challenging but solvable for {spec['audience']}
- If it cannot be uniquely determined, set Answer: Not enough information.
"""

def validate_sample(text, level):
    """Validation rules to ensure high-quality reasoning traces."""
    # Check for basic tags
    if not ("Question:" in text and "<think>" in text and "</think>" in text and "Answer:" in text):
        return False
    
    # Extract thought process
    try:
        thought = re.search(r"<think>(.*?)</think>", text, re.DOTALL).group(1).strip()
    except AttributeError:
        return False
        
    # Check minimum thought length
    if len(thought) < 20: 
        return False
        
    # Check for reasoning connectors (especially for levels > 0)
    reasoning_connectors = ["because", "therefore", "since", "so", "thus", "if", "then", "leads to", "conclude", "implies"]
    if level > 0 and not any(conn in thought.lower() for conn in reasoning_connectors):
        return False
    
    return True

def get_sample(level, topic, flavor, skill, format_style):
    prompt = get_system_prompt(level)
    
    user_instruction = (
        f"Create a Level {level} logic riddle/puzzle about: {topic}.\n"
        f"Context/Theme: {flavor}.\n"
        f"Target Skill: Use {skill} logic.\n"
        f"Presentation Style: {format_style}.\n"
        f"Ensure this puzzle is original and requires careful thinking. "
        f"Random entropy seed: {random.randint(1, 10000)}."
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_instruction}
        ],
        "max_tokens": 1200 if level > 4 else 800,
        "temperature": 0.8 # Diversity first
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
                if validate_sample(content, level):
                    return content
        return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data for TinyThinker.")
    parser.add_argument("--level", type=int, choices=range(0, 7), default=0, help="Cognitive level to generate (0-6)")
    parser.add_argument("--target", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    
    args = parser.parse_args()
    
    level = args.level
    target = args.target
    level_name = LEVEL_SPEC[level]['name'].lower().replace(" ", "_")
    output_path = args.output or f"data/raw/synthetic_logic_{level_name}.jsonl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating Level {level} ({LEVEL_SPEC[level]['name']}) focused data. Target: {target} samples.")
    
    start_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            start_count = sum(1 for _ in f)
    
    success_count = start_count
    pbar = tqdm(total=target, initial=start_count)
    
    while success_count < target:
        topic = random.choice(LEVEL_SPEC[level]['topics'])
        flavor = random.choice(FLAVORS)
        format_style = random.choice(FORMATS)
        skill = random.choice(REASONING_SKILLS_BY_LEVEL[level])
        
        res = get_sample(level, topic, flavor, skill, format_style)
        
        if res:
            with open(output_path, "a", encoding="utf-8") as f:
                data = {
                    "text": res,
                    "level": level,
                    "level_name": LEVEL_SPEC[level]['name'],
                    "topic": topic,
                    "flavor": flavor,
                    "skill": skill,
                    "format": format_style
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            success_count += 1
            pbar.update(1)
            time.sleep(1) # Balance speed and stability
        else:
            time.sleep(5)
    
    print(f"Generation complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()
