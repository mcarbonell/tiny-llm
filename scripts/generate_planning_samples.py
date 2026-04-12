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
RAW_LOG_PATH = "logs/api_raw.log"

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

# Domain Configuration
DOMAINS = {
    "household": {
        "audience": "general purpose assistant",
        "focus": "Physical tasks, household maintenance, and daily routines.",
        "topics": [
            "fixing a leaky faucet", "cleaning a kitchen after a big party", "organizing a messy garage",
            "planning a weekly meal prep", "caring for an indoor garden", "moving to a new apartment",
            "preparing for a winter storm", "hosting a backyard BBQ", "installing a smart thermostat",
            "setting up an emergency supplies kit", "deep cleaning an old rug", "organizing a home library",
            "repairing a broken fence panel", "setting up a home office for ergonomics", "planning a zero-waste kitchen transition",
            "winterizing a garden and pipes", "organizing a multi-generational family move", "setting up a compost system",
            "designing a small balcony garden", "performing a monthly home safety audit", "preparing for a long-term power outage",
            "planning a garage sale from scratch", "restoring an old wooden table", "cleaning solar panels safely",
            "organizing a loft or attic space", "preparing a guest room for a 1-week visit", "cleaning gutters before rain",
            "implementing a greywater recycling system", "training a puppy on household rules", "organizing a 10-year digital photo archive",
            "planning a cross-country move with a trailer", "deep cleaning laundry vents and dryer", "fixing a stuck sliding glass door",
            "building a raised garden bed", "setting up a home security network", "planning a holiday lights display",
            "organizing a community tool-sharing shed", "restoring a vintage bicycle", "planning a budget-friendly home renovation"
        ],
        "style": "Sequential natural language steps."
    },
    "technical": {
        "audience": "software engineer assistant",
        "focus": "Coding, debugging, system architecture, and technical workflows.",
        "topics": [
            "deploying a React app to production", "finding a memory leak in Python", "designing a REST API for a blog",
            "setting up a CI/CD pipeline", "migrating from SQL to NoSQL", "optimizing Docker image size",
            "refactoring a monolithic script", "implementing OAuth2 authentication", "debugging a race condition in C++",
            "setting up a Kubernetes cluster locally", "implementing a search engine with Elasticsearch",
            "auditing a web app for SQL injection", "migrating a legacy codebase to TypeScript", "setting up a data warehouse",
            "designing a real-time notification system", "optimizing a slow PostgreSQL query", "implementing blue-green deployment",
            "setting up a distributed locking system", "designing a rate limiter for an API", "hardening a Linux server",
            "setting up an ELK stack for logging", "implementing a custom compiler for a DSL", "building a blockchain explorer",
            "implementing a custom garbage collector", "designing high-availability DB clusters", "auditing smart contracts",
            "setting up GitOps with ArgoCD", "refactoring high-latency end-points", "designing a custom OS file system",
            "implementing Raft consensus algorithm", "building real-time collaborative editors", "setting up a network honeypot",
            "developing kernel-level drivers", "designing a global CDN architecture", "optimizing video streaming protocols",
            "implementing zero-knowledge proofs", "configuring a BGP router", "scaling a microservices mesh"
        ],
        "style": "Technical steps, including use of execute_code() or verify() where appropriate."
    },
    "creative": {
        "audience": "creative collaborator",
        "focus": "Writing, event planning, social scenarios, and complex coordination.",
        "topics": [
            "planning a surprise 30th birthday", "writing a short sci-fi plot", "organizing a charity auction",
            "mediating a roommate conflict", "designing a boutique store layout", "launching a niche podcast",
            "creating a marketing campaign", "planning a 10-day trip to Europe", "writing a pilot for a TV series",
            "organizing a 3-day music festival", "designing a world-building wiki for a novel", "launching a board game on Kickstarter",
            "curating a local art exhibition", "planning a destination wedding", "designing a multi-player escape room",
            "orchestrating a community garden launch", "writing a brand identity guide", "planning a viral social media stunt",
            "organizing a professional networking mixer", "designing a complex tabletop RPG campaign", "launching a non-profit awareness month",
            "directing a short film on a budget", "planning a multi-stage puzzle for D&D", "curating a soundtrack for a series",
            "writing a grant proposal for art", "designing a modular clothing collection", "planning interactive art installations",
            "conceptualizing corporate team building", "directing a live-streamed theatrical event", "designing a book cover and layout",
            "launching an indie game's PR strategy", "organizing an international film festival", "managing a 50-person design team"
        ],
        "style": "Nuanced steps considering social and creative constraints."
    },
    "agentic": {
        "audience": "Cognitive Operating System (COGA)",
        "focus": "Internal operations, memory management, and meta-cognition.",
        "topics": [
            "finding a user preference in logs", "summarizing operational logs", "optimizing scratchpad usage",
            "consolidating memories to long-term storage", "verifying contradictory info with tools",
            "planning a long-term reasoning chain", "cleaning up old memory slots", "preparing a morning brief",
            "detecting recursive logic loops", "partitioning memory for search speed", "simulating scenarios for safety",
            "auditing confidence scores of sources", "refactoring a 50-step plan into modules", "allocating compute budget (Fast vs Slow)",
            "recovering from a tool execution failure", "merging overlapping memory fragments", "deciding when to switch to tool usage",
            "evaluating the impact of new data on beliefs", "bi-annual self-reflection and bias audit", "synchronizing states across context windows",
            "identifying latent dependencies in the scratchpad", "preparing a diagnostic report for the system architect",
            "auditing past reasoning for logical fallacies", "optimizing the heartbeat trigger frequency", "pruning the context window in real-time",
            "consolidating multi-session summaries", "evaluating the security of a tool call", "generating a synthetic curriculum for self-test"
        ],
        "style": "High-precision steps using COGA primitives: remember(), recall(), WRITE(), calc(), lookup()."
    }
}

def get_system_prompt(domain):
    spec = DOMAINS[domain]
    
    coga_context = ""
    if domain == "agentic":
        coga_context = """
AVAILABLE COGA PRIMITIVES:
- WRITE(content): Write to the scratchpad.
- READ(query): Read from the scratchpad.
- EDIT(id, content): Modify a scratchpad slot.
- DELETE(id): Clear a scratchpad slot.
- COMMIT(id): Finalize an answer.
- remember(context, info): Save to long-term memory.
- recall(query): Fetch from long-term memory.
- lookup(query): Search external knowledge.
- calc(expr): Perform math.
- verify(fact): Logical consistency check.
- execute_code(snippet): Run code in sandbox.
"""

    return f"""You are an expert task planner for a {spec['audience']}.
Your goal is to generate high-quality training data for a model that thinks before it acts.

DOMAIN: {domain.upper()}
FOCUS: {spec['focus']}
{coga_context}

REQUIRED FORMAT:
Goal: [The objective or task]
<think> [Hierarchical decomposition. Identify constraints, dependencies, and potential obstacles. Explain WHY you choose this order.] </think>
Plan:
1. [Step 1]
2. [Step 2]
...

RULES:
1. Use ONLY English.
2. The <think> section must be analytical and show professional reasoning.
3. The Plan must be a numbered list of atomic, actionable steps.
4. For the 'agentic' domain, use the COGA primitives listed above in the plan.
5. For other domains, use natural language but you can use coprocessor calls if needed.
6. Start immediately with 'Goal:'.
"""

def validate_sample(text, domain):
    # Basic structural check: Must have a Goal and a Plan
    if not ("Goal:" in text and "Plan:" in text):
        return False
        
    # We removed the strict <think> validation to allow more flexible model outputs
    # but we still want some meat in the response
    if len(text) < 100: 
        return False
        
    if domain == "agentic":
        # For agentic, we still prefer seeing at least one COGA-like primitive/function call
        primitives = ["remember", "recall", "WRITE", "EDIT", "calc", "lookup", "verify", "execute_code", "(", ")"]
        if not any(p in text for p in primitives):
            # Not a hard failure, but good to note
            pass
            
    return True

def get_sample(domain, topic, model_name=MODEL_NAME):
    prompt = get_system_prompt(domain)
    
    user_instruction = (
        f"Create a complex Planning sample about: {topic}.\n"
        f"Ensure the plan handles non-obvious constraints.\n"
        f"Random entropy seed: {random.randint(1, 10000)}."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_instruction}
        ],
        "max_tokens": 1200,
        "temperature": 0.8
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\n[API] Sending request to {model_name} (Domain: {domain})...")
    try:
        # Use (connect_timeout, read_timeout)
        response = requests.post(API_URL, json=payload, headers=headers, timeout=(15, 120))
        print(f"[API] Received response (Status: {response.status_code})")
        
        if os.path.exists("logs"):
            with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"--- {time.ctime()} --- Status: {response.status_code} Domain: {domain} Model: {model_name}\n{response.text}\n")
        
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' in res_json:
                message = res_json['choices'][0]['message']
                # Use 'or' to handle cases where the key exists but is null
                content = message.get('content') or ""
                reasoning = message.get('reasoning') or ""
                
                # If there's a separate reasoning field, wrap it in <think> tags and prepend it
                if reasoning and (not content or "<think>" not in content):
                    content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}".strip()
                
                if validate_sample(content, domain):
                    return content
                else:
                    print(f"\n[Validation Failed] Plan rejected for quality in {domain}. Retrying...")
            else:
                print(f"\n[API Error] No choices in response: {res_json}")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 60
            print(f"\n[API Error] Rate limit exceeded (429). Waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            print(f"\n[API Error] Status {response.status_code}: {response.text}")
            
        return None
    except Exception as e:
        print(f"\n[Connection Error] {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic planning data for TinyThinker (COGA).")
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()), required=True, help="Domain to generate")
    parser.add_argument("--target", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    
    args = parser.parse_args()
    
    domain = args.domain
    target = args.target
    model_name = args.model
    output_path = args.output or f"data/raw/synthetic_planning_{domain}_v1.jsonl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print(f"Generating {target} samples for domain: {domain.upper()}")
    
    start_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            start_count = sum(1 for _ in f)
    
    success_count = start_count
    pbar = tqdm(total=target, initial=start_count)
    
    while success_count < target:
        topic = random.choice(DOMAINS[domain]['topics'])
        res = get_sample(domain, topic, model_name=model_name)
        
        if res:
            with open(output_path, "a", encoding="utf-8") as f:
                data = {
                    "text": res,
                    "domain": domain,
                    "topic": topic,
                    "model": model_name,
                    "timestamp": time.ctime()
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            success_count += 1
            pbar.update(1)
            time.sleep(1)
        else:
            time.sleep(2)
    
    print(f"Generation complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()
