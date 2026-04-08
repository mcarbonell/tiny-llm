import json

# CURRENT TOOL CALL DATA
with open("data/tool_dataset_clean.json", "r", encoding="utf-8") as f:
    tool_data = json.load(f)

SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]"

# GENERATE SIMPLE CHAT DATA (NO TOOLS)
chat_examples = [
    {"text": f"{SYSTEM_TEXT}\nUser: Hello!\nAssistant: <THINK> The user is greeting me. </THINK> Hello! How can I help you today?"},
    {"text": f"{SYSTEM_TEXT}\nUser: What is 2+2?\nAssistant: <THINK> Simple math. </THINK> 2 + 2 is 4."},
    {"text": f"{SYSTEM_TEXT}\nUser: Tell me a joke.\nAssistant: <THINK> User wants humor. </THINK> Why did the chicken cross the road? To get to the other side!"},
    {"text": f"{SYSTEM_TEXT}\nUser: Hi.\nAssistant: <THINK> Greeting. </THINK> Hi there!"}
] * 25 # 100 examples

# ADD SOME "THINK ONLY" EXAMPLES (No tool call needed for base knowledge)
wiki_examples = [
    {"text": f"{SYSTEM_TEXT}\nUser: What is the Sun?\nAssistant: <THINK> General knowledge. </THINK> The Sun is the star at the center of the Solar System."},
    {"text": f"{SYSTEM_TEXT}\nUser: Who was George Washington?\nAssistant: <THINK> History. </THINK> George Washington was the first President of the United States."}
] * 25 # 50 examples

mixed_dataset = tool_data + chat_examples + wiki_examples

with open("data/tool_dataset_mixed.json", "w", encoding="utf-8") as f:
    json.dump(mixed_dataset, f, indent=4, ensure_ascii=False)

print(f"Created mixed dataset with {len(mixed_dataset)} examples.")
