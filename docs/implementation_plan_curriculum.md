# 📚 Implementation Plan: TinyThinker Reasoning Curriculum (L0-L4)

## 🎯 Objective
Establish a formal training curriculum for TinyThinker to evolve its reasoning capabilities from basic pattern matching to multi-step logical deduction, using a "human development" analogy (TinyLogic).

## 🛠 Phase 1: Formalize Levels (TinyLogic Specification)
Based on `data/curriculum.md` and `PROJECT_STATUS.md`, we will focus on the most impactful levels for sub-50M models.

| Level | Name | Age Equiv. | Core Skills | Target Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **L0** | Foundation | 3-5y | Categorization, colors, basic relations (big/small). | 1 step, no inference. |
| **L1** | Concrete Early | 5-7y | Simple math (<20), 1-step cause-effect, arrival order. | 1-2 steps of reasoning. |
| **L2** | Concrete Advanced | 8-10y | Multi-step logic (3-4 entities), transitive relations (A>B>C). | 3-4 steps of reasoning. |
| **L3** | Pre-teen Structured | 10-13y | Proportional reasoning, schedules, inconsistency detection. | 5+ steps, structured rules. |
| **L4** | Formal Operations | 14-16y | Abstraction, counterfactuals ("What if?"), simple optimization. | Abstract/Hypothetical. |

## ⚙️ Phase 2: Refactor Synthetic Data Generation
Update `scripts/generate_rich_logic_openrouter.py` to:
1.  **Parameterize by Level**: Accept a `--level` argument.
2.  **Dynamic System Prompts**: Tailor the teacher persona and constraints for each level.
3.  **Categorized Topics**: Group topics by cognitive demand instead of a flat list.
4.  **Metadata Richness**: Save the level and topic in the JSONL output for better training control.

## 🧪 Phase 3: Data Production (Target: 5,000+ samples)
*   **L0-L1**: 2,000 samples (The foundation must be solid).
*   **L2-L3**: 2,000 samples (The "thinking" sweet spot).
*   **L4**: 1,000 samples (Advanced logic experiments).

## 📉 Phase 4: Training Strategy (Curriculum Learning)
1.  **Stage A (Base)**: Continue mixing TinyStories with Level 0-1 data (Language + Basic Logic).
2.  **Stage B (Advanced)**: Gradually increase the ratio of L2-L3 data.
3.  **Stage C (Expert)**: Inject L4 data as "expert examples".
4.  **Loss Weighting**: Experiment with weighting the `<think>` section vs the `Answer` section to prioritize learning the *process*.

---
## 🚀 Next Immediate Steps:
1.  **Refactor** `scripts/generate_rich_logic_openrouter.py` to support `--level`.
2.  **Add** more detailed topics for each level to the script.
3.  **Run** a small generation batch for L0 to validate the format.
