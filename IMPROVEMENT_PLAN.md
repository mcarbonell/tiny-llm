# Improvement Plan - TinyThinker

> Prioritized plan to move TinyThinker from a functional prototype to a more reproducible research codebase.

Last updated: 2026-04-08

## Current State

TinyThinker has a solid technical base for a small experimental LLM:

- Modern architecture in [`model/model.py`](model/model.py): RoPE, RMSNorm, SwiGLU, GQA, LoRA.
- Separate flows for pretraining, fine-tuning, chat, and evaluation.
- Basic unit and integration tests.
- Working logging and checkpointing.

There are also a few places where the repository looks more mature than the current implementation really is:

- `scripts/train.py` still hardcodes several settings even though [`scripts/config.py`](scripts/config.py) and YAML configs already exist.
- Evaluation is still good as a smoke test, but too weak to support strong claims about tool calling or agentic behavior.
- Documentation and status files sometimes describe the project as more finished than the code actually is.

## Active Run

Fine-tuning is currently running on CPU with LoRA. While that run is active:

- Safe to edit: documentation, plans, notes, and tests that do not affect the live process.
- Avoid editing: tokenizer artifacts, active datasets, checkpoints, and training scripts that are directly involved in the current run.
- The tokenizer and chat behavior should be revalidated after the run finishes.

## Changes Already Applied

These items are already in place and should not be reopened unless a regression appears:

1. `requirements.txt` was simplified to a small coherent dependency set.
2. `finetune.py` was updated for compatibility with `torch.amp.GradScaler`.
3. `chat.py` logging duplication was removed.
4. `test_tokenizer_roundtrip` was relaxed to account for ByteLevel decoding behavior.

## Immediate Priority

### P0 - After the Current Run

These are the highest-value fixes, but we should finish the active finetune before making deeper changes to the training path.

1. Consolidate configuration
   - Make [`scripts/train.py`](scripts/train.py) consume [`scripts/config.py`](scripts/config.py) and the YAML files in [`configs/`](configs).
   - Remove hardcoded training hyperparameters from multiple scripts.
   - Unify checkpoint paths, dataset paths, and common runtime settings.

2. Harden evaluation
   - Keep perplexity as a smoke test.
   - Improve tool-calling evaluation so it measures format correctness, trigger behavior, and usefulness.
   - Separate base language evaluation from agentic behavior evaluation.

3. Tighten chat and tokenizer validation
   - Recheck the regenerated tokenizer on real chat prompts.
   - Confirm that spacing and token reconstruction remain stable in interactive generation.
   - Keep tests focused on the actual user-visible bug, not only an idealized roundtrip.

4. Align documentation with the code
   - Reduce "mission accomplished" language where the repo is still experimental.
   - Fix encoding issues in [`README.md`](README.md), [`PROJECT_STATUS.md`](PROJECT_STATUS.md), and related docs.
   - Document current limitations and assumptions more clearly.

## Work That Is Safe Now

### P1 - Safe During the Active Run

1. Keep this plan current
   - Record new findings, decisions, and next steps as we validate the repo.

2. Prepare the post-run backlog
   - Leave the tokenizer, config, tests, and evaluation work clearly queued for after the finetune finishes.

3. Documentation-only improvements
   - Refine `.md` files that do not affect the live run.

## Known Good Fixes

These improvements are already in a good place:

- Duplicate attention calls in [`model/model.py`](model/model.py) were removed.
- Dead code in [`model/model.py`](model/model.py) was cleaned up.
- The output directory bug in [`scripts/train.py`](scripts/train.py) was fixed.
- KV-cache support was added to [`scripts/chat.py`](scripts/chat.py).
- Residual connections with gradient checkpointing were fixed in [`model/model.py`](model/model.py).
- Dataset validation was added in [`scripts/eval.py`](scripts/eval.py).
- Logging exists in training, fine-tuning, and chat.
- `requirements.txt` cleanup is done.
- `chat.py` logger duplication is gone.
- `test_tokenizer_roundtrip` is now less brittle.

Note: "implemented" does not always mean "fully closed". Some items still need better tests or a cleanup pass.

## Open Topic: Tokenizer

Current stance:

- The tokenizer was regenerated to address visible spacing issues in chat output.
- ByteLevel decoding can add or preserve leading spaces in ways that are valid but visually surprising.
- We should validate real chat behavior, not just a theoretical roundtrip.

Decision:

- Do not touch `model/tokenizer.json` during the active run.
- Re-evaluate tokenizer behavior after the run and decide whether the tests should assert a stricter user-facing rule.

Acceptance criteria after the run:

- Chat output does not glue words together or drop meaningful spaces.
- Tokenizer and decoder behave consistently on real prompts.
- Tests cover the actual display bug, not only a perfect roundtrip.

## Main Risks

1. Fragile reproducibility
   - If the environment cannot be rebuilt consistently, comparing runs will remain noisy.

2. Config drift
   - YAML configs and hardcoded script values can diverge and create confusion.

3. Over-optimistic evaluation
   - Perplexity plus tag presence is not enough to describe agentic quality.

4. Documentation drift
   - If the narrative moves ahead of the code, debugging and prioritization get harder.

## Secondary Backlog

### P2 - Quality and Ergonomics

1. Improve test coverage
   - Add tests focused on real chat behavior and tokenizer behavior.
   - Add tests that verify effective config handling in `train.py`.
   - Separate fast tests from slow tests more clearly.

2. Improve training ergonomics
   - Print a clearer summary of active hyperparameters on startup.
   - Make checkpoint and log naming more explicit.
   - Expose seeds and run metadata more visibly.

3. Improve dependency structure
   - Consider splitting training and dev dependencies into separate requirement files.

### P3 - Future

1. Sliding window or a better long-context strategy.
2. Multi-GPU or DDP support.
3. Docker or a more closed reproducible environment.
4. Optional experiment tracking integration.

## Post-Run Checklist

When the active finetune finishes:

1. Validate metrics and output quality on the new checkpoint.
2. Revisit tokenizer behavior with real chat examples.
3. Fix or redesign the tokenizer roundtrip test if needed.
4. Wire `train.py` to the unified configuration path.
5. Refresh docs and status claims to match the code.
