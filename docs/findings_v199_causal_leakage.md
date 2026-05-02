# Findings V199: Causal Leakage & The "UFO" Phenomenon

## Objective
To analyze the sudden drop in validation loss to **0.6086** observed in the **Spectral V5 (JPEG-LLM)** architecture during training.

## The Phenomenon (The UFO 🛸)
During the training of Spectral V5 on a 256-token context window, the loss curve behaved unnaturally:
- **Iteration 250:** val_loss 5.71 (Normal)
- **Iteration 750:** val_loss 1.69 (Suspicious)
- **Iteration 1750:** **val_loss 0.60** (Impossible for natural language)

Despite the extremely low loss, the model's generation was **completely broken**, producing infinite loops of characters (e.g., "ooooooooo", "prprprpr").

## Root Cause Analysis: Future-Leaking DCT
The failure was traced to a violation of the **Causality Principle** in the temporal compression module (`_compress_temporal`).

### 1. Global Transform over Causal Window
The 1D-DCT transform used for KV-Cache compression was applied to the **entire block of tokens** during training. 
- In a sequence of [T1, T2, T3, T4], the DCT coefficients for T1 were calculated using information from T2, T3, and T4.
- Because the DCT is a global frequency transform, any single coefficient contains "echoes" of the entire sequence.

### 2. Information Shortcut
The model stopped learning to predict the next word. Instead, it learned to **invert the DCT coefficients** to read the future tokens that were accidentally leaked into the "compressed" cache.
- The loss drop was a measure of how well the model could "decrypt" the future tokens from the spectral noise.

## Impact on Generation
When the model moved to inference (chat mode), it failed because:
1. Tokens are generated one by one.
2. There is **no future** to compress during inference.
3. The model, having only learned to decrypt leaked future info, was unable to handle a purely causal past, leading to "Semantic Blindness" and repetitive loops.

## Lessons Learned
1. **Spectral Causality is Hard:** Global transforms (DCT/FFT) cannot be applied directly to training sequences without strict masking or step-by-step application.
2. **Loss is not Intelligence:** A low loss can be a sign of a bug (data/causal leakage) rather than high model quality.
3. **The "JPEG-LLM" Hypothesis (V65):** The theory of a "blurred past" remains valid, but it must be implemented as a **moving window** or a **cumulative update** to preserve the arrow of time.

## Next Steps: Spectral V6 (Causal-JPEG)
Implement a strictly causal version of the temporal compression where:
- **Training:** Uses standard sharp attention (no leakage).
- **Compression:** Only happens on tokens strictly behind the current position.
- **Robustness:** Inject noise into the past during training to simulate the "blur" without looking at the future.
