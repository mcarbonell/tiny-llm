# Project Status: Complete Evolution of TinyThinker (V10 to V12 Delta-Phase)

**Last Updated:** July 22, 2026

## Genesis and Lineage: The Path to Sovereignty
This project is the culmination of a fractal evolution of sovereign algorithms:
1. **SOMA (The Macro Ancestor):** External orchestrator that gave volition to the model over its context (pin/unpin/edit). Proof that the LLM must manage its own attention.
2. **COGA (El Salto al Sistema):** Internalization of SOMA. Transition to a Mutable Scratchpad and Dynamic Recurrence within the inference engine.
3. **Attention Neuron (The Atomic Level):** Discovery of Spectral Cores (DCT/Walsh), Complex Phase Phasors, and $O(N)$ Matrix Delta Holographic Memory (V298/V299).
4. **TinyThinker Delta-Phase TCA (The Final Synthesis):** A Matrix-Free LLM with $O(N)$ complex phase memory and causal $k=4$ depthwise convolutions for infinite context and high-density associative recall on local hardware.

---

## Milestone: V11 Fourier-ALBERT Architecture Completed & Verified
* **Champion Config (`v11_e256_d1024_k512_l8` Run 2):** Best Val Loss = **4.1287** (9.44M params, constant LR=0.015).
* **Key Discoveries:**
  - *Virtual Block Sharing (ALBERT-style):* 8 virtual layers sharing weights acts as a strong regularizer that stabilizes training at high learning rates (0.015).
  - *Logical Walsh Rank ($k=512$):* High-rank spectral cores encode complex syntactic/logical structures without expanding parameter count.

---

## Milestone: Run Serio A (`serious_v1`) Completed & Diagnostic
Completed 2000 iterations on July 21, 2026 (`logs/serious_v1.log`).

### 1. Empirical Results (`serious_v1`)
* **Config:** `UnifiedSpectral` (hippo OFF, spherical OFF, FFN denso `use_fwht_kernel=False`, weight_tying ON).
* **Hyperparams:** `dim=2048`, `emb_dim=256`, `n_layers=8`, `k_walsh=256`, `vocab=32768`, `seq_len=1024`, `batch=8`, `lr=1e-3` (cosine decay to 1e-4).
* **Final Val Loss @2000:** **7.1373** (train_loss 6.3493).

### 2. Diagnostic & Lessons Learned
* **Learning Rate Sensitivity:** `serious_v1` used LR=1e-3 (decaying to 1e-4), which is 15x lower than V11 (LR=0.015). Spectral and phase architectures are extremely sensitive to LR; a conservative LR severely stalls convergence.
* **Rank ($k=256$ vs $k=512$):** $k=256$ limited the transformation capacity relative to V11's $k=512$.
* **Missing Write-Erasure Operator:** Both V11 and `serious_v1` relied on additive spectral accumulation without an explicit Delta Rule erasure operator, causing crosstalk noise over long contexts.

---

## Milestone: Breakthrough Transfer from Attention-Neuron (V298/V299)
Empirical breakthroughs in `attention-neuron` (July 21-22, 2026) solved the linear memory crosstalk limit:

1. **Matrix Delta Rule in Complex Phase (`DeltaPhaseHolographic` $O(N)$):**
   Updates memory via residual error signal: $M_t = M_{t-1} + \frac{\beta}{d_k} (e_t \otimes K_t)$, where $e_t = V_t - \text{Re}(M \bar{K}_t)/d_k$.
   Achieved **99.95% MQAR recall accuracy** in $O(N)$ time and memory.
2. **Iso-Floats Capacity Frontier (V299):**
   Under identical state memory budget (~2,048 floats/head), Complex Phase Delta Memory maintains **95.98% accuracy at 64 KV pairs ($L=512$)**, while Real-Valued DeltaNet Vanilla collapses to **73.14%** (+22.84% superiority for complex phase).
3. **Short Causal Conv1D ($k=4$):**
   Local depthwise convolution pairs Key-Value tokens before memory injection, enabling seamless sequence learning.

---

## Milestone: V12 Delta-Phase Architecture Upgraded with Spectral Lerp Router (V328 Breakthrough)
Completed August 10, 2026 (`model/model_spectral_v12_delta_phase.py`).

### 1. Upgrade Enhancements Integrated
* **Learnable Substrate Lerp FFN Router (`v328`):** Replaced legacy dense FFN with parallel orthogonal projections (FWHT + DCT-II + DWT Haar Wavelet) modulated by phase phasors and softmax lerp routing per layer.
* **Fast Chunk Scan:** Memory scan optimized for autograd checkpointing, achieving **3,113 tokens/sec streaming throughput** in TorchScript JIT mode.
* **Transparent Substrate Report:** Automatic layer-by-layer reporting of tuned spectral bases (FWHT / DCT-II / Haar) upon training completion.
* **Verified:** `tests/test_model_v12.py` unit test passed cleanly with 12.8M parameters.

---

## Next Steps (V12 Pre-training Launch)
1. **Launch Pre-training Run V12 (`configs/train_v12_delta_phase.yaml`):** Train V12 on `data/train_v2_32k.bin` (BPE 32K vocab) with LR=0.008.
2. **Monitor Substrate Selection:** Inspect the automated `print_substrate_report()` to verify DCT-II harmonic emergence in output layers.

---
*Document updated by TinyThinker Architect. Efficiency and phase memory are the path to Sovereign AI.*
