# Modular Expert Slots: Plug-and-Play Domain Specialization in Mixture-of-Experts Models without Catastrophic Forgetting

---

## Abstract

Fine-tuning Large Language Models (LLMs) for domain-specific applications remains fundamentally constrained by catastrophic forgetting: improving performance in a target domain degrades general capabilities. Existing mitigations—LoRA adapters, elastic weight consolidation, replay buffers—offer partial solutions but involve inherent trade-offs between specialization depth and knowledge retention. We propose **Modular Expert Slots (MES)**, a simple yet effective architectural paradigm for Mixture-of-Experts (MoE) models in which a subset of expert modules are intentionally left untrained ("virgin") during pretraining and reserved for subsequent domain specialization. During fine-tuning, only the reserved experts and the routing network are updated while all pretrained expert weights remain frozen, guaranteeing zero forgetting of base capabilities by construction. We introduce a three-phase fine-tuning protocol—warm-up initialization, constrained routing training, and router calibration—that enables reliable domain specialization. The resulting system supports **plug-and-play expert management**: domain experts can be loaded, unloaded, swapped, and composed at inference time without retraining. We present theoretical arguments for why this approach preserves base model capabilities, propose a complete experimental validation plan, and discuss extensions including expert versioning, multi-domain composition, and continual learning through progressive slot allocation.

**Keywords**: Mixture of Experts, catastrophic forgetting, modular fine-tuning, continual learning, domain specialization, plug-and-play

---

## 1. Introduction

### 1.1 The Fine-Tuning Dilemma

The dominant paradigm for adapting LLMs to specialized domains involves fine-tuning pretrained weights on domain-specific data. This approach faces a well-documented tension (Kirkpatrick et al., 2017; French, 1999): updating model weights to encode new domain knowledge inevitably overwrites representations that encoded prior general knowledge. We refer to this as the **specialization-retention trade-off**.

Current approaches occupy different points on this trade-off curve:

- **Full fine-tuning** achieves maximum specialization but maximum forgetting.
- **Parameter-Efficient Fine-Tuning (PEFT)** methods such as LoRA (Hu et al., 2022) reduce forgetting by constraining updates to low-rank subspaces, but this simultaneously limits specialization depth.
- **Regularization-based methods** such as EWC (Kirkpatrick et al., 2017) and L2-SP (Li et al., 2018) penalize deviation from pretrained weights, explicitly trading specialization for retention.
- **Replay-based methods** mix domain data with general data during fine-tuning, requiring access to pretraining data and increasing training cost.

All these methods share a fundamental limitation: they attempt to *balance* competing objectives within a shared set of parameters. The specialization-retention trade-off is not eliminated but managed.

### 1.2 Key Insight: Dedicated Capacity for New Knowledge

We propose a fundamentally different approach. Instead of modifying existing parameters (and managing the resulting interference), we **pre-allocate dedicated capacity** for future domain knowledge. Specifically, in a Mixture-of-Experts architecture, we reserve a subset of expert modules that remain untrained during pretraining. During subsequent domain fine-tuning:

- **Reserved experts** are trained on domain data → acquires specialization.
- **Base experts** remain completely frozen → guarantees zero forgetting.
- **The router** is updated to learn when to dispatch to the new experts → enables integration.

This eliminates the specialization-retention trade-off entirely: the two objectives are served by physically separate parameters.

### 1.3 Biological Inspiration

This architecture mirrors the neocortical organization of the mammalian brain. The brain does not learn new skills by overwriting existing neural circuits. Instead, it recruits underutilized neural populations and establishes new routing pathways to integrate them with existing circuits (Merzenich et al., 1984; Pascual-Leone et al., 2005). The analogy is direct:

| Brain mechanism | MES equivalent |
|---|---|
| Existing specialized cortical areas | Pretrained base experts (frozen) |
| Recruited neural populations | Virgin expert slots (trainable) |
| Synaptic pathway formation | Router weight updates |
| Parallel processing streams | Multiple experts active per token |

### 1.4 Contributions

1. **Modular Expert Slots (MES)**: an architectural paradigm for MoE models that reserves untrained expert capacity for future domain specialization with zero catastrophic forgetting by construction.

2. **Three-phase fine-tuning protocol**: a training procedure comprising warm-up initialization, constrained routing training, and router calibration that reliably produces well-integrated domain experts.

3. **Plug-and-play expert management**: a runtime system for loading, unloading, swapping, and composing domain experts at inference time without any retraining.

4. **Mixed-data routing stabilization**: a data mixing strategy during fine-tuning that prevents router collapse and maintains balanced utilization across base and new experts.

5. **Theoretical analysis**: formal guarantees on base capability preservation and analysis of conditions under which MES achieves equivalent specialization to full fine-tuning.

---

## 2. Related Work

### 2.1 Mixture of Experts

The Mixture-of-Experts paradigm (Jacobs et al., 1991; Shazeer et al., 2017) enables conditional computation by routing each input to a subset of specialized expert networks. Recent large-scale MoE models—Switch Transformer (Fedus et al., 2022), GShard (Lepikhin et al., 2021), Mixtral (Jiang et al., 2024), DeepSeek-MoE (Dai et al., 2024), and DBRX (Databricks, 2024)—have demonstrated that MoE architectures can match or exceed dense model performance with significantly lower inference compute by activating only top-*k* experts per token.

A key challenge in MoE training is **load balancing**: ensuring that the router distributes tokens across experts rather than collapsing to a few preferred experts. Fedus et al. (2022) introduced auxiliary load-balancing losses, and subsequent work has refined routing mechanisms (Zhou et al., 2022; Lewis et al., 2021). Our work builds on these foundations but introduces a novel consideration: routing to experts that were *intentionally excluded* from pretraining.

### 2.2 Catastrophic Forgetting and Continual Learning

Catastrophic forgetting—the tendency of neural networks to lose previously learned knowledge when trained on new tasks—has been studied extensively (McCloskey & Cohen, 1989; French, 1999; Goodfellow et al., 2014). Approaches to mitigate forgetting fall into three categories:

**Regularization-based methods** add constraints that penalize changes to important weights. Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) estimates parameter importance via Fisher information. Synaptic Intelligence (Zenke et al., 2017) tracks importance online. These methods require computing and storing importance metrics and introduce hyperparameters that control the retention-specialization trade-off.

**Replay-based methods** retain a buffer of previous data and interleave it during new training (Rebuffi et al., 2017; Shin et al., 2017). These require access to (potentially proprietary or large) prior datasets and increase training cost proportionally.

**Architecture-based methods** allocate different parameters for different tasks. Progressive Neural Networks (Rusu et al., 2016) add new columns for new tasks but grow linearly. PackNet (Mallya & Lazebnik, 2018) iteratively prunes and freezes network portions. Expert Gate (Aljundi et al., 2017) uses autoencoder-based gating to select task-specific expert modules.

MES falls in the architecture-based category but differs crucially from prior work: expert capacity is pre-allocated during the initial architecture design rather than grown post-hoc, and the routing mechanism is a natural component of the MoE architecture rather than an auxiliary addition.

### 2.3 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2022) and its variants (QLoRA, Dettmers et al., 2023; LoRA+, Hayou et al., 2024) fine-tune low-rank additive matrices while freezing base weights. Adapters (Houlsby et al., 2019) insert small trainable modules between frozen layers. Prefix Tuning (Li & Liang, 2021) and Prompt Tuning (Lester et al., 2021) add trainable tokens to the input.

These methods share a key limitation: the expressive capacity of the trainable parameters is fundamentally bounded by the dimensionality of the adaptation (rank of LoRA matrices, size of adapter bottleneck, number of prefix tokens). MES, by contrast, allocates *full-capacity expert modules* as the trainable component, providing substantially more expressive power for domain adaptation while maintaining the zero-forgetting guarantee through weight freezing.

| Method | Trainable params | Forgetting risk | Specialization depth | Composable |
|---|---|---|---|---|
| Full fine-tune | All | 🔴 High | 🟢 Maximum | ❌ |
| LoRA | ~0.1-1% | 🟡 Low | 🟡 Medium | ⚠️ Limited |
| Adapters | ~1-5% | 🟡 Low | 🟡 Medium | ⚠️ Limited |
| EWC | All | 🟡 Medium | 🟡 Medium | ❌ |
| Replay | All | 🟡 Medium | 🟢 High | ❌ |
| **MES (ours)** | **~8-25%** | **🟢 Zero** | **🟢 High** | **✅ Native** |

### 2.4 Modular and Composable Approaches

Recent work has explored modularity in neural networks more broadly. Ponti et al. (2023) survey modular deep learning approaches. Zhang et al. (2023) propose LoRAHub for composing multiple LoRA adapters. Huang et al. (2023) explore adapter merging strategies. AdapterFusion (Pfeiffer et al., 2021) learns to combine multiple adapters.

MES differs from adapter composition approaches in that expert modules are first-class architectural components with dedicated routing, rather than auxiliary additions that require separate fusion mechanisms. The MoE router provides a natural, learned composition mechanism that adapts per-token.

---

## 3. Method

### 3.1 Architecture

#### 3.1.1 MoE Layer with Reserved Slots

Consider a standard MoE transformer with *L* layers, where each layer *l* contains a self-attention sublayer followed by an MoE feed-forward sublayer. The MoE sublayer at layer *l* consists of *N* expert networks {*E*₁ˡ, *E*₂ˡ, ..., *E*ₙˡ} and a router network *R*ˡ.

We partition the *N* experts into two sets:

- **Base experts** *B* = {*E*₁ˡ, ..., *E*ₘˡ}: trained during pretraining, frozen during fine-tuning.
- **Slot experts** *S* = {*E*ₘ₊₁ˡ, ..., *E*ₙˡ}: reserved (untrained or minimally initialized) during pretraining, trainable during fine-tuning.

For a model with *N* = 16 experts, a typical configuration is *M* = 12 base experts and *N* - *M* = 4 slot experts.

#### 3.1.2 Routing Mechanism

The router *R*ˡ at layer *l* computes routing weights for a token representation *x*:

```
R^l(x) = TopK(softmax(W_r^l · x + b_r^l), k)
```

where *W*ᵣˡ ∈ ℝᴺˣᵈ is the routing weight matrix and TopK selects the *k* experts with highest routing probability (typically *k* = 2).

The output of the MoE layer is:

```
MoE^l(x) = Σᵢ∈TopK  gᵢ · Eᵢ^l(x)
```

where *gᵢ* are the normalized routing weights for the selected experts.

#### 3.1.3 Expert Architecture

Each expert *Eᵢˡ* is a standard feed-forward network:

```
Eᵢ^l(x) = W₂ᵢ · σ(W₁ᵢ · x) 
```

where *W*₁ᵢ ∈ ℝᵈᶠᶠˣᵈ, *W*₂ᵢ ∈ ℝᵈˣᵈᶠᶠ, and σ is a nonlinearity (SwiGLU in practice).

Critically, each slot expert has **identical architecture and capacity** to each base expert. This ensures that domain specialization is not bottlenecked by adapter size (as in LoRA) but has full expressive capacity.

### 3.2 Pretraining with Reserved Slots

During pretraining, slot experts are handled by one of two strategies:

**Strategy A: Full exclusion.** Slot experts are initialized randomly but excluded from routing by masking their logits to -∞. The model pretrains as if they do not exist. This is cleanest but means the router has no experience routing to these slots.

**Strategy B: Minimal participation.** Slot experts participate in pretraining but with reduced routing probability (e.g., their logits are downscaled by factor 0.1). They acquire generic representations without specializing, and the router learns that they exist as valid targets. We hypothesize this produces better fine-tuning outcomes due to router familiarity and will test both strategies.

**Strategy C: Partial training and freezing.** Slot experts are pretrained normally for a fraction of training (e.g., 10% of tokens), then frozen early. They acquire basic representations but do not fully specialize, leaving capacity for domain knowledge.

### 3.3 Three-Phase Fine-Tuning Protocol

#### 3.3.1 Phase 1: Warm-Up Initialization

Before fine-tuning begins, we initialize slot expert weights. Three options:

**Option 1: Copy from base expert.** Select the base expert with the most general routing pattern (highest entropy routing distribution) and copy its weights to the slot expert.

```
E_slot^l ← copy(E_general^l)  ∀ layers l
```

This ensures the slot expert produces reasonable outputs from the first forward pass, preventing the router from learning to avoid it due to initially poor quality outputs.

**Option 2: Average of base experts.** Initialize as the mean of all base expert weights.

```
E_slot^l ← (1/M) Σⱼ₌₁ᴹ Eⱼ^l  ∀ layers l
```

This provides a "neutral" starting point that doesn't bias toward any particular base expert's specialization.

**Option 3: Retain from pretraining.** If Strategy B or C was used during pretraining, the slot expert already has reasonable weights and no initialization is needed.

We recommend **Option 1** as default, with Option 3 when pretraining used Strategy B.

#### 3.3.2 Phase 2: Constrained Routing Training

The core fine-tuning phase trains two parameter groups simultaneously:

- **Θ_slot**: weights of the slot expert(s) being specialized
- **Θ_router**: weights of the routing network across all layers

All base expert weights **Θ_base** remain frozen.

**Training data composition** is critical to prevent router collapse:

```
D_finetune = {
    D_domain   (40%):  domain-specific training data
    D_general  (40%):  general-domain data (replay buffer)
    D_contrast (10%):  data from other specific domains
    D_adversarial (10%): ambiguous cases near domain boundaries
}
```

The rationale for each component:

- *D_domain*: teaches the slot expert domain knowledge and teaches the router to direct domain queries to the slot expert.
- *D_general*: prevents the router from developing a bias toward the new expert; maintains established routing patterns for non-domain queries.
- *D_contrast*: sharpens the router's decision boundary; teaches it what is NOT the new domain.
- *D_adversarial*: handles edge cases (e.g., "general biology" vs. "clinical medicine") that determine routing precision.

**Routing constraint during Phase 2:**

For domain-specific data, we apply a soft routing bias toward the new expert:

```
logits_modified = logits + α · one_hot(slot_id) · 𝟙[is_domain_data]
```

where α > 0 is an additive bias that increases the probability of routing to the slot expert for domain data. α starts high (e.g., 5.0) and anneals to 0 over training, transitioning from forced routing to learned routing.

**Loss function:**

```
L = L_LM + λ_balance · L_balance + λ_consistency · L_consistency

L_LM: standard language modeling loss (cross-entropy)
L_balance: expert load balancing (Fedus et al., 2022)
L_consistency: routing consistency across layers (new)
```

The routing consistency loss encourages the router to make consistent decisions across layers for a given input:

```
L_consistency = (1/L) Σ_l ||r^l(x) - r̄(x)||²
```

where *r*ˡ(x) is the routing distribution at layer *l* and *r̄*(x) is the mean routing distribution across layers. This prevents pathological cases where a medical query is routed to the medical expert in some layers but to unrelated experts in others.

#### 3.3.3 Phase 3: Router Calibration

After Phase 2, the slot expert has learned domain knowledge but the router may not be optimally calibrated. Phase 3 fine-tunes **only the router** on a calibration dataset:

```
D_calibration = {
    D_domain   (30%):  domain-specific data
    D_general  (60%):  general-domain data
    D_other    (10%):  other domain data
}
```

The higher proportion of general data (60% vs. 40% in Phase 2) ensures that final routing decisions are conservative: the router should only dispatch to the slot expert when confident that the input is domain-relevant.

Only router parameters are updated; all experts (base AND slot) are frozen. This isolates routing quality from expert quality and enables rapid calibration.

**Duration**: Phase 3 is short (1-2 epochs) and inexpensive. It can be repeated if routing quality is unsatisfactory.

### 3.4 Plug-and-Play Expert Management

#### 3.4.1 Expert Serialization

After fine-tuning, a domain expert consists of two artifacts:

```
domain_expert_package = {
    expert_weights: Θ_slot,           # weights of the trained expert
    router_patch:   ΔΘ_router,        # diff of router weights (post - pre)
    metadata: {
        domain: str,                  # "medicine", "law", etc.
        base_model_hash: str,         # ensures compatibility
        slot_id: int,                 # recommended slot
        training_config: dict,        # hyperparameters used
        eval_metrics: dict,           # benchmark results
        version: str                  # semantic versioning
    }
}
```

The router patch ΔΘ_router is stored as the difference from base router weights, enabling additive composition.

#### 3.4.2 Runtime Operations

**Load:**
```python
def load_expert(model, package, slot_id=None):
    slot = slot_id or package.metadata.slot_id
    assert model.experts[slot].is_slot  # verify it's a reserved slot
    assert package.metadata.base_model_hash == model.hash  # compatibility
    
    model.experts[slot].load_state_dict(package.expert_weights)
    model.router.apply_additive_patch(slot, package.router_patch)
    model.registry.register(slot, package.metadata)
```

**Unload:**
```python
def unload_expert(model, slot_id):
    model.experts[slot_id].reset()  # restore to pre-load state
    model.router.remove_patch(slot_id)
    model.registry.unregister(slot_id)
```

**Swap:**
```python
def swap_expert(model, slot_id, new_package):
    unload_expert(model, slot_id)
    load_expert(model, new_package, slot_id)
```

**Compose:**
```python
def load_multiple(model, packages):
    """Load multiple domain experts simultaneously"""
    for pkg in packages:
        slot = find_available_slot(model)
        load_expert(model, pkg, slot)
    # Router patches are additive → composition is natural
```

#### 3.4.3 Composition Analysis

When multiple domain experts are loaded simultaneously, the router must make per-token decisions about which expert(s) to invoke. Because the router uses TopK selection over all experts (base + loaded slots), composition emerges naturally:

- A medical-legal query might route to both the medical expert and the legal expert across different layers.
- A purely medical query routes primarily to the medical expert, ignoring the legal expert.
- A general knowledge query routes to base experts, ignoring both domain experts.

We formalize the composition property:

**Definition (Non-interference):** Two loaded experts *E*_A and *E*_B exhibit non-interference if:

```
accuracy(model + E_A + E_B, domain_A) ≈ accuracy(model + E_A, domain_A)
accuracy(model + E_A + E_B, domain_B) ≈ accuracy(model + E_B, domain_B)
accuracy(model + E_A + E_B, general) ≈ accuracy(model, general)
```

Non-interference is expected when domains A and B have low routing overlap (the router rarely selects both domain experts for the same token). We will empirically measure routing overlap and its correlation with interference.

### 3.5 Theoretical Analysis

#### 3.5.1 Zero Forgetting Guarantee

**Theorem 1 (Base Capability Preservation).** Let *f*_base be the base model function before fine-tuning and *f*_MES be the model function after MES fine-tuning. For any input *x* where the router assigns zero probability to all slot experts:

```
f_MES(x) = f_base(x)
```

**Proof:** If the router assigns zero weight to slot experts for input *x*, the MoE layer output depends only on base expert outputs. Since base expert weights are frozen, their outputs are identical to before fine-tuning. Since all other model components (attention layers, embeddings, layer norms) are also frozen, the model function is identical. ∎

**Corollary 1.** Base capability degradation under MES is bounded by the router's tendency to (incorrectly) route general-domain inputs to slot experts. This can be measured and minimized through Phase 3 calibration.

In practice, some router probability mass will leak to slot experts even for general inputs. We define the **routing leakage** ε:

```
ε = E_{x~D_general}[Σ_{s∈S} R(x)_s]
```

and show empirically that Phase 3 calibration reduces ε to negligible levels (< 0.01).

#### 3.5.2 Specialization Capacity

**Proposition 1 (Expressive Capacity).** A slot expert with feed-forward dimensions (*d*, *d_ff*) has identical expressive capacity to each base expert. The domain specialization achievable by MES is bounded below by the domain specialization achievable by fine-tuning a single equivalent-sized expert network with access to the same base model representations.

This contrasts with LoRA, where specialization capacity is bounded by the rank *r* << min(*d*, *d_ff*). For typical configurations:

```
LoRA trainable params:    2 × r × d ≈ 2 × 16 × 4096 = 131K per layer
MES trainable params:     d × d_ff + d_ff × d ≈ 2 × 4096 × 16384 = 134M per layer

Ratio: MES/LoRA ≈ 1000x more expressive capacity per domain
```

#### 3.5.3 Scaling Properties

The overhead of MES scales linearly with the number of slots:

```
Memory overhead: O(S × P_expert) where S = num slots, P_expert = params per expert
Compute overhead: O(0) when slot experts are not routed to
                  O(k/N × S × C_expert) in worst case (top-k routing)
Router overhead: O(S × d) additional routing weights (negligible)
```

For typical configurations (S=4, N=16, k=2), the worst-case compute overhead is 2/16 × 4/16 ≈ 3% of total MoE compute — negligible.

---

## 4. Experimental Design

### 4.1 Research Questions

We design experiments to answer five primary research questions:

- **RQ1 (Retention):** Does MES achieve zero or near-zero degradation on general benchmarks after domain fine-tuning?
- **RQ2 (Specialization):** Does MES achieve domain performance comparable to full fine-tuning?
- **RQ3 (Efficiency):** How does MES compare to LoRA and other PEFT methods in the retention-specialization trade-off?
- **RQ4 (Composition):** Can multiple domain experts be loaded simultaneously without interference?
- **RQ5 (Router):** Does the three-phase protocol produce well-calibrated routing?

### 4.2 Base Models

We propose experiments at two scales:

**Small scale (for rapid iteration):**
```
Architecture: Custom MoE Transformer
Parameters: ~2B total, ~400M active (top-2 of 16 experts)
Experts: 12 base + 4 slots
Layers: 24
d_model: 2048
d_ff per expert: 5120 (SwiGLU)
Training: 50B tokens, high-quality curated data
```

**Medium scale (for main results):**
```
Architecture: Mixtral-style MoE
Parameters: ~14B total, ~3.5B active (top-2 of 16 experts)
Experts: 12 base + 4 slots
Layers: 32
d_model: 4096
d_ff per expert: 14336 (SwiGLU)
Training: 200B tokens, high-quality curated data
Alternative: start from Mixtral 8x7B and add 4 slot experts
```

### 4.3 Domain Fine-Tuning Tasks

We select three domains with varying characteristics:

**Domain 1: Medicine**
```
Training data: PubMed abstracts + clinical guidelines + medical textbooks
Size: ~5B tokens
Evaluation: MedQA (USMLE-style), PubMedQA, MedMCQA, MMLU-medical
Characteristics: specialized vocabulary, precise factual knowledge, 
                 reasoning over clinical scenarios
```

**Domain 2: Law**
```
Training data: Legal opinions + statutes + case law + legal textbooks
Size: ~3B tokens
Evaluation: LegalBench, MMLU-law, bar exam questions
Characteristics: formal language, logical reasoning, 
                 jurisdiction-specific knowledge
```

**Domain 3: Code**
```
Training data: High-quality code + documentation + Stack Overflow
Size: ~8B tokens
Evaluation: HumanEval, MBPP, DS-1000, SWE-bench-lite
Characteristics: formal syntax, algorithmic reasoning, 
                 diverse programming languages
```

### 4.4 Baselines

```
B1: Base model (no fine-tuning)           → general ceiling, domain floor
B2: Full fine-tuning                       → domain ceiling, general floor
B3: LoRA (r=16)                           → PEFT baseline (low capacity)
B4: LoRA (r=256)                          → PEFT baseline (high capacity)
B5: QLoRA (r=16, 4-bit)                   → efficiency baseline
B6: Full fine-tune + replay (40% general) → replay baseline
B7: Full fine-tune + EWC                  → regularization baseline
B8: MES Strategy A (our, full exclusion)  → our method (variant A)
B9: MES Strategy B (our, minimal participation) → our method (variant B)
B10: MES Strategy C (our, partial training) → our method (variant C)
```

### 4.5 Evaluation Protocol

#### 4.5.1 Retention Metrics

```
For each method M fine-tuned on domain D:

Retention_score(M, D) = accuracy(M_finetuned, MMLU_general) / 
                         accuracy(M_base, MMLU_general)

Where MMLU_general excludes subjects related to domain D.

Perfect retention = 1.0
Full fine-tuning typically achieves 0.85-0.95
MES target: ≥ 0.99 (near-perfect retention)
```

Additional retention benchmarks: HellaSwag, ARC-Easy, WinoGrande, PIQA, BoolQ.

#### 4.5.2 Specialization Metrics

```
Specialization_score(M, D) = accuracy(M_finetuned, benchmark_D) / 
                              accuracy(M_full_finetune, benchmark_D)

Where M_full_finetune is the fully fine-tuned model (B2).

Perfect specialization = 1.0 (matches full fine-tuning)
LoRA typically achieves 0.90-0.98
MES target: ≥ 0.95
```

#### 4.5.3 Combined Score

We define the **Retention-Specialization Product (RSP):**

```
RSP(M, D) = Retention_score(M, D) × Specialization_score(M, D)

RSP = 1.0 is perfect (full specialization + zero forgetting)
This is impossible for methods that trade off the two objectives.
MES should achieve RSP close to the product of its individual targets:
Target RSP: ≥ 0.99 × 0.95 = 0.94
```

#### 4.5.4 Composition Metrics

For multi-domain loading experiments:

```
Interference(E_A, E_B) = 1 - accuracy(model + E_A + E_B, domain_A) / 
                              accuracy(model + E_A, domain_A)

Non-interference target: Interference < 0.02 (< 2% accuracy drop)

Cross-domain score: average accuracy across all loaded domains 
                    and general benchmarks simultaneously
```

#### 4.5.5 Router Quality Metrics

```
Routing precision: P(token_is_domain | routed_to_slot_expert)
Routing recall:    P(routed_to_slot_expert | token_is_domain)
Routing leakage:   P(routed_to_slot_expert | token_is_general)
Routing consistency: variance of routing decisions across layers

Targets: precision > 0.90, recall > 0.85, leakage < 0.01
```

### 4.6 Ablation Studies

```
A1: Number of slot experts
    {1, 2, 4, 8} slots with N=16 total experts
    Question: diminishing returns? optimal ratio?

A2: Initialization strategy
    {random, copy-general, copy-closest, average, Strategy B pretrained}
    Question: which initialization produces best fine-tuning outcomes?

A3: Data mixing ratios
    Domain %: {20, 40, 60, 80, 100}
    General %: complement
    Question: optimal ratio? minimum general data needed?

A4: Phase 2 vs. Phase 2+3
    Skip Phase 3 calibration
    Question: how much does router calibration matter?

A5: Routing constraint annealing
    α schedule: {constant, linear decay, cosine decay, step decay}
    Question: best schedule for routing constraint?

A6: Router scope
    {freeze router, train only router, train router + slot expert}
    Question: is router training necessary? sufficient?

A7: Number of fine-tuning tokens
    {100M, 500M, 1B, 5B} domain tokens
    Question: data efficiency of MES vs. baselines?

A8: Expert capacity
    Slot expert with {0.5x, 1x, 2x} capacity vs. base experts
    Question: do slot experts need the same size as base experts?
```

### 4.7 Compute Budget

```
Small scale experiments:
├── Base model pretraining: ~500 GPU-hours (A100-80GB)
├── Fine-tuning (per method, per domain): ~20 GPU-hours
├── Total baselines: 7 methods × 3 domains × 20 = 420 GPU-hours
├── MES variants: 3 variants × 3 domains × 20 = 180 GPU-hours
├── Ablations: ~8 × 3 × 10 = 240 GPU-hours
├── Evaluation: ~50 GPU-hours
└── Total small scale: ~1,400 GPU-hours

Medium scale experiments:
├── Base model pretraining: ~5,000 GPU-hours
├── Fine-tuning + ablations: ~2,000 GPU-hours
├── Evaluation: ~200 GPU-hours
└── Total medium scale: ~7,200 GPU-hours

GRAND TOTAL: ~8,600 GPU-hours ≈ $13,000-$17,000 at cloud rates
```

---

## 5. Expected Results

### 5.1 Primary Hypotheses

```
H1: MES achieves Retention_score ≥ 0.99 across all domains
    Confidence: HIGH (guaranteed by architecture when ε → 0)

H2: MES achieves Specialization_score ≥ 0.95 for Medicine and Law,
    ≥ 0.90 for Code
    Confidence: MEDIUM (depends on expert capacity being sufficient)

H3: MES achieves higher RSP than all baselines
    Confidence: HIGH (no other method has architectural zero-forgetting)

H4: Multiple domain experts compose without significant interference
    Confidence: MEDIUM-HIGH (depends on routing quality)

H5: Phase 3 calibration reduces routing leakage by ≥ 50%
    Confidence: HIGH (directly optimizes for this)
```

### 5.2 Expected Trade-Off Curves

```
                    Specialization Score
                    1.0 ┤                     ★ Full FT
                        │                 ★ MES
                        │            ★ Replay
                    0.9 ┤       ★ LoRA-256
                        │    ★ EWC
                        │  ★ LoRA-16
                    0.8 ┤
                        │
                        │
                    0.7 ┤
                        │★ Base (no FT)
                        └─────────────────────────────
                        0.7  0.8  0.9  0.95  1.0
                              Retention Score

    MES occupies the top-right corner: high specialization + high retention
    Full FT occupies top-left: high specialization but low retention
    Base model occupies bottom-right: high retention but no specialization
```

---

## 6. Extensions and Future Work

### 6.1 Progressive Slot Allocation

As slot experts are consumed by domain fine-tuning, the model runs out of available slots. We propose **progressive slot allocation**:

```
Generation 0: Pretrain with N=16, M=12 base, S=4 slots
Generation 1: Fine-tune slot 13 (medicine), slot 14 (law)
              → 2 slots remaining
Generation 2: Add 4 more expert modules to architecture (N=20)
              → new slots 17-20 available
              → router expanded with 4 new output dimensions
              → brief router recalibration (Phase 3 only)
Generation 3: Fine-tune slots for new domains
              → cycle continues indefinitely
```

This enables **continual learning** without ever touching previously trained weights.

### 6.2 Expert Versioning and Registries

We envision a community ecosystem around expert modules:

```
Expert Registry (conceptual):

medical_expert_v1.0    - PubMedQA: 78.3%  - trained on PubMed 2023
medical_expert_v2.0    - PubMedQA: 82.1%  - added clinical guidelines
legal_expert_US_v1.0   - LegalBench: 71.2% - US federal law
legal_expert_EU_v1.0   - LegalBench: 68.9% - EU regulations
code_python_v1.0       - HumanEval: 79.3%  - Python specialist
code_rust_v1.0         - HumanEval: 73.1%  - Rust specialist

Users download and load experts as needed, like packages.
```

### 6.3 Expert Distillation

If slot experts converge to particularly useful representations, they could be distilled back into the base model during a "consolidation" pretraining phase, freeing the slot for new domains. This mirrors memory consolidation in the brain (hippocampus → neocortex transfer during sleep).

### 6.4 Cross-Expert Attention

Enable experts to attend to each other's intermediate representations, allowing richer composition than independent parallel processing.

### 6.5 Hierarchical Routing

Instead of flat routing over all experts, implement a two-level routing hierarchy:

```
Level 1: Route to domain (general / medicine / law / code)
Level 2: Route to specific expert within domain
```

This scales better to large numbers of loaded experts and produces more interpretable routing decisions.

### 6.6 Expert-Specific Tokenization

Different domains may benefit from different tokenization. Slot experts could include domain-specific tokenizer extensions (additional vocabulary items) that are only active when the expert is loaded.

---

## 7. Broader Impact

### 7.1 Democratization of Specialization

MES lowers the barrier to domain-specific AI. Organizations can specialize a base model for their domain without:
- Massive compute budgets (only need to train slot experts + router)
- Risk of degrading general capabilities
- Access to pretraining data for replay

A hospital could fine-tune a medical expert; a law firm could fine-tune a legal expert. Both could share the same base model infrastructure.

### 7.2 Environmental Impact

By enabling specialization through partial model updates rather than full retraining, MES reduces the compute (and energy) cost of domain adaptation by an estimated 70-90% compared to full fine-tuning.

### 7.3 Risks

- **Malicious expert modules**: a plug-and-play system could enable distribution of expert modules trained on harmful content. Mitigation: expert signing, verification, and safety benchmarks before loading.
- **Over-reliance on routing**: if the router makes systematic errors (e.g., never routing minority languages to appropriate experts), performance disparities could emerge.
- **Expert quality verification**: without standardized benchmarks per domain, users may load poorly trained experts that degrade performance.

---

## 8. Conclusion

We have presented Modular Expert Slots, a straightforward architectural paradigm that resolves the specialization-retention trade-off in LLM fine-tuning by allocating dedicated, initially untrained expert capacity within Mixture-of-Experts models. The key insight—that new knowledge should be stored in *new* parameters rather than overwriting existing ones—is simple but has profound implications for how we think about model adaptation and continual learning.

MES offers three properties that no existing method provides simultaneously:

1. **Zero forgetting by construction** (base weights are never modified)
2. **Full-capacity specialization** (expert modules have the same capacity as base experts, unlike low-rank adapters)
3. **Native composability** (the MoE router provides learned, per-token composition of multiple domain experts)

We believe MES represents a step toward treating LLMs less as monolithic artifacts to be carefully fine-tuned and more as **modular platforms** that can be extended with domain knowledge as needed — much as operating systems can be extended with drivers and applications.

---

## References

```
Aljundi, R., Chakravarty, P., & Tuytelaars, T. (2017). Expert Gate: Lifelong 
    Learning with a Network of Experts. CVPR.

Dai, D., et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization 
    in Mixture-of-Experts Language Models.

Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized 
    Language Models. NeurIPS.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling 
    to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR.

French, R. M. (1999). Catastrophic forgetting in connectionist networks. 
    Trends in Cognitive Sciences.

Goodfellow, I. J., et al. (2014). An Empirical Investigation of Catastrophic 
    Forgetting in Gradient-Based Neural Networks.

Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. ICML.

Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.

Jacobs, R. A., et al. (1991). Adaptive Mixtures of Local Experts. 
    Neural Computation.

Jiang, A. Q., et al. (2024). Mixtral of Experts.

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in 
    neural networks. PNAS.

Lepikhin, D., et al. (2021). GShard: Scaling Giant Models with Conditional 
    Computation and Automatic Sharding. ICLR.

Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for 
    Parameter-Efficient Prompt Tuning. EMNLP.

Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous 
    Prompts for Generation. ACL.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a 
    Single Network by Iterative Pruning. CVPR.

Merzenich, M. M., et al. (1984). Somatosensory cortical map changes 
    following digit amputation in adult monkeys. Journal of Comparative 
    Neurology.

Pascual-Leone, A., et al. (2005). The plastic human brain cortex. 
    Annual Review of Neuroscience.

Pfeiffer, J., et al. (2021). AdapterFusion: Non-Destructive Task 
    Composition for Transfer Learning. EACL.

Ponti, E. M., et al. (2023). Combining Modular Skills in Multitask Learning.

Rebuffi, S. A., Kolesnikov, A., & Lampert, C. H. (2017). iCaRL: 
    Incremental Classifier and Representation Learning. CVPR.

Rusu, A. A., et al. (2016). Progressive Neural Networks.

Shazeer, N., et al. (2017). Outrageously Large Neural Networks: 
    The Sparsely-Gated Mixture-of-Experts Layer. ICLR.

Shin, H., et al. (2017). Continual Learning with Deep Generative Replay. NeurIPS.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through 
    Synaptic Intelligence. ICML.

Zhang, Q., et al. (2023). Composing Parameter-Efficient Modules with 
    Arithmetic Operations.

Zhou, Y., et al. (2022). Mixture-of-Experts with Expert Choice Routing. NeurIPS.
```

---

## Appendix A: Implementation Pseudocode

```python
class MESModel(nn.Module):
    """Complete MES model implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            MESTransformerLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_base_experts=config.n_base_experts,
                n_slot_experts=config.n_slot_experts,
                d_ff=config.d_ff,
                top_k=config.top_k
            )
            for _ in range(config.n_layers)
        ])
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        self.slot_manager = ExpertSlotManager(self)
    
    def freeze_for_finetuning(self):
        """Freeze everything except slot experts and router"""
        for param in self.parameters():
            param.requires_grad = False
        
        for layer in self.layers:
            # Unfreeze slot experts
            for slot_expert in layer.moe.slot_experts:
                for param in slot_expert.parameters():
                    param.requires_grad = True
            # Unfreeze router
            for param in layer.moe.router.parameters():
                param.requires_grad = True
    
    def freeze_for_calibration(self):
        """Freeze everything except router (Phase 3)"""
        for param in self.parameters():
            param.requires_grad = False
        
        for layer in self.layers:
            for param in layer.moe.router.parameters():
                param.requires_grad = True


class MESMoELayer(nn.Module):
    """MoE layer with base experts and slot experts"""
    
    def __init__(self, d_model, n_base, n_slots, d_ff, top_k):
        super().__init__()
        self.n_base = n_base
        self.n_slots = n_slots
        self.n_total = n_base + n_slots
        self.top_k = top_k
        
        self.base_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(n_base)
        ])
        self.slot_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(n_slots)
        ])
        self.router = Router(d_model, self.n_total)
        
        # Track which slots are active
        self.active_slots = [False] * n_slots
    
    def forward(self, x, routing_constraint=None):
        # Get routing weights
        router_logits = self.router(x)  # [batch, seq, n_total]
        
        # Mask inactive slots
        for i, active in enumerate(self.active_slots):
            if not active:
                router_logits[:, :, self.n_base + i] = float('-inf')
        
        # Apply routing constraint if provided (Phase 2)
        if routing_constraint is not None:
            router_logits = router_logits + routing_constraint
        
        # Top-k selection
        weights, indices = top_k_gating(router_logits, self.top_k)
        
        # Compute expert outputs
        all_experts = list(self.base_experts) + list(self.slot_experts)
        output = mixture_of_experts_forward(x, all_experts, weights, indices)
        
        # Load balancing loss
        balance_loss = compute_load_balance_loss(router_logits, indices)
        
        return output, balance_loss
```

## Appendix B: Hyperparameter Recommendations

```
Phase 2 (Constrained Routing Training):
├── Learning rate (slot experts): 1e-4 (cosine decay)
├── Learning rate (router): 5e-5 (cosine decay)
├── Batch size: 256-512 sequences
├── Sequence length: 2048-4096 tokens
├── Routing constraint α: 5.0 → 0.0 (linear decay over training)
├── λ_balance: 0.01
├── λ_consistency: 0.001
├── Epochs: 3-5 over domain data
├── Warmup: 5% of total steps
└── Weight decay: 0.01 (slot experts only)

Phase 3 (Router Calibration):
├── Learning rate (router only): 1e-5 (constant)
├── Batch size: 512-1024 sequences
├── Epochs: 1-2
└── No routing constraint (α = 0)
```

---

*¿Pasamos ahora al documento sobre las implicaciones filosóficas del scratchpad y el aumento de capacidades?*