# AI/ML Topics TL;DR (Practitioner Notes)
**Date:** 2025-08-07  
**Tags:** #ml #dl #llm #systems #privacy #multimodal #optimization  
**Status:** Draft

## Purpose
A just-enough-to-use guide. For each topic: What it is, When to use, How to apply quickly, Pitfalls. If you need depth, see the detailed glossary at `ai_ml_research_glossary.md`.

Legend: [What] / [Use When] / [How] / [Pitfalls]

---

## Parameter-Efficient Fine-Tuning (PEFT)
- Adapter Modules — Lightweight layers inserted per block.  
  [What] Small bottlenecks enable task adaption without touching full weights.  
  [Use When] Limited GPU, multiple tasks on one base model.  
  [How] Use PEFT libs (HuggingFace `peft`) with `Adapter` or `LoRA`.  
  [Pitfalls] Wrong target modules; overfitting with tiny data.
- LoRA / DoRA — Low‑rank updates on W matrices (DoRA adds better scaling).  
  [What] Factorized updates reduce trainable params.  
  [Use When] You want strong results at low memory cost.  
  [How] `peft` with LoRA config; set `r`, `alpha`, `target_modules`.  
  [Pitfalls] Too small `r` hurts quality; mis-specified target layers.
- IA³ / BitFit — Scale vectors only / bias-only FT.  
  [What] Ultra‑light PEFT variants.  
  [Use When] Extreme parameter budget constraints.  
  [How] `peft` IA3/BitFit.  
  [Pitfalls] May underfit for complex tasks.
- Prefix-/Prompt-Tuning — Learn soft prompts.  
  [What] Train continuous prompts with frozen model.  
  [Use When] Many tasks, minimal memory.  
  [How] `peft` Prompt/Prefix.  
  [Pitfalls] Needs careful prompt length; weaker than LoRA on some tasks.

## Training & Optimization
- Fine‑tuning (full)  
  [What] Update all weights.  
  [Use When] You own strong compute and large task data; need max quality.  
  [How] Standard FT with optimizer (AdamW), LR schedule, mixed precision.  
  [Pitfalls] Catastrophic forgetting; catastrophic cost.
- Gradient Checkpointing  
  [What] Trade compute for memory by recomputing activations.  
  [Use When] OOM on long context or big batch.  
  [How] Enable framework flag (`model.gradient_checkpointing_enable()`).  
  [Pitfalls] Slower wall‑clock; interacts with certain ops.
- DP‑SGD / Differential Privacy  
  [What] Clip/noise gradients to bound privacy loss.  
  [Use When] Sensitive data.  
  [How] Opacus / TF Privacy; set clipping norm, noise multiplier; track ε.  
  [Pitfalls] Quality drop; tune carefully.
- Self‑Supervised Learning (contrastive/masked)  
  [What] Learn from unlabeled data.  
  [Use When] Labels scarce; pretraining.  
  [How] CLIP/SimCLR/MAE recipes; strong augmentations.  
  [Pitfalls] Aug choices matter; collapse without negatives/regularization.
- Knowledge Distillation  
  [What] Compress teacher into student.  
  [Use When] Latency/size constraints.  
  [How] Match logits/features; KD loss with temperature.  
  [Pitfalls] Data mismatch; teacher errors propagate.
- Pruning (Structured/Unstructured)  
  [What] Remove weights/filters to shrink/accelerate.  
  [Use When] Deployment with tight budgets.  
  [How] Movement pruning / magnitude pruning; retrain after pruning.  
  [Pitfalls] Accuracy cliff if too aggressive.

## Reasoning & Alignment
- Chain‑of‑Thought (CoT)  
  [What] Let model write intermediate steps.  
  [Use When] Math/logic/multi‑step tasks.  
  [How] Prompt with "let’s think step by step" or finetune with rationales.  
  [Pitfalls] Longer latency; leaking rationales may not generalize.
- Tree‑/Graph‑of‑Thought (ToT/GoT)  
  [What] Explore multiple reasoning paths and select.  
  [Use When] Hard reasoning; planning.  
  [How] Sample branches + scoring; tool or framework support.  
  [Pitfalls] Costly; requires good scoring heuristics.
- Self‑Consistency Voting  
  [What] Sample N solutions; vote.  
  [Use When] Reduce reasoning variance.  
  [How] N>3 samples, majority vote; rerank with verifier if available.  
  [Pitfalls] Higher cost; diminishing returns.
- Instruction Tuning  
  [What] SFT on instruction–response pairs.  
  [Use When] Align model to follow tasks.  
  [How] Curate instructions; SFT with RLHF or DPO later.  
  [Pitfalls] Data quality > size; avoid contamination.
- RLHF / Constitutional AI / DPO/IPO/ORPO/KTO  
  [What] Align outputs via preferences or rule‑based self‑critique.  
  [Use When] Safety, helpfulness, preference fit.  
  [How] SFT → RM → PPO, or offline preference optimization (DPO‑style).  
  [Pitfalls] Reward hacking; instability; evaluation is tricky.
- ReAct / Function‑Calling / Toolformer‑style  
  [What] Interleave reasoning with tool use.  
  [Use When] Retrieval, code, calculators, APIs.  
  [How] Define tool schemas; structured function calling; routing prompts.  
  [Pitfalls] Tool error handling; prompt injection.

## Retrieval, Knowledge & Editing
- Retrieval‑Augmented Generation (RAG)  
  [What] Retrieve docs to ground outputs.  
  [Use When] Factuality + freshness.  
  [How] FAISS/HNSW, chunking + reranker; prompt with retrieved context.  
  [Pitfalls] Bad chunking; poor retrieval overwhelms model.
- Knowledge Editing (ROME/MEMIT/EasyEdit)  
  [What] Modify specific facts in model.  
  [Use When] Small targeted corrections.  
  [How] Apply editing algorithms/tools; verify side effects.  
  [Pitfalls] Local edits can cause global drift.
- Knowledge Graph Integration  
  [What] Use KG triples/paths to augment reasoning/retrieval.  
  [Use When] Structured factual domains.  
  [How] KG→features or KG‑RAG with GNN/reranker.  
  [Pitfalls] KG coverage/quality limits.

## Inference & Efficiency
- KV‑Cache / PagedAttention / Flash‑Decoding  
  [What] Cache and page K/V to cut memory bandwidth; continuous batching.  
  [Use When] Long prompts; high throughput.  
  [How] Use vLLM/TensorRT‑LLM; enable paged KV.  
  [Pitfalls] Framework compatibility; memory fragmentation.
- FlashAttention  
  [What] IO‑aware attention kernel.  
  [Use When] Speed up training/inference.  
  [How] Use libraries (xFormers/FlashAttn); ensure dtypes compat.  
  [Pitfalls] Kernel/device/version mismatch.
- Speculative Decoding / Draft‑and‑Verify  
  [What] Small model drafts; big model verifies; speedup.  
  [Use When] Latency matters; quality must match.  
  [How] Enable in serving stack (e.g., vLLM/TensorRT‑LLM).  
  [Pitfalls] Mismatch hurts acceptance rate; tuning needed.
- Early Exiting / Adaptive Computation  
  [What] Stop compute when confident.  
  [Use When] Tight latency budgets.  
  [How] Confidence thresholds per token/layer.  
  [Pitfalls] Quality variance; calibration required.
- Quantization (W8A8/W4A16/INT8/INT4/NF4)  
  [What] Lower precision weights/activations.  
  [Use When] Fit on smaller GPUs/CPUs; speedup.  
  [How] AWQ/GPTQ/AutoGPTQ, bitsandbytes 8‑bit, `QLoRA` for training.  
  [Pitfalls] Outlier channels; calibration data needed.
- Pruning  
  [What] Remove redundant weights/channels.  
  [Use When] Size/speed constraints.  
  [How] Magnitude/movement pruning + short finetune.  
  [Pitfalls] Hardware support varies.

## Architectures & Building Blocks
- Transformer  
  [What] Self‑attention + FFN backbone.  
  [Use When] Default for LLMs/seq tasks.  
  [How] Use proven configs; tune LR, context, tokenizer.  
  [Pitfalls] O(N²) attention scaling.
- Sparse Attention  
  [What] Subquadratic attention patterns.  
  [Use When] Long context with limited memory.  
  [How] Longformer/BigBird/Performer/FlashAttn‑2.  
  [Pitfalls] Quality vs speed trade‑offs.
- Mixture of Experts (MoE)  
  [What] Route tokens to experts; sparse compute.  
  [Use When] More parameters without linear compute growth.  
  [How] Switch/MoE layers; balance load with auxiliary loss.  
  [Pitfalls] Routing instability; comms overhead.
- State‑Space Models (Mamba/RWKV)  
  [What] Attention alternatives with long‑range capacity.  
  [Use When] Long sequences, throughput constraints.  
  [How] Use model libs; tune state dims, scan kernels.  
  [Pitfalls] Maturity varies; kernels matter.
- Perceiver‑IO  
  [What] Latent array with cross‑attention; modality‑agnostic.  
  [Use When] Multimodal/high‑dim inputs/outputs.  
  [How] Use latent size/bottleneck carefully.  
  [Pitfalls] Tuning latent/hops critical.
- Parameter Sharing  
  [What] Tie weights across layers (ALBERT‑style).  
  [Use When] Shrink params with small quality hit.  
  [How] Use configs enabling tying.  
  [Pitfalls] May limit expressivity.
- Reversible Layers  
  [What] Invertible blocks reduce activation storage.  
  [Use When] Memory is bottleneck.  
  [How] Use reversible residuals.  
  [Pitfalls] Numerical issues; limited support.

## Systems & Parallelism
- Tensor Parallelism  
  [What] Shard tensor dims across GPUs.  
  [Use When] Model too big for a single GPU.  
  [How] Megatron‑LM/DeepSpeed configs.  
  [Pitfalls] All‑reduce overhead; careful mapping.
- Pipeline Parallelism  
  [What] Split layers into stages.  
  [Use When] Deep models on multi‑GPU.  
  [How] Microbatches to fill pipeline.  
  [Pitfalls] Bubbles; balancing stages.
- Data Parallel / ZeRO (1‑3) / FSDP  
  [What] Shard states/params/gradients across devices.  
  [Use When] Memory scaling for large models.  
  [How] DeepSpeed ZeRO; PyTorch FSDP.  
  [Pitfalls] Checkpointing/sharding configs; CPU/NVMe offload stability.

## Multimodal & Retrieval
- Multi‑Modal Training  
  [What] Jointly learn text+image+audio+video.  
  [Use When] Cross‑modal tasks.  
  [How] CLIP‑style contrastive or BLIP/LLaVA with Q‑Former/resamplers.  
  [Pitfalls] Alignment/data curation crucial.
- Prompt Engineering  
  [What] Craft prompts/system messages/tools.  
  [Use When] Quick wins without training.  
  [How] Few‑shot, role hints, constraints, explicit formats.  
  [Pitfalls] Brittle; evaluate carefully.

## Privacy, Security & Deployment
- Watermarking & Cryptographic Key‑Insertion  
  [What] Embed detectable signals in outputs/weights.  
  [Use When] Output provenance; IP protection.  
  [How] Use watermark libs; bias token sampling subtly.  
  [Pitfalls] Robustness vs detectability trade‑off.
- Federated LLM  
  [What] On‑device/client training with aggregator.  
  [Use When] Data cannot centralize.  
  [How] FedAvg variants + DP; secure aggregation.  
  [Pitfalls] Heterogeneous data; stragglers; comms limits.

## Positional & Long-Context Tricks
- ALiBi / RoPE / xPos / Position Interpolation / NTK Scaling  
  [What] Different ways to encode or extend positions.  
  [Use When] Need longer contexts or better generalization.  
  [How] Choose RoPE by default; extend with NTK/PI; use LongRoPE/YaRN recipes.  
  [Pitfalls] Mismatch between train/infer scales; requires careful eval.

---

## Quick Practitioner Checklist
- Start with PEFT (LoRA) + 4/8‑bit quantization for most FT tasks.
- Use RAG before heavy finetuning if knowledge freshness/factuality matters.
- Enable FlashAttention and KV‑cache in serving; consider paged KV (vLLM) for long prompts.
- For latency: speculative decoding; for throughput: continuous batching.
- For large models: ZeRO‑3/FSDP + gradient checkpointing; consider tensor/pipeline parallel if needed.
- Prefer DPO‑style preference optimization unless strict RLHF is required.
- Add adversarial/data quality checks; track contamination; deduplicate.

For deeper details, see `ai_ml_research_glossary.md`.
