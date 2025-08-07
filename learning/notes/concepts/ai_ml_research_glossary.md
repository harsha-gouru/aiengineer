# AI/ML Research Glossary (Foundations → SOTA)
**Date:** 2025-08-07  
**Tags:** #ml #dl #llm #systems #privacy #multimodal #optimization #alignment  
**Status:** Draft

## Summary
A concise, practical glossary of concepts an ML researcher/engineer should know—from the Perceptron to modern LLM systems. Organized by theme with brief definitions and keywords to help you search deeper when needed.

## Contents
- Foundations & Classical ML
- Optimization & Training Mechanics
- Regularization, Compression, and Efficiency
- Neural Architectures and Building Blocks
- Positional Encodings & Long-Context Tricks
- LLM Training, Alignment, and Reasoning
- Retrieval, Knowledge, and Editing
- Multimodal Learning
- Inference & Decoding
- Systems, Parallelism, and Scaling
- Privacy, Security, and Safety
- Evaluation & Analysis
- Graphs & Knowledge Graphs
- Tokenization & Data
- Emerging and Related Topics

---

## Foundations & Classical ML
- Perceptron: Linear classifier trained by perceptron rule; foundation of neural networks.
- Linear/Logistic Regression: Linear predictors for regression/classification; convex objectives.
- Bias–Variance Trade-off: Decomposition of error; guides model complexity and regularization.
- VC Dimension: Capacity measure for statistical learning; bounds generalization.
- Maximum Likelihood / MAP: Parameter estimation via likelihood or posterior maximization.
- Naïve Bayes: Generative classifier assuming conditional independence of features.
- k-NN: Instance-based method using distance in feature space.
- SVM / Kernel Trick: Margin maximization with kernels (RBF, poly) for nonlinear separation.
- Decision Trees / Random Forests: Tree-based learners; ensembles reduce variance.
- Gradient Boosting (XGBoost/LightGBM/CatBoost): Sequential tree ensembles optimizing residuals.
- PCA/SVD/ICA: Linear dimensionality reduction and source separation.
- Clustering (k-means, GMM, EM): Unsupervised partitioning; mixture modeling via EM.
- Cross-Validation: Data splitting to estimate generalization (k-fold, LOOCV).
- Class Imbalance: Reweighting, resampling, focal loss, threshold tuning.
- Domain Shift (Covariate/Label/Concept): Mismatch between train and test data distributions.
- Transfer Learning: Reuse of pretrained representations on new tasks/domains.
- Meta-Learning (MAML/Reptile): Learn to adapt quickly to new tasks with few samples.

## Optimization & Training Mechanics
- Empirical Risk Minimization: Minimize average loss over data; core training paradigm.
- SGD / Mini-batch SGD: Stochastic optimization; scales to large datasets.
- Momentum / Nesterov: Accelerated gradient methods reducing oscillations.
- Adaptive Optimizers: Adagrad, RMSProp, Adam, AdamW (decoupled weight decay), AdaFactor, LAMB, Lion.
- Weight Decay: L2 regularization promoting smaller weights and generalization.
- Learning-Rate Schedules: Step, exponential, cosine decay, warmup, OneCycle.
- Gradient Clipping: Limit gradient norm/value to stabilize training.
- Mixed Precision: FP16/BF16/FP8 training for speed/memory; maintain stability with loss scaling.
- Gradient Accumulation: Simulate larger batches across steps to fit memory budgets.
- Gradient Checkpointing: Recompute activations in backward pass to reduce memory.
- Loss Functions: CE, MSE, Huber, focal loss, contrastive/NT-Xent, triplet loss.
- Label Smoothing: Soft targets to improve calibration and robustness.
- Initialization: Xavier/Glorot, Kaiming/He; affects training stability.
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm; stabilize and speed up training.
- Gating Activations: GeLU, SwiGLU/GeGLU; improved expressivity in FFNs.

## Regularization, Compression, and Efficiency
- Dropout / DropPath / Stochastic Depth: Randomly drop activations/paths to regularize.
- Early Stopping: Halt training when validation stops improving to avoid overfitting.
- Data Augmentation: Mixup, CutMix, RandAugment, AutoAugment, SpecAugment (audio), CutOut.
- Pruning (Structured/Unstructured): Remove weights/filters; movement pruning; Lottery Ticket Hypothesis.
- Quantization: PTQ/QAT; INT8/INT4/FP8; AWQ, GPTQ, AQLM, HQQ; BitsAndBytes NF4; W8A8/W4A16.
- Distillation: Transfer knowledge from larger teacher to smaller student (logits/features/attn).
- SLoRA / Hybrid LoRA-Quant: Combine low‑rank adaptation with quantized base for efficiency.
- Parameter Sharing: Reuse parameters across layers (e.g., ALBERT) to reduce size.
- Parameter Efficient FT (PEFT): Fine‑tune small adapters only (LoRA, IA³, BitFit, DoRA, AdaLoRA).
- Low-Rank Adapters (LoRA): Inject low‑rank matrices into weight updates; memory‑efficient FT.
- IA³ / BitFit: Scale vectors on attention/FFN; bias-only fine-tuning for minimal params.
- Adapter Modules: Bottleneck layers inserted between blocks for task-specific adaptation.

## Neural Architectures and Building Blocks
- MLP: Fully connected networks; universal approximators.
- CNNs: Convolutions for vision; ResNet (skip connections), EfficientNet (compound scaling), MobileNet (depthwise separable), ConvNeXt.
- RNN/LSTM/GRU: Sequential models with gating to address long-term dependencies.
- Transformer: Self-attention + FFN with residuals and normalization; de facto LLM backbone.
- Sparse Attention: Reduce O(N^2) to linear/subquadratic via patterns (Longformer, BigBird, Reformer, Linformer, Performer, Nyströmformer).
- Mixture of Experts (MoE): Sparse expert routing (Switch/GShard); improves parameter count vs compute.
- Reversible Layers: Invertible residual blocks enabling lower memory backprop.
- Parameter Sharing: Cross-layer weight tying (ALBERT) to shrink model size.
- State-Space Models: S4, RWKV, Mamba (selective SSM) offering long-range modeling without attention.
- Perceiver / Perceiver-IO: Latent array with cross-attention; scalable to high‑dimensional inputs and multimodal I/O.
- Graph Neural Networks: GCN, GraphSAGE, GAT; message passing over graphs.
- Vision Transformers (ViT/Swin): Transformers for vision; windowed attention and hierarchical features.

## Positional Encodings & Long-Context Tricks
- Absolute/Learned PE: Sinusoidal or learned embeddings for token positions.
- Relative Positioning: Improves generalization to longer sequences.
- ALiBi: Position-aware bias enabling length extrapolation.
- RoPE: Rotary embeddings rotating Q/K; used in many LLMs.
- xPos: Stable extrapolation variant for long contexts.
- Position Interpolation: Rescale positions to extend context at inference.
- NTK Scaling: Adjust embedding frequencies to extend context length.
- YaRN/LongRoPE: Training/finetune recipes for very long context windows.
- Multi-Query / Grouped-Query Attention: Share K/V across heads to reduce memory/latency.

## LLM Training, Alignment, and Reasoning
- Pretraining Objectives: Causal LM, Masked LM, span corruption (T5), denoising.
- Instruction Tuning (SFT): Supervised finetuning on instruction–response pairs.
- RLHF: SFT → reward modeling → policy optimization (PPO); align with human preferences.
- Constitutional AI: Rule‑based self‑critique and refinement without human preference labels.
- DPO/IPO/KTO/ORPO: Preference-optimization methods avoiding online RL while matching preferences.
- Self-Play / Self-Training: Generate synthetic data (self-instruct) to improve capabilities.
- Chain-of-Thought (CoT): Train/infer with intermediate natural language “rationales.”
- Tree-of-Thought (ToT): Explore branching reasoning paths, evaluate and select best path.
- Graph-of-Thought (GoT): Reasoning as a graph with reusable sub-results.
- Self-Consistency Voting: Sample multiple reasoning paths, vote for final answer.
- ReAct: Interleave reasoning with actions/tool invocations; enables tool use.
- Toolformer-Style / Function Calling: Models learn to call tools/APIs with structured arguments.
- Program-of-Thought / PAL: Delegate subproblems to code execution (e.g., Python) for reliability.

## Retrieval, Knowledge, and Editing
- Retrieval-Augmented Generation (RAG): Retrieve relevant docs and ground model outputs.
- Indexing: HNSW, IVF/IVFPQ, ScaNN, FAISS; approximate nearest neighbors (ANN).
- Chunking & Reranking: Document segmentation; cross-encoder rerankers for precision.
- Knowledge Graph Integration: Use KG triples / GNNs / path reasoning to augment LLMs.
- Knowledge Distillation: Compress teachers into efficient students; logits/feature distill.
- Knowledge Editing: ROME, MEMIT, MEND, SERAC, EasyEdit; modify specific facts without retraining.

## Multimodal Learning
- Vision–Language: CLIP (contrastive), BLIP/BLIP‑2, Flamingo, LLaVA; Q‑Former, resamplers.
- Audio / Speech: wav2vec 2.0 (SSL), Whisper (ASR); HuBERT; alignment with text.
- Video: TimeSformer, ViViT; temporal attention.
- Perceiver‑IO: Unified latent interface for arbitrary input/output modalities.
- Multi-Modal Training: Joint or staged pretraining across text, image, audio, video.

## Inference & Decoding
- Greedy / Beam Search: Deterministic decoding; beam trades speed for quality/diversity.
- Sampling: Top‑k, nucleus (top‑p), temperature; repetition/penalty, typical decoding.
- Contrastive Decoding: Use a weaker model to penalize “boring” or generic outputs.
- Speculative Decoding / Draft-and-Verify: Fast model proposes tokens, larger model verifies; Medusa/Lookahead variants.
- Early Exiting / Adaptive Computation: Halt per‑token/layer when confident to save compute.
- KV-Cache: Store past keys/values to achieve O(1) per-token inference; reuse across steps.
- KV-Cache Compression & Flash-Decoding / PagedAttention: Reduce memory bandwidth; vLLM‑style paged cache, continuous batching.
- FlashAttention: IO‑aware attention kernels for faster training/inference.

## Systems, Parallelism, and Scaling
- Data Parallelism: Replicate model across devices; aggregate gradients.
- Model Parallelism: Split model across devices (tensor parallel, pipeline parallel).
- Tensor Parallelism: Shard tensor dimensions (Megatron‑LM) to fit larger models.
- Pipeline Parallelism: Partition layers into stages; microbatching to keep devices busy.
- ZeRO (Stage 1–3): Shard optimizer states/gradients/parameters; offload to CPU/NVMe.
- FSDP: Fully Sharded Data Parallel; shard parameters/activations efficiently.
- Sequence/Context Parallelism: Partition along sequence for long context scaling.
- Gradient Checkpointing: Activation rematerialization to cut memory costs.
- Mixed Precision & Quantized Inference: FP16/BF16/FP8; INT8/INT4 for deployment.
- TensorRT‑LLM / xFormers / CUTLASS: High‑performance kernels/libraries for LLMs.
- Throughput Optimizations: Continuous batching, chunked prefills, speculative decoding.
- Distributed Communication: NCCL, NVLink/InfiniBand; all‑reduce/all‑gather patterns.

## Privacy, Security, and Safety
- DP‑SGD / Differential Privacy: Add noise+clipping to gradients to bound privacy loss (ε, δ); Rényi DP.
- PATE: Teacher ensembles for private student training.
- Federated Learning: On‑device training with secure aggregation; personalization and DP.
- Watermarking & Cryptographic Key Insertion: Embed detectable signals to identify model outputs.
- Red‑Teaming & Guardrails: Safety eval, jailbreak resistance, content filters, policy enforcement.
- Secure Inference: Enclave/HE/SMPC approaches for private model serving (costly today).

## Evaluation & Analysis
- Language Tasks: Perplexity, token accuracy, BLEU/ROUGE/METEOR/CIDEr, BERTScore.
- LLM Benchmarks: MMLU, HELM, BIG‑bench, GSM8K, TruthfulQA, HellaSwag, ARC, HumanEval/MBPP.
- Preference Eval: Pairwise comparisons, win‑rate, Elo; model‑as‑judge caveats.
- Uncertainty & Calibration: Temperature scaling, ECE, conformal prediction.
- Explainability: SHAP, LIME, Integrated Gradients, attention rollout.
- Data Quality: Deduplication, contamination checks, filtering (toxicity/PII), data curation.

## Graphs & Knowledge Graphs
- KG Embeddings: TransE/DistMult/ComplEx/RotatE; learnable representations of triples.
- GNNs on KGs: Path reasoning, link prediction; hybrid KG‑RAG pipelines.
- Graph Augmented Reasoning: Use graph structure to guide retrieval/reasoning.

## Tokenization & Data
- Tokenization: BPE, SentencePiece (Unigram), WordPiece; byte‑level tokenization.
- Vocabulary & Segmentation: Lowercasing, normalization, special tokens, control codes.
- Data Mixtures: Proportions across web/code/math/sci; curriculum and temperature sampling.
- Long‑Context Data: Packing, sliding windows, extrapolation recipes; position scaling.

## Generative Models
- VAEs: Latent variable models trained with ELBO; reparameterization trick; β‑VAE for disentanglement.
- GANs: Generator–discriminator minimax game; variants (WGAN‑GP, StyleGAN, BigGAN); mode collapse and stabilization.
- Diffusion Models: Denoising diffusion probabilistic models; score-based SDEs; classifier‑free guidance; latent diffusion; ControlNet/T2I‑Adapters; consistency models.
- Normalizing Flows: Invertible transformations with tractable Jacobians (RealNVP, Glow); exact likelihood.
- Energy‑Based Models: Unnormalized densities trained via contrastive divergence/score matching; computationally heavy sampling.
- Autoregressive Decoders (Vision/Audio): PixelCNN/PixelSNAIL, Wavenet; tokenized images/audio with VQ‑VAE/codec models.
- Guidance & Conditioning: Classifier/classifier‑free guidance, CFG scale, prompt‑to‑image conditioning, LoRA for diffusion backbones.

## Reinforcement Learning Essentials
- MDPs/Bandits: States, actions, rewards, transitions; episodic vs continuing; contextual bandits.
- Value‑Based: Q‑learning, DQN (Double/Dueling/PER/Rainbow), distributional RL.
- Policy‑Based: REINFORCE; Actor‑Critic; PPO (clipped objective); TRPO; A3C/A2C; entropy regularization.
- Continuous Control: DDPG/TD3, SAC (maximum entropy RL).
- Model‑Based & World Models: Learn dynamics/latent world; MPC/planning; Dreamer.
- Offline RL: Learning from static datasets; behavioral regularization, CQL/IQL.
- Imitation/IRL: Behavior cloning, GAIL, inverse RL.
- RL for LLMs: RLHF/RLAIF; preference optimization alternatives (DPO/IPO/KTO/ORPO) relate to offline RL with preferences.

## Causal ML & OOD Robustness
- SCMs/DAGs: Pearl’s structural causal models, do‑calculus, backdoor/frontdoor criteria, interventions/counterfactuals.
- Invariant Risk Minimization (IRM): Learn predictors stable across environments; related: Risk extrapolation (REx).
- Distribution Shift & DG: Group DRO, CORAL, MMD; domain adversarial training (DANN); test‑time adaptation (TENT).
- Counterfactual Evaluation: Uplift modeling, causal forests, doubly robust estimation.

## Adversarial Robustness & Security
- Attacks: FGSM/PGD/CW, spatial/patch attacks; transfer attacks; jailbreaks for LLMs.
- Defenses: Adversarial training, input denoising, randomized smoothing; certified robustness (IBP/CROWN/DeepPoly).
- Privacy Attacks: Membership inference, model inversion; defenses via DP‑SGD, regularization.
- Supply‑Chain Risks: Data poisoning, dependency compromise; dataset filters/dedup and SBOMs.

## Hyperparameter Optimization & AutoML
- Search: Grid vs random; Bayesian optimization (GP‑EI/PI/UCB, TPE), SMAC; multi‑fidelity (Hyperband/ASHA/BOHB).
- Population‑Based Training (PBT): Evolve hyperparams during training.
- NAS: Reinforcement learning/evolution/grad‑based (DARTS); search spaces and weight sharing.

## Scaling Laws & Data-Centric AI
- Scaling Laws: Parameter/data/compute trade‑offs (Kaplan) vs data‑optimal (Chinchilla); compute/throughput aware.
- Double Descent & Grokking: Non‑monotonic generalization with model/epoch scaling; delayed comprehension.
- Data Quality: Deduplication, contamination checks, PII/toxicity filtering, temperature/mixture design, curriculum.
- Dataset/Tokenizer Design: Byte‑level vs wordpiece, vocab size, multilingual balance.

## Emerging and Related Topics
- State‑Space Advances: Mamba variants, RWKV improvements, Hyena/RetNet‑like hybrids.
- Long‑Context Training: YaRN/LongRoPE, attention sinks, windowed KV reuse.
- Efficiency: FP8 training, 4‑bit QLoRA; activation/weight quantization research.
- Preference Optimization: DPO/IPO/ORPO/KTO refinements; rejection sampling finetuning (RSF).
- Knowledge Editing at Scale: Multi‑fact edits, side‑effect mitigation.
- Multimodal LLMs: Vision‑language agents with tool use and planning.

---

## Quick Cross‑Reference of User‑Requested Terms
- Adapter Modules → Regularization/Compression/Efficiency
- Chain‑of‑Thought → LLM Training/Alignment/Reasoning
- Contrastive Learning → Optimization/Losses; Multimodal (e.g., CLIP)
- DP‑SGD / Differential Privacy → Privacy/Security
- Early Exiting / Adaptive Computation → Inference & Decoding
- Fine‑tuning → PEFT/Full FT; Optimization sections
- FlashAttention → Inference & Systems
- Gradient Checkpointing → Optimization; Systems
- Graph‑of‑Thought → LLM Reasoning
- Instruction Tuning → LLM Training
- IA³ / BitFit → PEFT
- Knowledge Distillation → Regularization/Compression
- Knowledge Editing (ROME/MEMIT/EasyEdit) → Retrieval/Knowledge/Editing
- Knowledge Graph Integration → Graphs & KGs
- KV‑Cache Compression & Flash‑Decoding / PagedAttention → Inference & Systems
- Mixture of Experts → Architectures; Systems
- Multi‑Modal Training → Multimodal
- Parameter Sharing → Architectures; Efficiency
- Perceiver‑IO → Architectures; Multimodal
- Positional Encoding Tricks (ALiBi, RoPE, Position Interpolation, xPos, NTK Scaling) → Positional Encodings
- Prefix‑/Prompt‑Tuning → PEFT
- Pruning (Structured/Unstructured) → Regularization/Compression
- Prompt Engineering → LLM Reasoning & Inference
- Quantization → Regularization/Compression; Systems
- ReAct / Function‑Calling / Toolformer‑style → LLM Reasoning & Tools
- Retrieval‑Augmented Generation → Retrieval
- RLHF & Constitutional AI → LLM Alignment
- Self‑Consistency Voting → LLM Reasoning
- Self‑Supervised Learning → Optimization/Objectives (contrastive, masked, denoising)
- SLoRA / Hybrid LoRA‑Quant → PEFT & Compression
- Sparse Attention → Architectures
- Speculative Decoding / Draft‑and‑Verify → Inference
- State‑Space Models (Mamba, RWKV) → Architectures
- Tensor Parallelism → Systems
- Transformer → Architectures
- Tree‑of‑Thought → LLM Reasoning
- ZeRO (1‑3) & FSDP → Systems
- Watermarking & Cryptographic Key‑Insertion → Safety/Security
- Federated LLM → Privacy/Security
- Pipeline Parallelism → Systems
- Reversible Layers → Architectures/Efficiency

---

## Notes
- This file aims to be concise. For deeper dives, create focused notes under `learning/notes/concepts/` (e.g., `flashattention.md`, `rag_system_design.md`).
- Last major audit: 2025‑08‑07.
