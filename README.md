# Holographic-attention
Accelerating Induction Head formation in Transformers via learned Holographic Phase Modulation in QK-heads. (100M scale results)

This repository contains the implementation and preliminary experimental results for **Holographic Phase Modulation**—an architectural modification designed to mitigate feature interference in attention heads through learned phase rotations.

## Overview and Key Findings
Standard dot-product attention can become noisy when features are compressed in high-dimensional superposition. By rotating Query and Key heads in a complex-valued phase space, this method provides a geometric prior that allows the model to "orthogonalize" head interactions. 

I tested this hypothesis on a 100M parameter scale, training for 0.5B tokens on the FineWeb-Edu dataset. The results indicate a notable structural shift in how the model learns:

* **The Induction crossover:** The phase-modulated model initially tracks the baseline but experiences a much sharper phase transition around Step 5000. Ultimately, it achieves an induction score **2.7x higher** than the matched baseline.
* **Attention Sharpness:** The holographic runs consistently show lower attention entropy, suggesting the formation of more specialized and sparse attention circuits.
* **Optimization Signal:** The learned `Phase Norm` grew steadily from 0.0 to 4.0, confirming that the optimizer actively relies on the phase mechanism to minimize loss.

## Experimental Results
We ran controlled A/B/C tests comparing a standard baseline (`phase_mult = 0.0`) with two holographic variants (`0.05` and `0.15` multipliers). 

| Metric (at Step 5k-8k) | Baseline | Holo-Strong (0.15) | Observation |
| :--- | :--- | :--- | :--- |
| **Induction Score** | 0.0039 | **0.0106** | Accelerated circuit formation |
| **Attention Entropy** | 3.3385 | **3.3011** | Tighter feature packing |
| **Validation Loss** | 3.6167 | **3.6061** | Consistent efficiency gap |

## Repository Structure
- `model.py`: Core model definition (Holographic Causal Self-Attention, SwiGLU, RMSNorm).
- `train.py`: Distributed training script using Hugging Face `accelerate`.
- `dataset.py`: Tokenization and streaming logic for FineWeb-Edu.
- `logs/`: Raw CSV training logs for all experiments to ensure reproducibility.

## About the Author & Collaboration
I am a 13 year old independent researcher with a strong interest in Mechanistic Interpretability and transformer internals. 

This is an early-stage proof of concept, and I am currently looking for:
- **Mentorship** to help formalize the mathematical framework behind these phase circuits.
- **Compute Collaborators** to test if these scaling laws hold at the 1B+ parameter scale on A100/H100 clusters.
