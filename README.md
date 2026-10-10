# Holographic-attention
Accelerating Induction Head formation in Transformers via learned Holographic Phase Modulation in QK-heads. (100M scale results)

## Overview and Findings
Standard dot-product attention can become noisy when features are compressed in high-dimensional superposition. By rotating Query and Key heads in a complex phase space, this method allows the model to disentangle head interactions. 

I tested this on a 100M parameter scale, training for 0.5B tokens on the FineWeb-Edu dataset. The results indicate a notable structural shift in how the model learns:

* **The Induction crossover:** The phase model has a sharp phase transition around Step 5000. Interestingly, it achieves an induction score 2.7x higher than the matched baseline.
* **Attention Sharpness:** The holographic runs consistently show lower attention entropy.
* **Optimization Signal:** The learned `Phase Norm` grew from 0.0 to 4.0, so the optimizer really relies on the phase mechanism to minimize loss.

## Results
We ran controlled tests comparing a standard baseline (phase_mult = 0.0) with two different variants (0.05 and 0.15 multipliers). 

| Metric (at step 5k-8k) | Baseline | Holo-Strong (0.15) | Observation |
| :--- | :--- | :--- | :--- |
| **Induction Score** | 0.0039 | **0.0106** | Has better circuit formation |
| **Attention Entropy** | 3.3385 | **3.3011** | Tighter feature packing |
| **Validation Loss** | 3.6167 | **3.6061** | Consistent efficiency gap |

## Repository Structure
- `model.py`: Core model definition.
- `train.py`: Training script`.
- `dataset.py`: Tokenization and streaming for FineWeb-Edu dataset.
- `logs/`: CSV training logs for all experiments.

**Contact:** DM me on X ([@fanat50399337](https://x.com/fanat50399337)), or find me on the MI Discord server
