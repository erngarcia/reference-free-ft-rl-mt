# Reference-Free RL Fine-Tuning for MT

This repository contains the experimental setup for:

> **Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective**
> Anonymous Submission
> [arXiv:2605.15976](https://doi.org/10.48550/arXiv.2605.15976) [cs.CL]

Production machine translation relies overwhelmingly on encoder-decoder Seq2Seq models, yet reinforcement learning approaches to MT fine-tuning have largely targeted decoder-only LLMs at ≥7B parameters, with limited systematic study of encoder-decoder architectures. We apply Group Relative Policy Optimization (GRPO) to NLLB-200 (600M and 1.3B) using a hybrid reference-free reward (LaBSE and COMET-Kiwi) that requires no parallel data at fine-tuning time, evaluating across 13 typologically diverse languages. GRPO yields consistent improvements on all 13 languages, up to +5.03 chrF++ for Traditional Chinese, and, without any target-language data, competes with 3-epoch supervised fine-tuning on morphologically complex languages. We identify a consistent empirical pattern in which gains are largest where baseline performance is weakest and reward discriminability is highest, making this approach most effective precisely where parallel data is scarcest, and replicate this pattern across English and Spanish source languages.

## Overview

The core of this repository is [train_grpo.py](train_grpo.py), which fine-tunes NLLB-200 models with GRPO using LoRA adapters and 4-bit quantization. For each hypothesis sampled from the policy, reward is computed reference-free from the source sentence alone (no target-language references needed), combining:

- **LaBSE** — cross-lingual sentence-embedding similarity between source and hypothesis
- **COMET-Kiwi** (`Unbabel/wmt22-cometkiwi-da`) — reference-free MT quality estimation

Several reward shaping modes are supported (standard, soft-rank, contrastive against a greedy baseline, and their combination), along with an adaptive sampling-temperature controller that raises exploration temperature on reward plateaus. Final evaluation is reference-based, using chrF++, BLEU, COMET-22, and BERTScore against held-out FLORES-200 data.

## Experiments

Configuration files correspond to the experiments in the paper:

| Config | Model | Training data |
|---|---|---|
| [experiments_configs/exp_a_600m.yaml](experiments_configs/exp_a_600m.yaml) | NLLB-200 distilled 600M | FLORES-200 dev |
| [experiments_configs/exp_a_13b.yaml](experiments_configs/exp_a_13b.yaml) | NLLB-200 distilled 1.3B | FLORES-200 dev |
| [experiments_configs/exp_b_600m.yaml](experiments_configs/exp_b_600m.yaml) | NLLB-200 distilled 600M | CC-News (10k English sentences) |
| [experiments_configs/exp_b_13b.yaml](experiments_configs/exp_b_13b.yaml) | NLLB-200 distilled 1.3B | CC-News (10k English sentences) |

All configs target the same 13 languages used in the paper: Basque, Swahili, Tibetan, Turkish, Japanese, Traditional Chinese, Simplified Chinese, Yoruba, Modern Standard Arabic, Belarusian, Bengali, Czech, and Polish.

## Usage

```bash
python train_grpo.py --config experiments_configs/exp_a_600m.yaml
```

CLI arguments override config values, e.g. to run a single language:

```bash
python train_grpo.py --config experiments_configs/exp_a_600m.yaml --languages zho_Hant
```

Each run writes to `results/{tag}/`, including the resolved config, per-step training metrics, LoRA adapters (final, best, and optionally a fixed-step checkpoint), and an evaluation summary against vanilla NLLB-200.

## Ablations

- [sft_ablation.py](sft_ablation.py) (config: [ablations_configs/sft_ablation.yaml](ablations_configs/sft_ablation.yaml)) — trains supervised fine-tuning (SFT) baselines with the same LoRA/quantization setup as GRPO, then compares vanilla vs SFT vs GRPO with paired bootstrap significance testing.
- [ablation_data_size.py](ablation_data_size.py) (config: [ablations_configs/ablation_data_size.yaml](ablations_configs/ablation_data_size.yaml)) — the training-data-size ablation (Table 3 in the paper): trains vanilla/SFT/GRPO for each language across N ∈ {100, 250, 500, 1000} training sentences and reports chrF/COMET-22 with significance tests, reusing the training and evaluation routines from `train_grpo.py` and `sft_ablation.py`.

```bash
python ablation_data_size.py --config ablations_configs/ablation_data_size.yaml
```

## Citation

```bibtex
@article{anonymous2026referencefree,
  title   = {Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective},
  author  = {Anonymous Submission},
  journal = {arXiv preprint arXiv:2605.15976},
  year    = {2026}
}
```
