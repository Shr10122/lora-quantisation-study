# LoRA vs full fine-tuning on RoBERTa-base (SST-2)

A small, reproducible empirical study of parameter efficient fine tuning. The longer term question this repo builds toward:

> **Is the error introduced by 4-bit quantisation of a pretrained model low-rank, and can a LoRA-style adapter repair it?**

This is not a new method. Stage 1 measures a standard comparison carefully (several seeds, two training lengths, memory and time reported). Later stages test LoRA's low-rank assumption directly and then the quantisation question.

**Status:** Stage 1 (baselines) done · Stage 2 (spectrum of the fine-tuning update ΔW) in progress · Stage 3 (rank sweep, quantised base model) planned

---

## Setup

| Item | Value |
|---|---|
| Model | `roberta-base` |
| Task | SST-2 (GLUE, `nyu-mll/glue`), binary sentiment |
| Evaluation | GLUE validation split, 872 examples (test labels are hidden). One example = 0.115 accuracy points |
| Training | batch size 32, max length 128, fp16, 1 or 3 epochs, seeds 0, 1, 2 |
| Full fine-tuning | learning rate 2e-5 |
| LoRA | rank r = 8, alpha = 8, dropout 0.1, applied to `query` and `value`, learning rate 5e-4 (not tuned) |
| Hardware | Google Colab free-tier GPU |

The LoRA learning rate and rank were set once and not tuned, so the comparison favours neither side by search effort.

## Results

Accuracy on the validation set, mean ± standard deviation over 3 seeds.

| | 1 epoch | 3 epochs | Eval loss (3 ep) | Train time (3 ep) | Trainable params | Peak GPU memory |
|---|---|---|---|---|---|---|
| Full fine-tuning | 93.92 ± 0.23 | 94.30 ± 0.48 | 0.246 | ~535 s | 124,647,170 | 2.62 GB |
| LoRA, r = 8 | 92.85 ± 0.07 | 93.69 ± 0.00 | 0.203 | ~360 s | 887,042 | 1.39 GB |

Per-seed accuracy (%):

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Full, 1 epoch | 94.15 | 93.92 | 93.69 |
| LoRA, 1 epoch | 92.78 | 92.89 | 92.89 |
| Full, 3 epochs | 94.15 | 94.84 | 93.92 |
| LoRA, 3 epochs | 93.69 | 93.69 | 93.69 |

Trainable parameters for LoRA: 294,912 in the adapters (12 layers × 2 matrices × 2 × 768 × 8) plus 592,130 in the classification head, which is trained in full. That is 0.71% of full fine-tuning's count, or 0.24% for the adapters alone.

### What the measurements show

- At 1 epoch, LoRA trails full fine-tuning by about 1.1 points, and the gap is larger than the seed to seed spread.
- Training 3× longer narrows the gap to about 0.6 points (about 5 of 872 examples). LoRA gained 0.84 points and full fine-tuning gained 0.38, so under-training explains part of the gap but not all of it. The remaining gap is suggestive, not conclusive, with 3 seeds.
- LoRA used 47% less peak memory and about a third less time per run. It was not faster to reach a given accuracy: full fine-tuning reached 93.92% in about 170 s at 1 epoch, while LoRA needed about 360 s for 93.69%.
- LoRA had a lower eval loss than full fine-tuning in all three seeds despite lower accuracy. One possible explanation (untested) is that full fine-tuning is more overconfident when wrong.

### Limitations

- One model, one dataset, one rank, one learning rate per method. Nothing here supports claims beyond this setting.
- Three seeds, and a validation set where one example moves accuracy by 0.115 points. A standard deviation of 0.00 for LoRA at 3 epochs reflects this granularity (all three seeds got 817/872 correct), not perfect stability.
- Eval loss was not recorded in the 1-epoch runs, so the loss comparison covers 3 epochs only.

## Reproduce

```bash
pip install transformers datasets peft accelerate
# On Colab, if PEFT complains about an incompatible torchao version:
pip uninstall -y torchao

# usage: python src/train.py <full|lora> <seed> [rank]    (EPOCHS env var, default 1)
EPOCHS=3 python src/train.py full 0
EPOCHS=3 python src/train.py lora 0 8
```

Each run writes `results/<mode>_<seed>_r<rank>_e<epochs>.json` with accuracy, trainable parameters and peak memory. Full fine-tuning runs also write `results/dW_sv_full_<seed>_e<epochs>.npz`, the singular values of the update ΔW and of the pretrained weights for the query and value matrices.

Seed 0 is deterministic here: re-running it reproduced the same accuracy and eval loss to every printed digit.

Provenance note: the six 1-epoch JSON files were recreated from the printed training logs after a Colab session reset. The 3-epoch files are original script output. Exact library versions are in `requirements.txt`.

## Repository layout

```
src/train.py        training and evaluation script
results/            one JSON file per run, plus ΔW singular values
RESEARCH_LOG.md     dated log of experiments, predictions and findings
requirements.txt    pinned library versions
```

## Roadmap

1. **Spectrum of ΔW** (in progress): fine-tune fully, then take the singular values of ΔW = W_finetuned − W_pretrained for the attention matrices and compare with the pretrained weights. This tests LoRA's central assumption that the update is low-rank.
2. **Rank sweep** (planned): r ∈ {1, 4, 8, 16, 32} at 3 epochs, to check whether rank 8 is the bottleneck behind the remaining gap.
3. **Quantised base model** (planned): block-wise 4-bit quantisation of the pretrained weights, then ask how low-rank the quantisation residual is and how much of it an adapter can repair.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023.
- Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*, 2018.
