# Research log

How to write an entry (4 lines is enough, one entry per work session):

- **Did:** what I ran or read
- **Result:** the numbers or what I learned
- **Surprise / mistake:** what I got wrong or didn't expect
- **Next:** the one thing I'll do next

Write predictions *before* running an experiment, never after seeing the numbers.

Dates marked (approx.) were reconstructed afterwards. Correct them if you remember better.

## Open questions

1. Is the fine-tuning update ΔW actually low-rank, and how does its spectrum compare with the pretrained weights?
2. What explains the remaining ~0.6-point gap between LoRA and full fine-tuning: rank, which matrices get adapters, learning rate, or training length?
3. Does LoRA's lower eval loss mean it is better calibrated than full fine-tuning?
4. Is the error from 4-bit quantisation low-rank, and can an adapter repair it?

---

## Oct 19–22 (approx.): LoRA paper and toy checks

- **Did:** Read the LoRA paper, wrote notes in my own words, corrected them after feedback. Ran a toy gradient check of the initialisation.
- **Result:** LoRA freezes the pretrained weight W and learns a low-rank update ΔW = BA. B starts at zero and A is random, so ΔW = 0 at step 0. With both at zero the gradients were (0, 0) and nothing could ever move. With A random and B zero they were (0, 37.99), so B learns first and A follows.
- **Surprise / mistake:** My notes said LoRA compresses the pretrained weights. It does not: W stays full size and only the small update is learned. I also mixed up parameters and dimensions (175B is GPT-3's parameter count).
- **Next:** Check how low-rank a real pretrained weight matrix is.

## Oct 19–22 (approx.): SVD of a pretrained weight, and a quantisation artifact

- **Did:** Took the SVD of one GPT-2 MLP weight matrix (768 × 3072) and tried naive 4-bit quantisation with a single scale for the whole matrix.
- **Result:** The pretrained weight is nearly full-rank: relative error 0.97 at rank 4, 0.85 at rank 64, 0.59 at rank 256. Naive 4-bit quantisation gave 0.92 error.
- **Surprise / mistake:** The 0.92 was an artifact. One scale for the whole matrix lets a few outlier weights set the scale, so most weights rounded to zero. Real QLoRA quantises in small blocks. Also, LoRA only claims ΔW is low-rank, not W, so this SVD tested the wrong thing for LoRA.
- **Next:** Build the real experiment: fine-tune, then look at ΔW.

## Oct 24: Repo set up, full vs LoRA baselines at 1 epoch

- **Did:** Created the repo, wrote `src/train.py`, ran full fine-tuning and LoRA (r = 8) on RoBERTa-base / SST-2, seeds 0, 1, 2, on Colab.
- **Result:** Full fine-tuning 93.92 ± 0.23%, 124.6M trainable parameters, 2.62 GB peak. LoRA 92.85 ± 0.07%, 887,042 trainable parameters (294,912 in the adapters, 592,130 in the classification head), 1.39 GB peak. The gap is about 1.1 points, larger than the seed spread. LoRA's final training loss was higher (about 0.215 vs 0.175).
- **Surprise / mistake:** Memory fell by 47%, more than I expected. Setup bugs: the dataset name needed the `nyu-mll/glue` form, a stray file named `results` blocked the output folder, and Colab's preinstalled `torchao` was incompatible with PEFT (fixed by uninstalling it). All three are fixed or documented.
- **Next:** Test whether more training closes the gap (hypothesis: LoRA is under-trained at 1 epoch).

## Oct 24–25: 3-epoch runs

- **Did:** Reran both modes for 3 epochs, seeds 0, 1, 2. Added `EPOCHS` and `eval_loss` logging to the script.
- **Result:** Full 94.30 ± 0.48%, LoRA 93.69 ± 0.00%. The gap shrank from 1.07 to 0.61 points. LoRA gained 0.84 points, full fine-tuning gained 0.38. LoRA's eval loss (about 0.20) was lower than full fine-tuning's (about 0.25) in all three seeds.
- **Surprise / mistake:** All three LoRA seeds scored exactly 817/872, which comes from the small validation set (one example = 0.115 points), so a standard deviation of 0.00 is not real stability. My first edit of `train.py` had a duplicated argument and a variable used before it was defined; fixed.
- **Next:** Under-training explains part of the gap, not all. Look at ΔW directly.

## Oct 25: ΔW spectrum run

- **Did:** Reran full fine-tuning (seed 0, 3 epochs) with code that saves the singular values of ΔW and of the pretrained weights for the query and value matrices. The run reproduced the earlier seed-0 numbers exactly.
- **Prediction (write this BEFORE looking at the analysis output):** *the top 8 singular values of ΔW hold about ___% of its energy, versus about ___% for the pretrained W.*
  If you already saw the numbers, write: "no prediction recorded beforehand".
- **Result:** *(fill in: top-8 energy of ΔW and of W0, and what the plot shows)*
- **Surprise / mistake:** *(fill in)*
- **Next:** Rank sweep (r = 1, 4, 8, 16, 32) to check whether rank 8 is the bottleneck.
