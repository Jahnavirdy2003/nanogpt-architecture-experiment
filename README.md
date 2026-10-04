# Homework 1: nanoGPT Experiments

Character-level language modeling on Shakespeare, comparing the original nanoGPT model with five independent modifications. The repository also includes the Question 2 RoPE similarity plot code.

Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. The upstream MIT license is included in `LICENSE`.

## Models

| File | Modification from baseline |
|---|---|
| `model_baseline.py` | Original pre-LayerNorm, GELU, learned absolute positions, multi-head attention |
| `model_rmsnorm.py` | Replace LayerNorm with RMSNorm |
| `model_swiglu.py` | Replace GELU MLP with SwiGLU, hidden width 1024 to match projection weight count |
| `model_nope.py` | Remove learned positional embeddings |
| `model_rope.py` | Replace learned positional embeddings with RoPE on queries and keys, base 10000 |
| `model_gqa.py` | Six query heads share three key/value heads, group size two |

Each variant starts from the baseline; modifications are not cumulative. GQA explicitly repeats shared key/value tensors for compatibility with the attention implementation and does not realize all memory savings of an optimized GQA kernel.

## Setup

Run commands from the repository root. The experiments used a Google Colab Tesla T4 GPU. In Colab, use its installed GPU-enabled PyTorch; elsewhere, install a suitable PyTorch build for your device first.

```sh
python -m pip install numpy matplotlib requests tiktoken
python data/shakespeare_char/prepare.py
```

Preparation downloads Shakespeare and creates `train.bin`, `val.bin`, and `meta.pkl` under `data/shakespeare_char/`. There are 65 characters, 1,003,854 training tokens, and 111,540 validation tokens. No pretrained checkpoint is needed.

## Run an experiment

`train.py` imports `model.py`. Select a variant by copying its file to `model.py`, then launch training in a fresh process. The example below uses the baseline; replace `baseline` in the three filenames with `rmsnorm`, `swiglu`, `nope`, `rope`, or `gqa` to run another variant.

Linux or Colab terminal:

```sh
cp model_baseline.py model.py
python -u train.py config/train_shakespeare_char.py --dtype=float16 --compile=False --out_dir=out-baseline 2>&1 | tee baseline_training.log
```

Windows PowerShell with an appropriate NVIDIA GPU:

```powershell
Copy-Item model_baseline.py model.py
python -u train.py config/train_shakespeare_char.py --dtype=float16 --compile=False --out_dir=out-baseline 2>&1 | Tee-Object baseline_training.log
```

In a Colab code cell, prefix shell commands with `!` and change directories using `%cd`. Select and train one variant at a time. Running a command again overwrites its log; use a different output directory and log filename if preserving an earlier run.

For durable Colab results, use an output directory and log path under your mounted Google Drive, as demonstrated in the notebook. Checkpoints are intentionally omitted from this repository.

## Shared training settings

- Seed: 1337, single GPU.
- Training budget: `max_iters=5000` for every model.
- Six layers, six query heads, embedding width 384, context length 256.
- Batch size 64, gradient accumulation one, dropout 0.2.
- AdamW with the remaining defaults in `train.py` and `config/train_shakespeare_char.py`.
- Learning rate: 100 warm-up iterations, peak 0.001, cosine decay toward 0.0001 over the configured training budget.
- Float16 precision and `compile=False` for all runs.
- Evaluation every 250 iterations, averaging 200 batches per split with dropout disabled.
- No variant-specific learning-rate tuning.

The plotted training evaluation losses differ from the individual training-batch losses printed between evaluations. Exact numerical reproduction can vary with software versions and GPU kernels.

## Measured results

| Model | Best validation loss | Best step | Final training evaluation loss | Final validation loss |
|---|---:|---:|---:|---:|
| Baseline | 1.4677 | 1750 | 0.6188 | 1.7094 |
| RMSNorm | 1.4665 | 1750 | 0.6154 | 1.7168 |
| SwiGLU | 1.4953 | 1250 | 0.5188 | 1.8843 |
| NoPE | 1.5264 | 2750 | 1.0710 | 1.5753 |
| RoPE | 1.4741 | 1500 | 0.5436 | 1.7522 |
| GQA | 1.4570 | 2000 | 0.6248 | 1.7110 |

These are single-run comparisons. Small differences do not establish a consistent advantage across seeds.

## Logs, plots, and notebook

- `*_training.log`: saved training output.
- `*_losses.csv`: evaluation losses extracted from the logs.
- `positional_comparison.csv`: positional-encoding results summary.
- `baseline_loss_plot.png`, `baseline_vs_*.png`, and `positional_encoding_comparison.png`: loss curves.
- `Homework1_nanoGPT.ipynb`: experiment implementation, checks, training commands, plotting code, and saved outputs.

The notebook preserves the original interactive Colab workflow, including troubleshooting cells. It is not a one-click Run All notebook. Its paths refer to `/content/nanoGPT` and `/content/drive/MyDrive/Homework1_nanoGPT`. To regenerate plots from a local checkout, change the plotting cells' `folder` variable to `Path(".")`; the saved logs are sufficient and retraining is unnecessary. The model source files in this repository can be used directly without replaying notebook editing cells.

## Question 2 RoPE plot

```sh
python rope_plot.py
```

This saves `rope_attention.png` beside the script and displays the plot. It plots unscaled query-key similarity for distances 0 through 65536, with vector dimension 128 and RoPE base 10000.
