# Codeforces trace SFT — complete training package

Self-contained repo to fine-tune **Qwen/Qwen2.5-Coder-7B-Instruct** on Codeforces-style traces (`<think>` + `<code>`).

Everything needed to train is in this repository: **code**, **configs**, and **JSONL datasets** (stored with **Git LFS**).

---

## Repository layout

```
codeforces-sft/
├── README.md                 ← you are here
├── CLUSTER_RUNBOOK.md        ← detailed cluster / troubleshooting guide
├── run_sft.py                ← single entrypoint: python3 run_sft.py
├── sft_config.yaml           ← hyperparameters
├── accelerate_zero3.yaml     ← GPU count + DeepSpeed ZeRO-3
├── requirements.txt
├── data/
│   ├── train_curriculum.jsonl   (Git LFS, ~215 MB)
│   ├── train.jsonl              (Git LFS, ~215 MB)
│   └── test.jsonl               (Git LFS, ~24 MB, benchmarking only)
└── sft/                      ← trainer, dataset loader, config (required by run_sft.py)
```

---

## What training does

| Setting | Value |
|---------|--------|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` (auto-downloaded from Hugging Face) |
| Epochs | **10** total |
| Epochs 1–2 | `data/train_curriculum.jsonl`, **no shuffle** (curriculum order) |
| Epochs 3–10 | `data/train.jsonl`, **shuffled** each epoch |
| Checkpoints | Saved **every epoch** under `./checkpoints/checkpoint-*` |
| Eval during training | **Off** — use saved checkpoints + `test.jsonl` later |
| Loss | Assistant tokens only + optional KL (`kl_beta: 0.1`) |

---

## Step 0 — Clone and pull datasets (Git LFS)

JSONL files are too large for normal Git. You **must** have [Git LFS](https://git-lfs.github.com/) installed.

```bash
git lfs install
git clone https://github.com/nappeheimer/codeforces-sft.git
cd codeforces-sft
git lfs pull
```

Verify data (should show ~215M / ~215M / ~24M):

```bash
ls -lh data/*.jsonl
wc -l data/*.jsonl
# expect: 8167 train_curriculum, 8167 train, 894 test
```

If files are tiny (~130 bytes), LFS was not pulled — run `git lfs pull` again.

---

## Step 1 — Environment (cluster / GPU machine)

**Python 3.10 or 3.11** recommended.

```bash
cd codeforces-sft
python3 -m venv .venv
source .venv/bin/activate
```

**PyTorch with CUDA** (match your driver; example CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Dependencies:**

```bash
pip install -r requirements.txt
```

**Optional but recommended on A100:**

```bash
pip install flash-attn --no-build-isolation
```

**Hugging Face token** (model download on first run):

```bash
export HF_TOKEN=hf_your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Create a token: https://huggingface.co/settings/tokens (read access is enough).

---

## Step 2 — Configure GPUs

Edit **`accelerate_zero3.yaml`**:

```yaml
num_processes: 8   # must equal the number of GPUs you use
```

For a 1-GPU smoke test, set `num_processes: 1`.

---

## Step 3 — Preflight (no training yet)

Checks paths, row counts, and message format:

```bash
python3 run_sft.py --check-only
```

Expected last line:

```text
All preflight checks passed. Ready to train.
```

---

## Step 4 — Train (one command)

```bash
python3 run_sft.py
```

That is the only command needed. It will:

1. Launch **Accelerate + DeepSpeed ZeRO-3** when `num_processes > 1`
2. **Download** the base model from Hugging Face on first run (~15 GB, one-time)
3. Run **phase 1** (epochs 1–2, curriculum, no shuffle)
4. **Resume** and run **phase 2** (epochs 3–10, shuffled)
5. Save a checkpoint **after each epoch**

Use `tmux` or `screen` for long jobs.

---

## Step 5 — Outputs

| Artifact | Location |
|----------|----------|
| Per-epoch checkpoints | `checkpoints/checkpoint-<step>/` |
| Training metadata | `checkpoints/trainer_state.json` |
| Final weights | `checkpoints/` (top-level after `save_model`) |
| TensorBoard | under `checkpoints/` if `report_to: tensorboard` |

Use different `checkpoint-*` folders to benchmark each epoch later.

**`data/test.jsonl`** is not used during training — only for your downstream eval scripts.

---

## Hyperparameters (`sft_config.yaml`)

| Key | Default | Notes |
|-----|---------|--------|
| `num_train_epochs` | 10 | |
| `curriculum_no_shuffle_epochs` | 2 | |
| `learning_rate` | 4e-5 | cosine + 3% warmup |
| `max_seq_length` | 32768 | |
| `per_device_train_batch_size` | 1 | |
| `gradient_accumulation_steps` | 8 | effective batch ≈ GPUs × 8 |
| `kl_beta` | 0.1 | set **0** if host RAM tight (see below) |
| `do_eval` | false | |
| `output_dir` | `./checkpoints` | |
| `save_total_limit` | null | keep all epoch checkpoints |

---

## Hardware guidelines

| Resource | Guidance |
|----------|----------|
| GPUs | **8× A100 80GB** (`accelerate_zero3.yaml` → `num_processes: 8`) |
| Disk | ~50 GB+ free (model cache + checkpoints) |
| CPU RAM | **≥ 128 GB** host RAM if `kl_beta: 0.1` (~14 GB ref model per GPU process on CPU). Use `kl_beta: 0` if RAM-limited. |
| Network | Hugging Face access on first run |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Tiny JSONL after clone | `git lfs install && git lfs pull` |
| `HF_TOKEN is not set` | `export HF_TOKEN=hf_...` |
| CUDA OOM | Rare on 8× A100 80GB with ZeRO-3; confirm `num_processes: 8`; lower `max_seq_length` only if needed |
| Host RAM OOM | `kl_beta: 0` in `sft_config.yaml` |
| No flash-attn | OK — training uses `sdpa` automatically |

More detail: **[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md)**

---

## Quick copy-paste (full run)

```bash
git lfs install
git clone https://github.com/nappeheimer/codeforces-sft.git
cd codeforces-sft
git lfs pull
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export HF_TOKEN=hf_...
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 run_sft.py --check-only
python3 run_sft.py
```
