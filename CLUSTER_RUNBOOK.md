# Cluster runbook — Codeforces trace SFT

This document is for the person running training on the GPU cluster. Read it start to finish once before launching.

**Repository:** https://github.com/nappeheimer/codeforces-sft

---

## 0. Clone and download data (Git LFS — required)

```bash
git lfs install
git clone https://github.com/nappeheimer/codeforces-sft.git
cd codeforces-sft
git lfs pull
ls -lh data/*.jsonl    # each train file ~215M, test ~24M
```

If JSONL files are only ~100 bytes, run `git lfs pull` again.

---

## 1. What this job does

- **Base model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (downloaded automatically from Hugging Face on first run).
- **Training data:** JSONL chat traces (system + user problem + assistant thinking/code).
- **Schedule:**
  - **Epochs 1–2:** `data/train_curriculum.jsonl` — fixed curriculum order, **no shuffle**.
  - **Epochs 3–10:** `data/train.jsonl` — **shuffled** each epoch.
- **Checkpoints:** Saved **after every epoch** under `./checkpoints/checkpoint-*` for later benchmarking.
- **Eval during training:** **Off** — you benchmark saved checkpoints separately.
- **Loss:** Assistant tokens only + optional KL penalty (`kl_beta`) vs a frozen copy of the base model on CPU.

---

## 2. What is in this repo

Everything is in the GitHub repo (clone + `git lfs pull`):

| Item | Path |
|------|------|
| Training code | `run_sft.py`, `sft/`, `sft_config.yaml`, `accelerate_zero3.yaml`, `requirements.txt` |
| Curriculum JSONL | `data/train_curriculum.jsonl` (Git LFS, 8167 lines) |
| Shuffle JSONL | `data/train.jsonl` (Git LFS, 8167 lines) |
| Test JSONL | `data/test.jsonl` (benchmarking after training only) |

---

## 3. Hardware expectations

| Resource | Minimum guidance |
|----------|------------------|
| GPUs | **8× A100 80GB** (default: `num_processes: 8` in `accelerate_zero3.yaml`) |
| GPU VRAM | 32k `max_seq_length`, batch size 1 per GPU, ZeRO-3 — comfortable on 80GB A100s |
| Disk | ~50 GB free (model cache ~15 GB + checkpoints ~15–30 GB per epoch × 10) |
| CPU RAM | **≥ 64 GB** recommended; with `kl_beta: 0.1` each GPU process loads a **~14 GB** ref model on CPU → **~8 × 14 GB ≈ 112 GB** on an 8-GPU node. If RAM is tight, set `kl_beta: 0` in `sft_config.yaml`. |
| Network | Hugging Face access on first run (model download) |

---

## 4. Software setup (one time per machine)

### 4.1 Python

Use **Python 3.10 or 3.11** (3.12 often OK). Check:

```bash
python3 --version
```

### 4.2 Create environment (recommended)

```bash
cd codeforces-sft
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install PyTorch with CUDA

Match your cluster’s CUDA version. Example (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPUs:

```bash
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

### 4.4 Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Flash Attention (recommended, not required)

Faster training on A100:

```bash
pip install flash-attn --no-build-isolation
```

If install fails, training falls back to `sdpa` attention automatically.

### 4.6 Hugging Face token

Create a token at https://huggingface.co/settings/tokens (read access is enough).

```bash
export HF_TOKEN=hf_your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Optional: persist in `~/.bashrc`.

---

## 5. Configure GPUs

Edit **`accelerate_zero3.yaml`**:

```yaml
num_processes: 8   # must match number of GPUs you will use
```

For a quick smoke test on **1 GPU**, set `num_processes: 1` (DeepSpeed ZeRO-3 still works on one GPU).

---

## 6. Configure paths and hyperparameters

Edit **`sft_config.yaml`** only if needed. Defaults:

| Setting | Default | Meaning |
|---------|---------|---------|
| `curriculum_data_path` | `data/train_curriculum.jsonl` | Epochs 1–2 |
| `shuffle_data_path` | `data/train.jsonl` | Epochs 3–10 |
| `num_train_epochs` | `10` | Total epochs |
| `curriculum_no_shuffle_epochs` | `2` | No-shuffle phase length |
| `max_seq_length` | `32768` | Max tokens per example |
| `learning_rate` | `4e-5` | |
| `per_device_train_batch_size` | `1` | |
| `gradient_accumulation_steps` | `8` | Effective batch = GPUs × 1 × 8 |
| `kl_beta` | `0.1` | `0` disables KL / saves CPU RAM |
| `do_eval` | `false` | No validation during training |
| `output_dir` | `./checkpoints` | Where checkpoints go |
| `save_total_limit` | `null` | Keep **all** epoch checkpoints |
| `report_to` | `tensorboard` | Set to `none` if you do not want TB logs |

---

## 7. Preflight (do this before the long run)

Confirms JSONL paths, row counts, and message format **without** loading the 7B model:

```bash
cd codeforces-sft
source .venv/bin/activate   # if using venv
export HF_TOKEN=hf_...
python3 run_sft.py --check-only
```

Expected ending:

```text
All preflight checks passed. Ready to train.
```

If you see `PREFLIGHT FAILED`, fix paths under `data/` or `sft_config.yaml`.

---

## 8. Start training (single command)

```bash
cd codeforces-sft
source .venv/bin/activate
export HF_TOKEN=hf_...
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: long run in tmux/screen
# tmux new -s sft

python3 run_sft.py
```

### What you should see

1. `[launch] Multi-GPU (8 processes): ...` (or single-process if `num_processes: 1`)
2. `[dataset] Loading curriculum ...` / `Loaded 8167 rows`
3. `[train] Loading model ...` — **first run downloads ~15 GB from Hugging Face** (can take 10–30+ minutes)
4. `[KLReg] Loading reference model on CPU` (if `kl_beta > 0`)
5. `[train] === Phase 1/2: curriculum, epochs 1–2 ...`
6. Periodic `[train] step=... | loss=...`
7. `[train] Checkpoint saved ...` after each epoch
8. `[train] === Phase 2/2: shuffled, epochs 3–10 ...`
9. `[train] Done. Checkpoints are under: ...`

### How long it might take

Rough order of magnitude on **8× A100 80GB**, 8167 examples, 32k max length, 10 epochs: **many hours to a few days** depending on average sequence length and I/O. Watch `loss` and checkpoint writes to confirm progress.

---

## 9. Outputs after training

| Output | Location |
|--------|----------|
| Per-epoch checkpoints | `checkpoints/checkpoint-<step>/` (one per epoch; use for benchmarking) |
| Training state | `checkpoints/trainer_state.json` |
| Final merged weights | `checkpoints/` (from final `save_model`) |
| TensorBoard logs | `checkpoints/runs/` or similar under `output_dir` |

To benchmark epoch 5, for example, load:

```text
checkpoints/checkpoint-<step_for_epoch_5>/
```

(Step numbers depend on steps per epoch; `trainer_state.json` lists epoch ↔ step.)

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `HF_TOKEN is not set` | `export HF_TOKEN=hf_...` |
| `JSONL not found` | Put files in `data/` or fix paths in `sft_config.yaml` |
| CUDA OOM | Uncommon on 8×80GB with ZeRO-3; confirm `num_processes: 8`; lower `max_seq_length` only if needed |
| Host OOM (CPU RAM) | Set `kl_beta: 0` in `sft_config.yaml` |
| `flash_attn` errors | Ignore (sdpa fallback) or install flash-attn for CUDA match |
| NCCL timeout / hang | Check GPU interconnect; try `export NCCL_DEBUG=INFO` |
| Stuck after launch | Wait — first Hub download is slow; watch GPU `nvidia-smi` |
| `assistant_only_loss` / template error | Upgrade: `pip install -U trl transformers` |

---

## 11. Re-running / resuming

- **Fresh run:** Delete or rename `checkpoints/` (or change `output_dir` in yaml).
- **Resume:** Not automated in this runbook; phase 2 already resumes from phase 1 checkpoints internally. For a manual resume, use Hugging Face Trainer `resume_from_checkpoint` (contact the sender if needed).

---

## 12. Quick reference — copy/paste

```bash
cd codeforces-sft
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # adjust CUDA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation                            # optional
export HF_TOKEN=hf_...
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 run_sft.py --check-only
python3 run_sft.py
```

---

## 13. Contact / handoff checklist for the sender

Before shipping, confirm:

- [ ] `data/train_curriculum.jsonl` and `data/train.jsonl` are in the bundle
- [ ] `python3 run_sft.py --check-only` passes on a machine with the JSONL files
- [ ] `accelerate_zero3.yaml` `num_processes` matches recipient GPU count
- [ ] Recipient has `HF_TOKEN` and internet for first model download
- [ ] `CLUSTER_RUNBOOK.md` is included
