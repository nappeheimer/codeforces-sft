from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_NAME = "sft_config.yaml"


def _find_config() -> Path | None:
    for path in (
        Path(__file__).resolve().parent.parent / _CONFIG_NAME,
        Path(_CONFIG_NAME),
    ):
        if path.exists():
            return path
    return None


def _load_yaml() -> dict:
    path = _find_config()
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass
class TrainConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    eos_token: str = "<|im_end|>"

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path: str = ""  # legacy fallback if curriculum_data_path unset
    curriculum_data_path: str = ""
    shuffle_data_path: str = ""
    eval_data_path: str = ""
    eval_split_ratio: float = 0.0
    curriculum_no_shuffle_epochs: int = 2

    # ── Sequence ─────────────────────────────────────────────────────────────
    max_seq_length: int = 32_768
    packing: bool = False

    # ── Optimisation ─────────────────────────────────────────────────────────
    learning_rate: float = 4.0e-5
    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    optimizer_type: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    warmup_ratio: float = 0.03

    # ── Evaluation ───────────────────────────────────────────────────────────
    do_eval: bool = False
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"

    # ── KL regularisation ────────────────────────────────────────────────────
    kl_beta: float = 0.1

    # ── Efficiency ───────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True
    bf16: bool = True
    use_liger_kernel: bool = False

    # ── Logging / saving ─────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "./checkpoints"
    hub_model_id: Optional[str] = None
    logging_steps: int = 10
    save_total_limit: Optional[int] = None  # keep every epoch checkpoint
    report_to: str = "tensorboard"
    wandb_project: str = "finecf-cots-sft"
    wandb_run_name: str = "qwen25-coder-7b-finecf-kl-reg"


def load_config() -> TrainConfig:
    raw = _load_yaml()
    valid = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in valid})
    # Resolve paths relative to config file directory.
    base = _find_config()
    root = base.parent if base else Path.cwd()

    def _resolve(p: str) -> str:
        if not p:
            return p
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((root / path).resolve())

    cfg.curriculum_data_path = _resolve(cfg.curriculum_data_path or cfg.data_path)
    cfg.shuffle_data_path = _resolve(cfg.shuffle_data_path or cfg.curriculum_data_path)
    cfg.eval_data_path = _resolve(cfg.eval_data_path)
    cfg.data_path = cfg.curriculum_data_path
    cfg.output_dir = _resolve(cfg.output_dir)
    return cfg
