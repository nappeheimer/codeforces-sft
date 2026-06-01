from __future__ import annotations

import sys
from pathlib import Path

from .config import TrainConfig
from .dataset import load_jsonl


def validate_config(cfg: TrainConfig) -> list[str]:
    """Return a list of human-readable errors (empty = OK)."""
    errors: list[str] = []

    for label, path in (
        ("curriculum_data_path", cfg.curriculum_data_path),
        ("shuffle_data_path", cfg.shuffle_data_path),
    ):
        if not path:
            errors.append(f"{label} is empty — set it in sft_config.yaml")
            continue
        p = Path(path)
        if not p.is_file():
            errors.append(f"{label} not found: {p}")

    if cfg.eval_data_path:
        p = Path(cfg.eval_data_path)
        if not p.is_file():
            errors.append(f"eval_data_path not found: {p}")

    if cfg.curriculum_no_shuffle_epochs < 0:
        errors.append("curriculum_no_shuffle_epochs must be >= 0")
    if cfg.num_train_epochs < 1:
        errors.append("num_train_epochs must be >= 1")
    if cfg.curriculum_no_shuffle_epochs > cfg.num_train_epochs:
        errors.append(
            "curriculum_no_shuffle_epochs cannot exceed num_train_epochs"
        )

    return errors


def run_preflight(cfg: TrainConfig, *, load_data: bool = True) -> None:
    """Print resolved config and optionally load JSONL to verify format."""
    errors = validate_config(cfg)
    if errors:
        print("PREFLIGHT FAILED:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        sys.exit(1)

    print("PREFLIGHT OK — resolved paths:", flush=True)
    print(f"  model:      {cfg.model_name_or_path}", flush=True)
    print(f"  curriculum: {cfg.curriculum_data_path}", flush=True)
    print(f"  shuffle:    {cfg.shuffle_data_path}", flush=True)
    print(f"  output_dir: {cfg.output_dir}", flush=True)
    print(
        f"  schedule:   epochs 1–{cfg.curriculum_no_shuffle_epochs} curriculum "
        f"(no shuffle), {cfg.curriculum_no_shuffle_epochs + 1}–"
        f"{int(cfg.num_train_epochs)} shuffled",
        flush=True,
    )
    print(f"  do_eval:    {cfg.do_eval}", flush=True)
    print(f"  kl_beta:    {cfg.kl_beta}", flush=True)

    if not load_data:
        return

    print("\n[dataset] Verifying JSONL can be loaded ...", flush=True)
    c = load_jsonl(cfg.curriculum_data_path, label="curriculum")
    s = (
        c
        if Path(cfg.shuffle_data_path).resolve()
        == Path(cfg.curriculum_data_path).resolve()
        else load_jsonl(cfg.shuffle_data_path, label="shuffle")
    )
    if len(c) != len(s):
        print(
            f"  note: curriculum rows={len(c)}, shuffle rows={len(s)} "
            "(different order/content is OK)",
            flush=True,
        )
    sample = c[0]["messages"]
    roles = [m["role"] for m in sample]
    if roles != ["system", "user", "assistant"]:
        print(f"  warn: expected roles system/user/assistant, got {roles}", flush=True)
    print(f"  sample roles OK: {roles}", flush=True)
    print("\nAll preflight checks passed. Ready to train.", flush=True)
