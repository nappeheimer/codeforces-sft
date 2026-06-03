#!/usr/bin/env python3
"""
SFT launcher and trainer entry point.

Run everything with one command from this directory:

    export HF_TOKEN=hf_...
    python3 run_sft.py

Phase 1 (epochs 1–N): train_curriculum.jsonl, sequential order (no shuffle).
Phase 2 (epochs N+1–total): train.jsonl, shuffled each epoch.

Hyperparameters: sft_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from sft import TrainConfig, load_config, run_preflight

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parent


def find_file(name: str) -> Path:
    for path in (_REPO_ROOT / name, Path(name)):
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {name!r} (looked in {_REPO_ROOT} and cwd)")


def _accelerate_num_processes(accel_cfg: Path) -> int:
    data = yaml.safe_load(accel_cfg.read_text(encoding="utf-8")) or {}
    return int(data.get("num_processes", 1))


def launch(args: argparse.Namespace) -> None:
    """Re-exec under accelerate when num_processes > 1; else train in-process."""
    accel_cfg = find_file("accelerate_zero3.yaml")
    num_procs = _accelerate_num_processes(accel_cfg)

    if num_procs <= 1:
        print("[launch] Single-process mode (num_processes=1)", flush=True)
        os.environ.setdefault("LOCAL_RANK", "0")
        train(_load_cfg_from_args(args))
        return

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accel_cfg),
        str(_REPO_ROOT / "run_sft.py"),
    ] + sys.argv[1:]
    print(f"[launch] Multi-GPU ({num_procs} processes): {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.run(cmd, check=False).returncode)


def _patch_chat_template_for_assistant_loss(tokenizer) -> None:
    """
    Inject {% generation %} / {% endgeneration %} into the tokenizer's Jinja2 chat template.

    Qwen2.5-Coder merges user, system, and assistant (no-tool-calls) into one generic branch:
        {%- if (message.role == "user") or ... or (message.role == "assistant" and not message.tool_calls) %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    TRL cannot detect assistant tokens there. We split that branch so the assistant arm gets
    its own block with {% generation %} markers wrapping only the content.
    """
    tpl = tokenizer.chat_template
    if not tpl or "{% generation %}" in tpl:
        return

    old = (
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first)'
        ' or (message.role == "assistant" and not message.tool_calls) %}\n'
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    )
    new = (
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n'
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
        '    {%- elif message.role == "assistant" and not message.tool_calls %}\n'
        "        {{- '<|im_start|>assistant\\n' }}{% generation %}{{- message.content + '<|im_end|>\\n' }}{% endgeneration %}"
    )

    if old not in tpl:
        raise RuntimeError(
            "Could not patch chat template: expected Qwen2.5 assistant branch not found. "
            "Inspect tokenizer.chat_template and update _patch_chat_template_for_assistant_loss."
        )
    tokenizer.chat_template = tpl.replace(old, new)
    print("[train] Chat template patched: added {% generation %} markers.", flush=True)


def _maybe_login() -> None:
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
    else:
        print(
            "[warn] HF_TOKEN not set — using cached/local Hugging Face credentials if any",
            flush=True,
        )


def _build_sft_config(cfg: TrainConfig, *, num_train_epochs: float):
    from trl import SFTConfig

    save_limit = cfg.save_total_limit
    return SFTConfig(
        output_dir=cfg.output_dir,
        hub_model_id=cfg.hub_model_id,
        push_to_hub=cfg.hub_model_id is not None,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optimizer_type,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.lr_scheduler_kwargs,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        fp16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        do_eval=cfg.do_eval,
        eval_strategy=cfg.eval_strategy if cfg.do_eval else "no",
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        load_best_model_at_end=cfg.load_best_model_at_end and cfg.do_eval,
        metric_for_best_model=cfg.metric_for_best_model,
        max_length=cfg.max_seq_length,
        packing=cfg.packing,
        assistant_only_loss=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=cfg.logging_steps,
        save_total_limit=save_limit,
        report_to=cfg.report_to,
        remove_unused_columns=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )


def _create_trainer(
    cfg: TrainConfig,
    *,
    sft_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    model,
    shuffle_enabled: bool,
):
    from sft.trainer import KLRegSFTTrainer

    return KLRegSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        ref_model_name_or_path=cfg.model_name_or_path,
        kl_beta=cfg.kl_beta,
        shuffle_enabled=shuffle_enabled,
        use_lora=cfg.use_lora,
    )


def train(cfg: TrainConfig, *, resume_from_checkpoint: str | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sft import build_datasets

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        run_preflight(cfg, load_data=True)

    if is_main:
        print("=" * 72, flush=True)
        print("[train] Codeforces trace SFT", flush=True)
        print(f"[train] Model: {cfg.model_name_or_path}", flush=True)
        print(f"[train] Output: {cfg.output_dir}", flush=True)
        print(f"[train] Total epochs: {int(cfg.num_train_epochs)}", flush=True)
        print(
            f"[train] Curriculum epochs (no shuffle): {cfg.curriculum_no_shuffle_epochs}",
            flush=True,
        )
        print("=" * 72, flush=True)

    _maybe_login()

    print("[train] Loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    tokenizer.eos_token = cfg.eos_token
    eos_id = tokenizer.convert_tokens_to_ids(cfg.eos_token)
    if eos_id != tokenizer.unk_token_id:
        tokenizer.eos_token_id = eos_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _patch_chat_template_for_assistant_loss(tokenizer)

    if cfg.use_liger_kernel:
        try:
            import importlib

            liger = importlib.import_module("liger_kernel.transformers")
            liger.apply_liger_kernel_to_qwen2()
            print("[train] Liger kernel applied.", flush=True)
        except ImportError:
            print("[train] liger-kernel not installed, skipping.", flush=True)

    attn_impl = "flash_attention_2"
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError
    except ImportError:
        attn_impl = "sdpa"
        print("[train] flash-attn not installed; using sdpa attention.", flush=True)

    print(f"[train] Loading model (bf16, {attn_impl}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation=attn_impl,
    )
    print("[train] Model loaded.", flush=True)

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # enable_input_require_grads is required for gradient checkpointing + PEFT
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()
            print("[train] LoRA adapters applied.", flush=True)

    curriculum_ds, shuffle_ds, eval_dataset = build_datasets(cfg)
    do_eval = cfg.do_eval and eval_dataset is not None

    total_epochs = int(cfg.num_train_epochs)
    curriculum_epochs = min(cfg.curriculum_no_shuffle_epochs, total_epochs)
    shuffle_epochs = total_epochs - curriculum_epochs

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # One trainer for both phases so phase-2 init does not wipe epoch checkpoints.
    sft_args = _build_sft_config(cfg, num_train_epochs=float(total_epochs))
    setattr(sft_args, "overwrite_output_dir", True)
    trainer = _create_trainer(
        cfg,
        sft_args=sft_args,
        train_dataset=curriculum_ds,
        eval_dataset=eval_dataset if do_eval else None,
        tokenizer=tokenizer,
        model=model,
        shuffle_enabled=(curriculum_epochs == 0),
    )

    # ── Phase 1: curriculum, no shuffle ───────────────────────────────────────
    if curriculum_epochs > 0:
        if is_main:
            print(
                f"\n[train] === Phase 1/2: curriculum, epochs 1–{curriculum_epochs} "
                "(sequential, no shuffle) ===\n",
                flush=True,
            )
        trainer.args.num_train_epochs = float(curriculum_epochs)
        # trainer already has curriculum_ds tokenized from __init__ — don't overwrite with raw
        trainer.set_shuffle_enabled(False)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if is_main:
            print("[train] Phase 1 complete.", flush=True)

    # ── Phase 2: shuffled training to total_epochs ────────────────────────────
    if shuffle_epochs > 0:
        if is_main:
            print(
                f"\n[train] === Phase 2/2: shuffled, epochs "
                f"{curriculum_epochs + 1}–{total_epochs} ===\n",
                flush=True,
            )
        trainer.args.num_train_epochs = float(total_epochs)
        trainer.set_train_dataset(shuffle_ds)
        trainer.set_shuffle_enabled(True)
        # Pass the resume checkpoint to Phase 2 only when Phase 1 was skipped
        # (curriculum_epochs == 0). When Phase 1 ran, its train() already consumed
        # the checkpoint and Phase 2 continues from the in-memory state.
        phase2_resume = resume_from_checkpoint if curriculum_epochs == 0 else None
        trainer.train(resume_from_checkpoint=phase2_resume)
        if is_main:
            print("[train] Phase 2 complete.", flush=True)

    if is_main:
        print("[train] Saving final weights and tokenizer ...", flush=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("[train] Done. Checkpoints are under:", cfg.output_dir, flush=True)


def _load_cfg_from_args(args: argparse.Namespace) -> tuple[TrainConfig, str | None]:
    cfg = load_config()
    if args.data_path:
        p = str(Path(args.data_path).resolve())
        cfg.curriculum_data_path = p
        cfg.data_path = p
    if not cfg.curriculum_data_path:
        raise ValueError(
            "Set curriculum_data_path in sft_config.yaml or pass --data_path"
        )
    resume = getattr(args, "resume", None)
    return cfg, resume


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run Codeforces trace SFT")
    parser.add_argument(
        "--data_path",
        default=None,
        help="Override curriculum_data_path (legacy alias)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config and JSONL paths only; do not train",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume from a checkpoint directory, e.g. ./checkpoints/checkpoint-200",
    )
    args = parser.parse_args()

    if args.check_only:
        cfg, _ = _load_cfg_from_args(args)
        run_preflight(cfg, load_data=True)
        return

    if not os.environ.get("HF_TOKEN"):
        print(
            "[warn] HF_TOKEN is not set — will use cached HF credentials if the model "
            "is already downloaded. If this is a fresh node, set: export HF_TOKEN=hf_...",
            flush=True,
        )

    # Top-level invocation: launch via accelerate or train on one GPU.
    if "LOCAL_RANK" not in os.environ:
        launch(args)
        return

    cfg, resume = _load_cfg_from_args(args)
    train(cfg, resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
