from __future__ import annotations

import logging
from collections.abc import Sized
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class ProgressPrintCallback(TrainerCallback):
    """Rank-0 heartbeat logs so long runs do not look stuck."""

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(
                f"[train] Starting: {args.num_train_epochs} epochs, "
                f"output_dir={args.output_dir}",
                flush=True,
            )

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(f"[train] >>> Epoch {int(state.epoch) + 1} beginning ...", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if state.global_step and state.global_step % max(args.logging_steps, 1) == 0:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            parts = [f"step={state.global_step}"]
            if loss is not None:
                parts.append(f"loss={loss:.4f}")
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            if "train/kl_loss" in logs:
                parts.append(f"kl={logs['train/kl_loss']:.4f}")
            print(f"[train] {' | '.join(parts)}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(
                f"[train] <<< Epoch {int(state.epoch)} finished "
                f"(global_step={state.global_step})",
                flush=True,
            )

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(
                f"[train] Checkpoint saved at step {state.global_step} "
                f"under {args.output_dir}",
                flush=True,
            )


class KLRegSFTTrainer(SFTTrainer):
    """
    SFTTrainer with per-token forward-KL on assistant tokens only.

    Total loss = SFT CE (assistant_only_loss via TRL)
               + kl_beta * mean KL(pi_theta || pi_ref) on assistant tokens

    Set shuffle_enabled=False for curriculum phases (sequential sampler).
    """

    def __init__(
        self,
        *args,
        ref_model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        kl_beta: float = 0.1,
        shuffle_enabled: bool = True,
        **kwargs,
    ):
        self.kl_beta = kl_beta
        self._ref_model: Optional[torch.nn.Module] = None
        self._ref_model_name_or_path = ref_model_name_or_path
        self.shuffle_enabled = shuffle_enabled

        super().__init__(*args, **kwargs)
        self.add_callback(ProgressPrintCallback())

        if kl_beta > 0:
            self._load_ref_model()

    def set_shuffle_enabled(self, enabled: bool) -> None:
        """Switch sampler mode and force dataloader rebuild on next step."""
        self.shuffle_enabled = enabled
        self._train_dataloader = None

    def set_train_dataset(self, dataset) -> None:
        self.train_dataset = dataset
        self._train_dataloader = None

    def _load_ref_model(self) -> None:
        if self.args.local_rank <= 0:
            print(
                f"[KLReg] Loading reference model on CPU: {self._ref_model_name_or_path}",
                flush=True,
            )
        self._ref_model = AutoModelForCausalLM.from_pretrained(
            self._ref_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        if self.args.local_rank <= 0:
            print("[KLReg] Reference model ready.", flush=True)

    @torch.no_grad()
    def _ref_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self._ref_model is not None
        out = self._ref_model(input_ids=input_ids.cpu())
        return out.logits.to(dtype=torch.float32, device=input_ids.device)

    @staticmethod
    def _assistant_mask(inputs: dict) -> torch.Tensor:
        labels = inputs.get("labels")
        if labels is not None:
            return labels != -100
        return inputs["attention_mask"].bool()

    def _compute_kl(
        self,
        train_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_p = F.log_softmax(train_logits, dim=-1)
        log_q = F.log_softmax(ref_logits, dim=-1)
        per_token_kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        per_token_kl = per_token_kl * token_mask
        return per_token_kl.sum() / token_mask.sum().clamp(min=1)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.shuffle_enabled:
            return super()._get_train_sampler(train_dataset)

        if self.args.world_size > 1:
            from torch.utils.data import DistributedSampler

            return DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
                seed=self.args.seed,
            )
        return SequentialSampler(cast(Sized, dataset))

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        sft_loss: torch.Tensor
        outputs: Any = None

        if return_outputs:
            loss_out = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
            sft_loss, outputs = cast(tuple[torch.Tensor, Any], loss_out)
        else:
            sft_loss = cast(
                torch.Tensor,
                super().compute_loss(
                    model,
                    inputs,
                    return_outputs=False,
                    num_items_in_batch=num_items_in_batch,
                    **kwargs,
                ),
            )

        if self.kl_beta > 0 and self._ref_model is not None:
            input_ids = inputs["input_ids"]
            token_mask = self._assistant_mask(inputs)

            with torch.no_grad():
                logits = getattr(outputs, "logits", None) if outputs is not None else None
                if logits is not None:
                    train_logits = logits.float()
                else:
                    train_logits = model(
                        input_ids=input_ids,
                        attention_mask=inputs.get("attention_mask"),
                    ).logits.float()

            ref_logits = self._ref_logits(input_ids)
            kl_loss = self._compute_kl(train_logits, ref_logits, token_mask)

            self.log(
                {
                    "train/sft_loss": float(sft_loss.detach().item()),
                    "train/kl_loss": float(kl_loss.detach().item()),
                    "train/kl_beta": self.kl_beta,
                }
            )
            total_loss = sft_loss + self.kl_beta * kl_loss
        else:
            total_loss = sft_loss

        return (total_loss, outputs) if return_outputs else total_loss
