from __future__ import annotations

import logging
import math
from collections.abc import Sized
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, SequentialSampler
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class CurriculumSampler(Sampler):
    """
    Single-sampler curriculum learning for distributed training.

    - Epochs 0 .. curriculum_epochs-1: iterate dataset in sequential order
      (curriculum_ds is pre-sorted easy→hard, so this gives curriculum order).
    - Epochs curriculum_epochs .. total: randomly shuffle indices each epoch.

    Trainer calls dataloader.set_epoch(epoch) between epochs; Accelerate's
    DataLoaderShard forwards that call to batch_sampler.sampler (us), so
    self.epoch is always current when __iter__ fires.
    """

    def __init__(
        self,
        dataset_size: int,
        curriculum_epochs: int,
        num_replicas: int,
        rank: int,
        seed: int = 42,
    ) -> None:
        self.n = dataset_size
        self.curriculum_epochs = curriculum_epochs
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(self.n / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.epoch < self.curriculum_epochs:
            indices = list(range(self.n))
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.n, generator=g).tolist()
        # Pad so every replica gets the same number of samples.
        indices += indices[: (self.total_size - len(indices))]
        # Subsample for this rank.
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


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
        if any(k.startswith("eval_") for k in logs):
            parts = [f"epoch={state.epoch:.1f}", f"step={state.global_step}"]
            for k, v in sorted(logs.items()):
                if k.startswith("eval_") and k != "eval_runtime" and k != "eval_samples_per_second" and k != "eval_steps_per_second":
                    parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            print(f"[eval]  {' | '.join(parts)}", flush=True)
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
        use_lora: bool = False,
        curriculum_epochs: int = 0,
        **kwargs,
    ):
        self.kl_beta = kl_beta
        self._ref_model: Optional[torch.nn.Module] = None
        self._ref_model_name_or_path = ref_model_name_or_path
        self.shuffle_enabled = shuffle_enabled
        self._use_lora = use_lora
        self._curriculum_epochs = curriculum_epochs

        super().__init__(*args, **kwargs)
        self.add_callback(ProgressPrintCallback())

        if kl_beta > 0:
            self._load_ref_model()

    def set_shuffle_enabled(self, enabled: bool) -> None:
        """Switch sampler mode and force dataloader rebuild on next step."""
        self.shuffle_enabled = enabled
        self._train_dataloader = None

    def set_train_dataset(self, dataset) -> None:
        # If the dataset is raw (not yet tokenized), run it through TRL's preparation
        # pipeline so it has input_ids/labels/assistant_masks before the dataloader builds.
        if "input_ids" not in dataset.column_names:
            dataset = self._prepare_dataset(
                dataset,
                processing_class=self.processing_class,
                args=self.args,
                packing=self.args.packing,
                formatting_func=None,
                dataset_name="train",
            )
        self.train_dataset = dataset
        self._train_dataloader = None

    def _prepare_dataset(self, dataset, processing_class, args, packing, formatting_func, dataset_name, **kwargs):
        from pathlib import Path
        from datasets import Dataset as HFDataset

        cache_dir = Path(args.output_dir) / ".tok_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache key: dataset size + name (simple but effective for fixed data)
        cache_path = cache_dir / f"{dataset_name}_{len(dataset)}.arrow"

        if cache_path.exists():
            if args.local_rank <= 0:
                print(f"[cache] Loading tokenized {dataset_name} dataset from {cache_path}", flush=True)
            return HFDataset.load_from_disk(str(cache_path))

        result = super()._prepare_dataset(dataset, processing_class, args, packing, formatting_func, dataset_name, **kwargs)

        if args.local_rank <= 0:
            result.save_to_disk(str(cache_path))
            print(f"[cache] Saved tokenized {dataset_name} dataset to {cache_path}", flush=True)
        return result

    def _load_ref_model(self) -> None:
        if self._use_lora:
            # With LoRA the frozen base IS the reference model.
            # disable_adapter() in _ref_logits gives base-model output — zero extra VRAM.
            from peft import PeftModel

            if not isinstance(self.model, PeftModel):
                raise RuntimeError(
                    "[KLReg] use_lora=True but model is not a PeftModel. "
                    "Make sure get_peft_model() is called before the trainer is created."
                )
            # Sentinel: non-None signals KL is active; actual inference uses disable_adapter()
            self._ref_model = self.model
            if self.args.local_rank <= 0:
                print(
                    "[KLReg] LoRA mode: ref = frozen base via disable_adapter(). "
                    "Zero extra VRAM.",
                    flush=True,
                )
            return

        # Full fine-tuning path (only viable without 32k context — kept for reference)
        device = (
            torch.device(f"cuda:{self.args.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.args.local_rank <= 0:
            print(
                f"[KLReg] Loading reference model on {device}: {self._ref_model_name_or_path}",
                flush=True,
            )
        self._ref_model = AutoModelForCausalLM.from_pretrained(
            self._ref_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        if self.args.local_rank <= 0:
            print(f"[KLReg] Reference model ready on {device}.", flush=True)

    @torch.no_grad()
    def _ref_logits(self, input_ids: torch.Tensor, ds_model=None) -> torch.Tensor:
        assert self._ref_model is not None
        if self._use_lora:
            # Route through the DeepSpeed engine with adapters disabled.
            # ZeRO-3 allgathers params one layer at a time (~0.5 GB peak per layer),
            # vs GatheredParameters which temporarily gathers all 14 GB at once.
            # This is safe with use_reentrant=True: the strict tensor-metadata check
            # that caused CheckpointError only fires in use_reentrant=False.
            fwd = ds_model if ds_model is not None else self.model
            with self.model.disable_adapter():
                out = fwd(input_ids=input_ids)
            return out.logits.to(dtype=torch.bfloat16, device=input_ids.device)
        out = self._ref_model(input_ids=input_ids.to(self._ref_model.device))
        return out.logits.to(dtype=torch.bfloat16, device=input_ids.device)

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
        chunk_size: int = 256,
    ) -> torch.Tensor:
        # Computing KL over the full [B, T, V] tensor at once creates 5 large intermediates
        # (log_p, log_q, exp, diff, product) — up to ~47 GB peak for T=32k, V=152k in bf16.
        # Chunking along the sequence dimension keeps only [B, chunk, V] alive at a time,
        # capping peak allocation at chunk_size × vocab × 2 bytes ≈ 75 MB per chunk.
        total_kl = train_logits.new_zeros((), dtype=torch.float32)
        total_tokens = token_mask.float().sum().clamp(min=1)
        T = train_logits.size(1)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            t_chunk = train_logits[:, start:end, :].bfloat16()
            r_chunk = ref_logits[:, start:end, :].bfloat16()
            m_chunk = token_mask[:, start:end].float()

            log_p = F.log_softmax(t_chunk, dim=-1)
            log_q = F.log_softmax(r_chunk, dim=-1)
            per_tok = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B, chunk]
            total_kl = total_kl + (per_tok.float() * m_chunk).sum()

            del t_chunk, r_chunk, log_p, log_q, per_tok, m_chunk

        return total_kl / total_tokens

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self._curriculum_epochs > 0:
            # CurriculumSampler handles both phases via set_epoch():
            # sequential for epochs 0..curriculum_epochs-1, then shuffled.
            return CurriculumSampler(
                dataset_size=len(dataset),
                curriculum_epochs=self._curriculum_epochs,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )

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
        # ── Step 1: ref logits FIRST (before gradient checkpointing activates) ──────
        # Critical ordering: computing ref logits AFTER super().compute_loss() causes
        # ZeRO-3 to free parameters mid-step, which corrupts gradient checkpointing's
        # recomputation pass (tensors appear as shape [0] instead of [hidden_dim]).
        # By computing ref logits first, ZeRO-3 is in a clean state when the SFT
        # forward starts, and gradient checkpointing recomputation is unaffected.
        ref_logits = None
        if self.kl_beta > 0 and self._ref_model is not None:
            input_ids = inputs["input_ids"]
            ref_logits = self._ref_logits(input_ids, ds_model=model)

        # ── Step 2: SFT forward + loss (gradient checkpointing activates here) ──────
        loss_out = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )
        sft_loss, outputs = cast(tuple[torch.Tensor, Any], loss_out)

        # ── Step 3: KL term using logits from the SFT forward ────────────────────────
        if ref_logits is not None:
            token_mask = self._assistant_mask(inputs)

            logits = getattr(outputs, "logits", None)
            if logits is not None:
                train_logits = logits
            else:
                train_logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                ).logits

            kl_loss = self._compute_kl(train_logits, ref_logits, token_mask)
            # ref_logits is fully detached (@no_grad) and no longer needed — free 12.5 GB
            # before the backward pass. train_logits shares storage with outputs.logits
            # and is kept alive by the autograd graph, so del here just drops our alias.
            del ref_logits
            del train_logits

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
