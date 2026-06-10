import os

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Disables Rust deadlock so we can use 32-core dataset filtering
)
import sys
import logging
import argparse

from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    TrainerCallback,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Failsafe: Instantly crash on GPU deadlocks to prevent massive idle bills
# NCCL_BLOCKING_WAIT removed to allow DeepSpeed ZeRO-2 to correctly overlap comms with compute!
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


class CurriculumSampler(Sampler):
    """
    A Sampler that respects the order of the curriculum phase
    and switches to shuffling for the regular phase.
    Relies on HuggingFace Accelerate to shard the batches across GPUs.
    """

    def __init__(
        self,
        dataset,
        shuffle_after_epoch=0,
        seed=42,
    ):
        self.dataset = dataset
        self.shuffle_after_epoch = shuffle_after_epoch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if not hasattr(self, "sorted_indices"):
            # Exact token length: use input_ids because SFTTrainer removes text columns
            lengths = [len(item["input_ids"]) for item in self.dataset]
            self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        if self.epoch < self.shuffle_after_epoch:
            # Phase 1: Strictly ordered by length (shortest to longest warmup)
            indices = self.sorted_indices.copy()
        else:
            # Phase 2: Bucket Shuffled (Length-Grouped)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # 256 = Global batch size (8 GPUs * 8 per_device * 4 grad_accum)
            # Groups sequences into buckets of 256 so every global step has uniform lengths
            bucket_size = 256
            buckets = [self.sorted_indices[i:i + bucket_size] for i in range(0, len(self.sorted_indices), bucket_size)]
            
            # Shuffle the order of the buckets
            bucket_order = torch.randperm(len(buckets), generator=g).tolist()
            
            indices = []
            for b_idx in bucket_order:
                bucket = buckets[b_idx]
                # Shuffle the sequences within the bucket itself
                intra_bucket_order = torch.randperm(len(bucket), generator=g).tolist()
                indices.extend([bucket[i] for i in intra_bucket_order])

        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch


class CurriculumCallback(TrainerCallback):
    """
    Ensures the CurriculumSampler advances its epoch, since Trainer only
    calls set_epoch automatically for DistributedSampler subclasses.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.trainer, "curriculum_sampler"):
            self.trainer.curriculum_sampler.set_epoch(round(state.epoch or 0))


class OlympicCoderTrainer(SFTTrainer):
    """
    Overridden SFTTrainer to inject our Curriculum Sampler.
    """

    def __init__(self, *args, curriculum_epochs=0, **kwargs):
        # Filter out FINCH kwargs if they are accidentally passed
        kwargs.pop("finch_base_lr", None)
        kwargs.pop("finch_eta_max", None)
        kwargs.pop("finch_ema_alpha", None)
        kwargs.pop("kl_beta", None)
        kwargs.pop("ref_model", None)
        self.curriculum_epochs = curriculum_epochs
        self.curriculum_sampler = None
        super().__init__(*args, **kwargs)

    def _get_train_sampler(
        self, train_dataset=None
    ) -> Optional[torch.utils.data.Sampler]:
        target_dataset = (
            train_dataset if train_dataset is not None else self.train_dataset
        )
        if target_dataset is None:
            return None
        self.curriculum_sampler = CurriculumSampler(
            target_dataset,
            shuffle_after_epoch=self.curriculum_epochs,
            seed=self.args.seed,
        )
        return self.curriculum_sampler


class EpochSnapshotCallback(TrainerCallback):
    """
    Saves a permanent snapshot of the model weights at the end of each epoch.
    Named 'epoch_X' so the Trainer's rolling deletion ignores it.
    """

    def __init__(self, trainer, stop_after_epoch=None):
        self.trainer = trainer
        self.stop_after_epoch = stop_after_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch or 0)
        # Use a custom prefix to prevent HF from auto-deleting the snapshot
        output_dir = f"/opt/ml/checkpoints/epoch_{epoch}_snapshot"
        logger.info(f"*** Saving permanent epoch snapshot to {output_dir} ***")

        if dist.is_initialized():
            dist.barrier()

        self.trainer.save_model(output_dir)

        if self.stop_after_epoch is not None and epoch >= self.stop_after_epoch:
            logger.info(
                f"Reached stop_after_epoch={self.stop_after_epoch}. Halting training gracefully!"
            )
            if dist.is_initialized():
                dist.barrier()
            control.should_training_stop = True


class VRAMLoggingCallback(TrainerCallback):
    """
    Prints the exact peak VRAM usage in Gigabytes every 5 steps.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_reserved() / (1024**3)
                # Only log on rank 0 to prevent 8x log spam, and reset the peak stats
                # to measure the *current* 5 steps instead of the all-time watermark.
                if state.is_world_process_zero:
                    logger.info(
                        f"*** Peak VRAM Reserved (Last 5 steps): {max_mem:.2f} GB / 141.0 GB ***"
                    )
                torch.cuda.reset_peak_memory_stats()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Olympic Coder SFT Training Script (Final Stability Version)"
    )

    # SageMaker native environment variables
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )

    # Curriculum Configuration
    parser.add_argument("--curriculum_file", type=str, default="train_curriculum.jsonl")
    parser.add_argument("--curriculum_epochs", type=int, default=2)
    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--stop_after_epoch", type=int, default=None)

    # Model & Hyperparams
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Optimization
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument(
        "--dry_run", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Manual Liger Kernel application (Qwen2.5 is compatible with Qwen2 kernels)
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        apply_liger_kernel_to_qwen2()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("Successfully applied Liger Kernels to Qwen model.")
    except ImportError:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.warning("Liger-kernel not found, skipping optimization.")

    logger.info("*** Starting STABLE SFT Training ***")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    # Do NOT use eos_token as pad_token, as it masks true EOS tokens in the loss.
    # Qwen has a dedicated <|endoftext|> token (id 151643) that is safe for padding.
    tokenizer.pad_token = "<|endoftext|>"
    # Flash Attention 2 requires right padding for causal training to prevent positional corruption
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    # 2. Load and Merge Dataset
    def load_jsonl(filename, limit=None):
        path = Path(args.train_dir) / filename
        if not path.exists():
            matches = list(Path(args.train_dir).rglob(filename))
            if not matches:
                raise FileNotFoundError(f"Missing {filename}")
            path = matches[0]

        ds = load_dataset("json", data_files=str(path), split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    limit = 5000 if args.dry_run else None
    logger.info(f"Loading curriculum data: {args.curriculum_file}")

    # Use only the curriculum dataset. The custom sampler will handle
    # iterating over it sequentially first, then randomly shuffling it later.
    full_dataset = load_jsonl(args.curriculum_file, limit)
    logger.info(f"Full dataset size before filter: {len(full_dataset)}")

    # 2.5 Filter out sequences that exceed max_length to avoid truncation corruption
    logger.info(
        f"Filtering dataset to strictly skip traces exceeding {args.max_length} tokens..."
    )

    def filter_by_length(example):
        try:
            # SFTTrainer will apply chat template, so we measure the exact token count
            # Using tokenize=True directly avoids double-tokenization
            ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
            # SFTTrainer re-tokenizes from a stringified version during formatting,
            # which can add 1-3 whitespace tokens. Using a safety buffer prevents
            # SFTTrainer from silently hard-truncating the sequence and chopping off the EOS token.
            return len(ids) <= (args.max_length - 100)
        except Exception as e:
            logger.warning(f"Skipping sample due to formatting error: {e}")
            return False

    # Safely parallelize tokenization across 32 cores for massive speedup
    full_dataset = full_dataset.filter(filter_by_length, num_proc=32)
    logger.info(f"Filtered dataset size: {len(full_dataset)}")

    # 3. Setup Standard TrainingArguments (More stable than SFTConfig)
    training_args = TrainingArguments(
        output_dir="/opt/ml/checkpoints",
        num_train_epochs=args.total_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="no",  # Transformers >= 4.41 deprecates evaluation_strategy
        save_strategy="epoch",  # User explicitly requested epoch-level saves to maximize I/O efficiency
        save_total_limit=2,
        deepspeed=args.deepspeed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        ddp_timeout=1800,  # Failsafe: Crash after 30 mins of deadlock instead of 4 hours
    )

    # 4. Setup Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Using FA2 for 32k context
        trust_remote_code=True,
        use_cache=False,
    )

    # 4.5 Setup DataCollator to mask user prompts from loss calculation
    # Mask both user and system prompts across ALL turns in a conversation.
    # Passing only response_template can cause TRL to leak user prompts into the loss.
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Initialize Trainer with parameters passed directly (Maximum Stability)
    trainer = OlympicCoderTrainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        curriculum_epochs=args.curriculum_epochs,
        max_seq_length=args.max_length,
        dataset_text_field=None,  # Auto-detects conversational 'messages' format
        packing=False,  # Preserves reasoning chains
    )

    # Inject the custom callbacks
    trainer.add_callback(CurriculumCallback(trainer))
    trainer.add_callback(
        EpochSnapshotCallback(trainer, stop_after_epoch=args.stop_after_epoch)
    )
    trainer.add_callback(VRAMLoggingCallback())

    # 6. Train
    logger.info("Starting training loop...")

    last_checkpoint = get_last_checkpoint("/opt/ml/checkpoints")
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # 7. Save Final Model
    logger.info(f"Saving final model to {args.model_dir}...")
    if dist.is_initialized():
        dist.barrier()

    trainer.save_model(args.model_dir)
    # Note: Explicit tokenizer.save_pretrained() is removed because trainer.save_model
    # already saves the processing_class safely on Rank 0. Calling it explicitly here
    # would cause all 8 GPUs to race-write to the exact same file.

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
