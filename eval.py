#!/usr/bin/env python3
"""
eval.py — two-part evaluation for the fine-tuned Codeforces SFT model.

Part 1 (--loss): Compute per-example NLL loss on the held-out eval set (same
                 5% split used during training) and report mean loss /
                 perplexity on assistant tokens only.

Part 2 (--generate): Sample N problems from the eval set, generate solutions
                     with the fine-tuned model, then write a JSON report to --out.

Examples are distributed across all visible GPUs; each GPU runs a full model
copy independently (data parallelism, no sharding).

Usage:
    python eval.py --loss
    python eval.py --generate --n 382 --out eval_results.json
    python eval.py --loss --generate --n 50 --out eval_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── helpers ──────────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = Path("./checkpoints")
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _find_adapter(adapter_dir: Optional[str]) -> Path:
    if adapter_dir:
        p = Path(adapter_dir)
        if not p.exists():
            sys.exit(f"[eval] adapter not found: {p}")
        return p
    state_file = CHECKPOINTS_DIR / "checkpoint-150" / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return Path(best)
    if (CHECKPOINTS_DIR / "adapter_model.safetensors").exists():
        return CHECKPOINTS_DIR
    ckpts = sorted(CHECKPOINTS_DIR.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        return ckpts[-1]
    sys.exit("[eval] No adapter found. Pass --adapter <path>.")


def _load_model(adapter_path: Path, device: str):
    print(f"[{device}] Loading base model {BASE_MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        use_cache=True,
    )
    print(f"[{device}] Loading LoRA adapters from {adapter_path} ...", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"[{device}] Model ready.", flush=True)
    return tok, model


def _load_base_model(device: str):
    print(f"[{device}] Loading base model (no LoRA) for comparison ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        use_cache=True,
    )
    model.eval()
    return tok, model


def _load_eval_examples(n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load eval split using the same logic as training (5% held-out)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from sft.config import load_config
    from sft.dataset import build_datasets

    cfg = load_config()
    _, _, eval_ds = build_datasets(cfg)
    if eval_ds is None:
        sys.exit("[eval] No eval split found. Check eval_split_ratio in sft_config.yaml.")

    examples = [eval_ds[i] for i in range(len(eval_ds))]
    print(f"[eval] Loaded {len(examples)} eval examples.", flush=True)

    if n is not None and n < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n)
        print(f"[eval] Sampled {n} for generation.", flush=True)

    return examples


def _split(lst: list, n: int) -> list[list]:
    """Round-robin split into n chunks so lengths differ by at most 1."""
    chunks: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(lst):
        chunks[i % n].append(item)
    return chunks


# ── Part 1: loss / perplexity ─────────────────────────────────────────────────

def _compute_loss_on_dataset(tok, model, examples: list[dict], device: str) -> dict:
    """Compute mean loss on assistant tokens only. Returns aggregatable dict."""
    total_loss = 0.0
    total_tokens = 0
    per_example = []

    print(f"[{device}] Computing loss on {len(examples)} examples ...", flush=True)

    for i, ex in enumerate(examples):
        global_idx = ex.get("_global_idx", i)

        if "messages" not in ex:
            input_ids = torch.tensor(ex["input_ids"]).unsqueeze(0).to(device)
            labels = torch.tensor(ex["labels"]).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, labels=labels)
            loss_val = out.loss.item()
            n_toks = (labels != -100).sum().item()
            total_loss += loss_val * n_toks
            total_tokens += n_toks
            per_example.append({"index": global_idx, "loss": loss_val, "tokens": n_toks})
            if (i + 1) % 50 == 0:
                print(f"  [{device}] [{i+1}/{len(examples)}] mean loss={total_loss/max(total_tokens,1):.4f}", flush=True)
            continue

        msgs = ex["messages"]
        full = tok.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = full["input_ids"].to(device)
        labels = input_ids.clone()

        prompt_ids = tok.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        labels[:, :prompt_ids.shape[1]] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)

        loss_val = out.loss.item()
        n_toks = (labels != -100).sum().item()
        total_loss += loss_val * n_toks
        total_tokens += n_toks
        per_example.append({"index": global_idx, "loss": loss_val, "tokens": n_toks})

        if (i + 1) % 50 == 0:
            print(f"  [{device}] [{i+1}/{len(examples)}] mean loss={total_loss/max(total_tokens,1):.4f}", flush=True)

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "total_tokens": total_tokens,
        "n_examples": len(examples),
        "per_example": per_example,
    }


def _merge_loss_results(rank_files: list[Path]) -> dict:
    total_loss_sum = 0.0
    total_tokens = 0
    n_examples = 0
    per_example = []
    for p in rank_files:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        total_loss_sum += d["mean_loss"] * d["total_tokens"]
        total_tokens += d["total_tokens"]
        n_examples += d["n_examples"]
        per_example.extend(d["per_example"])
    per_example.sort(key=lambda x: x["index"])
    mean_loss = total_loss_sum / max(total_tokens, 1)
    return {
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "total_tokens": total_tokens,
        "n_examples": n_examples,
        "per_example": per_example,
    }


# ── Part 2: generation ────────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    m = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _generate_solution(tok, model, msgs: list[dict], device: str,
                       max_new_tokens: int, temperature: float) -> str:
    prompt_ids = tok.apply_chat_template(
        msgs[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(prompt_ids).to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    generated = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    toks_generated = out.shape[1] - prompt_ids.shape[1]
    print(f"    [{device}] {toks_generated} tokens in {elapsed:.1f}s ({toks_generated/elapsed:.0f} tok/s)", flush=True)
    return generated


def _run_generation(tok, ft_model, base_model, examples: list[dict],
                    device: str, max_new_tokens: int, temperature: float,
                    out_path: Path) -> list[dict]:
    results = []

    for i, ex in enumerate(examples):
        msgs = ex.get("messages", [])
        if not msgs:
            continue

        global_idx = ex.get("_global_idx", i)
        user_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ref_assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        print(f"\n[{device}] Problem {i+1}/{len(examples)} (global #{global_idx}, {len(user_content)} chars)", flush=True)

        ft_response = _generate_solution(tok, ft_model, msgs, device, max_new_tokens, temperature)

        base_response = None
        if base_model is not None:
            base_response = _generate_solution(tok, base_model, msgs, device, max_new_tokens, temperature)

        result = {
            "index": global_idx,
            "problem": user_content,
            "reference": {
                "full": ref_assistant,
                "think": _extract_think(ref_assistant),
                "code": _extract_code(ref_assistant),
            },
            "finetuned": {
                "full": ft_response,
                "think": _extract_think(ft_response),
                "code": _extract_code(ft_response),
            },
        }
        if base_response is not None:
            result["base"] = {
                "full": base_response,
                "think": _extract_think(base_response),
                "code": _extract_code(base_response),
            }

        results.append(result)
        print(f"  [REF code]  {len(result['reference']['code'])} chars")
        print(f"  [FT  code]  {len(result['finetuned']['code'])} chars")

        # Incremental save after every example
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n[{device}] Generation saved to {out_path}", flush=True)
    return results


def _merge_gen_results(rank_files: list[Path], out_path: Path) -> list[dict]:
    results = []
    for p in rank_files:
        if not p.exists():
            continue
        results.extend(json.loads(p.read_text()))
    results.sort(key=lambda r: r["index"])
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def _print_generation_summary(results: list[dict]):
    ft_has_code = sum(1 for r in results if r["finetuned"]["code"])
    ref_has_code = sum(1 for r in results if r["reference"]["code"])
    print(f"\n{'='*60}")
    print(f"Generation summary ({len(results)} problems)")
    print(f"  Reference solutions with code block : {ref_has_code}/{len(results)}")
    print(f"  FT model solutions with code block  : {ft_has_code}/{len(results)}")
    if results and results[0].get("base"):
        base_has_code = sum(1 for r in results if r.get("base", {}).get("code"))
        print(f"  Base model solutions with code block: {base_has_code}/{len(results)}")
    print(f"{'='*60}\n")


# ── worker ────────────────────────────────────────────────────────────────────

def _worker(rank: int, adapter_path_str: str, loss_examples: list[dict],
            gen_examples: list[dict], compare_base: bool,
            max_new_tokens: int, temperature: float, out_prefix: str):
    """
    Runs in a separate process. Loads a full model on cuda:{rank} and processes
    its slice of examples for loss and/or generation.
    """
    device = f"cuda:{rank}"
    adapter_path = Path(adapter_path_str)

    if not loss_examples and not gen_examples:
        return

    tok, ft_model = _load_model(adapter_path, device)

    base_model = None
    if gen_examples and compare_base:
        _, base_model = _load_base_model(device)

    if loss_examples:
        loss_result = _compute_loss_on_dataset(tok, ft_model, loss_examples, device)
        Path(f"{out_prefix}.rank{rank}.loss.json").write_text(
            json.dumps(loss_result, indent=2))
        print(f"[{device}] Loss done — mean={loss_result['mean_loss']:.4f}", flush=True)

    if gen_examples:
        gen_out = Path(f"{out_prefix}.rank{rank}.gen.json")
        _run_generation(tok, ft_model, base_model, gen_examples,
                        device, max_new_tokens, temperature, gen_out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Codeforces SFT model")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--loss", action="store_true", help="Compute loss/perplexity on full eval set")
    parser.add_argument("--generate", action="store_true", help="Generate solutions on sampled eval problems")
    parser.add_argument("--compare-base", action="store_true", help="Also generate with base model (no LoRA)")
    parser.add_argument("--n", type=int, default=20, help="Number of problems to generate (default: 20)")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.loss and not args.generate:
        parser.print_help()
        print("\nError: specify at least one of --loss or --generate")
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        sys.exit("[eval] No GPUs found.")
    print(f"[eval] {n_gpus} GPU(s) visible (cuda:0 .. cuda:{n_gpus-1})", flush=True)

    adapter_path = _find_adapter(args.adapter)
    print(f"[eval] Using adapter: {adapter_path}", flush=True)

    out_prefix = str(Path(args.out).with_suffix(""))

    # Load examples once in the main process, then split
    loss_chunks: list[list] = [[] for _ in range(n_gpus)]
    gen_chunks: list[list] = [[] for _ in range(n_gpus)]

    if args.loss:
        loss_examples = _load_eval_examples()
        for idx, ex in enumerate(loss_examples):
            ex["_global_idx"] = idx
        loss_chunks = _split(loss_examples, n_gpus)
        print(f"[eval] Loss: {len(loss_examples)} examples → ~{len(loss_chunks[0])} per GPU", flush=True)

    if args.generate:
        gen_examples = _load_eval_examples(n=args.n, seed=args.seed)
        for idx, ex in enumerate(gen_examples):
            ex["_global_idx"] = idx
        gen_chunks = _split(gen_examples, n_gpus)
        print(f"[eval] Generate: {len(gen_examples)} examples → ~{len(gen_chunks[0])} per GPU", flush=True)

    # Spawn one process per GPU
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                str(adapter_path),
                loss_chunks[rank],
                gen_chunks[rank],
                args.compare_base,
                args.max_new_tokens,
                args.temperature,
                out_prefix,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"[eval] WARNING: ranks {failed} exited with non-zero code", flush=True)

    # ── Merge loss ──────────────────────────────────────────────────────────
    if args.loss:
        rank_loss_files = [Path(f"{out_prefix}.rank{r}.loss.json") for r in range(n_gpus)]
        merged_loss = _merge_loss_results(rank_loss_files)
        print(f"\n{'='*60}")
        print(f"EVAL LOSS RESULTS ({merged_loss['n_examples']} examples, {merged_loss['total_tokens']} assistant tokens)")
        print(f"  Mean loss  : {merged_loss['mean_loss']:.4f}")
        print(f"  Perplexity : {merged_loss['perplexity']:.2f}")
        print(f"{'='*60}\n")
        loss_out = Path(args.out).with_suffix(".loss.json")
        loss_out.write_text(json.dumps(merged_loss, indent=2))
        print(f"[eval] Loss details saved to {loss_out}", flush=True)

    # ── Merge generation ────────────────────────────────────────────────────
    if args.generate:
        rank_gen_files = [Path(f"{out_prefix}.rank{r}.gen.json") for r in range(n_gpus)]
        results = _merge_gen_results(rank_gen_files, Path(args.out))
        print(f"[eval] Generation results saved to {args.out} ({len(results)} problems)", flush=True)
        _print_generation_summary(results)

        if results:
            r = results[0]
            print("=" * 60)
            print("SAMPLE OUTPUT — Problem 1")
            print("=" * 60)
            print("PROBLEM (truncated to 600):")
            print(r["problem"][:600])
            print("\nFINE-TUNED THINK (truncated to 500):")
            print(r["finetuned"]["think"][:500] or "(none)")
            print("\nFINE-TUNED CODE:")
            code = r["finetuned"]["code"]
            print(code[:1500] if code else "(no code block found)")
            print("\nREFERENCE CODE (first 800 chars):")
            print(r["reference"]["code"][:800])
            print("=" * 60)


if __name__ == "__main__":
    main()
