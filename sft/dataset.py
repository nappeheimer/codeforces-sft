from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from .config import TrainConfig

logger = logging.getLogger(__name__)


def _parse_jsonl_line(raw: object, lineno: int) -> list[dict]:
    """Accept either a message list or {\"messages\": [...]}."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "messages" in raw:
        msgs = raw["messages"]
        if not isinstance(msgs, list):
            raise ValueError(f"Line {lineno}: 'messages' must be a list")
        return msgs
    raise ValueError(
        f"Line {lineno}: expected a JSON array or {{\"messages\": [...]}}, got {type(raw)}"
    )


def load_jsonl(
    path: str | Path,
    *,
    label: str | None = None,
    max_chars: int = 0,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    tag = label or path.name
    print(f"[dataset] Loading {tag} from {path} ...", flush=True)
    rows: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            msgs = _parse_jsonl_line(json.loads(line), lineno)
            if max_chars > 0:
                total_chars = sum(len(m.get("content", "")) for m in msgs)
                if total_chars > max_chars:
                    skipped += 1
                    continue
            rows.append({"messages": msgs})

    if skipped:
        print(
            f"[dataset] Skipped {skipped} examples > {max_chars:,} chars "
            f"({skipped/(len(rows)+skipped):.1%} of total) from {tag}",
            flush=True,
        )
    print(f"[dataset] Loaded {len(rows)} rows from {tag}", flush=True)
    logger.info("Loaded %d conversations from %s", len(rows), path)
    return Dataset.from_list(rows)


def build_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset, Dataset | None]:
    """
    Returns (curriculum_dataset, shuffle_dataset, eval_dataset).

    curriculum_data_path: epochs 1–N with fixed order (no shuffle).
    shuffle_data_path:    epochs N+1+ with shuffled sampler.
    eval_data_path:       held-out eval; if set, eval_split_ratio is ignored.
    """
    curriculum_path = cfg.curriculum_data_path or cfg.data_path
    if not curriculum_path:
        raise ValueError(
            "Set curriculum_data_path or data_path in sft_config.yaml"
        )

    shuffle_path = cfg.shuffle_data_path or curriculum_path
    max_chars = cfg.max_example_chars
    curriculum_ds = load_jsonl(curriculum_path, label="curriculum", max_chars=max_chars)
    if Path(shuffle_path).resolve() == Path(curriculum_path).resolve():
        shuffle_ds = curriculum_ds
        print("[dataset] shuffle_data_path == curriculum (same file, shuffle applied later)", flush=True)
    else:
        shuffle_ds = load_jsonl(shuffle_path, label="shuffle", max_chars=max_chars)

    eval_ds: Dataset | None = None
    if cfg.eval_data_path:
        eval_ds = load_jsonl(cfg.eval_data_path, label="eval")
    elif cfg.eval_split_ratio > 0:
        splits = curriculum_ds.train_test_split(
            test_size=cfg.eval_split_ratio, seed=cfg.seed
        )
        curriculum_ds = splits["train"]
        eval_ds = splits["test"]
        print(
            f"[dataset] Held out {len(eval_ds)} eval rows ({cfg.eval_split_ratio:.0%} split)",
            flush=True,
        )

    return curriculum_ds, shuffle_ds, eval_ds
