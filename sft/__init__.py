from .config import TrainConfig, load_config
from .dataset import build_datasets, load_jsonl
from .validate import run_preflight, validate_config

__all__ = [
    "TrainConfig",
    "load_config",
    "build_datasets",
    "load_jsonl",
    "KLRegSFTTrainer",
    "validate_config",
    "run_preflight",
]


def __getattr__(name: str):
    if name == "KLRegSFTTrainer":
        from .trainer import KLRegSFTTrainer

        return KLRegSFTTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
