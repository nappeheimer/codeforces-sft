Training JSONL files (included in this repo via Git LFS):

  train_curriculum.jsonl  — 8167 rows, curriculum order (epochs 1–2, no shuffle)
  train.jsonl             — 8167 rows, parquet order (epochs 3–10, shuffled)
  test.jsonl              — 894 rows, for post-training benchmarking only

Each line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

After cloning:  git lfs pull
