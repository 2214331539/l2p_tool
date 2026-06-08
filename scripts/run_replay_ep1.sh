#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
GPU="${GPU:-2}"
OUT="${OUT:-tool_l2p_runs_replay}"

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" train_tool_l2p.py \
  --output_dir "$OUT" \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --l2p_variant replay \
  --loss_type global_ce \
  --replay_strategy balanced_per_task \
  --replay_samples_per_task 2000 \
  --replay_ratio 0.25 \
  --batch_size 64 \
  --epochs 1 \
  --prompt_top_k 5 \
  --learning_rate 5e-4 \
  --seed 42 \
  --save_replay_buffer \
  2>&1 | tee "$OUT/tmux_run.log"

"$PY" -m tool_l2p.summarize --metrics_path "$OUT/metrics.json" | tee -a "$OUT/tmux_run.log"

