# Minimal L2P-Tool Baseline

This directory adds an independent L2P-style tool retrieval baseline without
changing the existing ToolHCL training/evaluation path.

The baseline freezes a text encoder and trains only:

- `prompt_keys`
- `prompt_pool`
- `fusion` MLP
- `classifier`

For each query, the model selects top-N prompts by query-key cosine similarity,
mean-pools the selected prompts, fuses them with the query embedding, and
classifies over sampled tools.

## Data

Preferred input files are the original retrieval files:

- `data/train/raw/retrieval_train.json` or `data/base/raw/retrieval_train.json`
- `data/train/raw/retrieval_eval.json` or `data/base/raw/retrieval_eval.json`
- `data/task1/raw/retrieval_train.json`
- `data/task1/raw/retrieval_eval.json`
- `data/task2/raw/retrieval_train.json`
- `data/task2/raw/retrieval_eval.json`
- `data/task3/raw/retrieval_train.json`
- `data/task3/raw/retrieval_eval.json`

If a file is absent, the loader tries existing ToolHCL-compatible fallback files
such as `data/train/cache/training_samples_cache.pt` and
`data/task*/clusters/task*_retrieval_clean.json`. It still converts every sample
to:

```json
{
  "query_text": "...",
  "target_tool_id": 0,
  "target_tool_name": "...",
  "task_id": "base"
}
```

Only `query_text` and `target_tool_id` are used for training.

## Mini-Subset Split

The first run creates and saves:

- sampled split: `tool_l2p_runs/splits/mini_seed42.json`
- mapping: `tool_l2p_runs/splits/tool_mapping_seed42.json`

Later train/eval commands load the same files and do not resample tools.
Classifier output size is the total number of sampled tools across
`base/task1/task2/task3`, with consecutive internal ids.

## Full-Data L2P-Tool-NoReplay

The default experiment mode is full data, ToolHCL-compatible frozen encoder,
and no replay/distillation. Incremental stages train only on the current task's
training queries, while the loss and global evaluation are masked over all
visible tools for the checkpoint stage.

This run writes checkpoints to `checkpoints/tool_l2p_full/` and metrics to
`tool_l2p_runs_full/metrics.json`.

```bash
CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/llama-factory/bin/python -m tool_l2p.train \
  --stage base \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --seed 42 \
  --split_path tool_l2p_runs_full/splits/full_seed42.json \
  --output_dir checkpoints/tool_l2p_full \
  --metrics_path tool_l2p_runs_full/metrics.json

CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/llama-factory/bin/python -m tool_l2p.train \
  --stage task1 \
  --ckpt checkpoints/tool_l2p_full/base.pt \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --seed 42 \
  --split_path tool_l2p_runs_full/splits/full_seed42.json \
  --output_dir checkpoints/tool_l2p_full \
  --metrics_path tool_l2p_runs_full/metrics.json

CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/llama-factory/bin/python -m tool_l2p.train \
  --stage task2 \
  --ckpt checkpoints/tool_l2p_full/task1.pt \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --seed 42 \
  --split_path tool_l2p_runs_full/splits/full_seed42.json \
  --output_dir checkpoints/tool_l2p_full \
  --metrics_path tool_l2p_runs_full/metrics.json

CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/llama-factory/bin/python -m tool_l2p.train \
  --stage task3 \
  --ckpt checkpoints/tool_l2p_full/task2.pt \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --seed 42 \
  --split_path tool_l2p_runs_full/splits/full_seed42.json \
  --output_dir checkpoints/tool_l2p_full \
  --metrics_path tool_l2p_runs_full/metrics.json
```

Summarize the continual-learning results:

```bash
CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/llama-factory/bin/python -m tool_l2p.summarize \
  --metrics_path tool_l2p_runs_full/metrics.json
```

## Full-Data L2P-Tool-Regularized-NoReplay

`regularized_no_replay` still trains only on the current task at each
incremental stage. It does not use replay buffers, old-task rehearsal, or
teacher-student distillation. It adds only general regularization around the
L2P prompt-pool baseline:

- `sampled_ce` or `local_ce` to avoid treating every old tool as a strong
  negative on every current-task batch.
- optional old classifier row freezing.
- optional fusion freezing after base.
- optional prompt usage balance loss.
- optional train-time top-m prompt sampling.

The wrapper below runs `base -> task1 -> task2 -> task3` and writes all outputs
to `tool_l2p_runs_regularized_no_replay/`:

```bash
PY=/opt/miniconda3/envs/llama-factory/bin/python

CUDA_VISIBLE_DEVICES=2 $PY train_tool_l2p.py \
  --full_data \
  --encoder_backend toolhcl \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --l2p_variant regularized_no_replay \
  --loss_type sampled_ce \
  --num_current_negatives 256 \
  --num_old_negatives 64 \
  --freeze_old_classifier_rows \
  --freeze_fusion_after_base \
  --prompt_balance_loss_weight 0.01 \
  --prompt_top_k 5 \
  --prompt_top_m 10 \
  --prompt_sampling_train \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --seed 42 \
  --output_dir tool_l2p_runs_regularized_no_replay
```

Outputs:

- `tool_l2p_runs_regularized_no_replay/metrics.json`
- `tool_l2p_runs_regularized_no_replay/train.log`
- `tool_l2p_runs_regularized_no_replay/config.json`
- `tool_l2p_runs_regularized_no_replay/prompt_usage.json`
- `tool_l2p_runs_regularized_no_replay/classifier_stats.json`
- `tool_l2p_runs_regularized_no_replay/checkpoints/{base,task1,task2,task3}.pt`

Summarize:

```bash
CUDA_VISIBLE_DEVICES=2 $PY -m tool_l2p.summarize \
  --metrics_path tool_l2p_runs_regularized_no_replay/metrics.json
```

## Full-Data L2P-Tool-Replay

`replay` enables a standard continual-learning replay buffer. It still does not
use ToolHCL hierarchy, box geometry, L1/L2 routers, capability space, or
distillation. The recommended main setting uses global CE because old tools now
appear again as positive labels through replay samples.

```bash
PY=/opt/miniconda3/envs/llama-factory/bin/python

CUDA_VISIBLE_DEVICES=2 $PY train_tool_l2p.py \
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
  --output_dir tool_l2p_runs_replay \
  --save_replay_buffer
```

Light version:

```bash
CUDA_VISIBLE_DEVICES=2 $PY train_tool_l2p.py \
  --full_data \
  --encoder_backend toolhcl_llm \
  --encoder_model auto \
  --llm_pooling embedding_only \
  --device cuda \
  --l2p_variant replay \
  --loss_type global_ce \
  --replay_strategy balanced_per_task \
  --replay_samples_per_task 1000 \
  --replay_ratio 0.20 \
  --batch_size 64 \
  --epochs 1 \
  --prompt_top_k 5 \
  --learning_rate 5e-4 \
  --seed 42 \
  --output_dir tool_l2p_runs_replay_light \
  --save_replay_buffer
```

Replay outputs include `replay_stats.json` and
`replay_buffer_after_{base,task1,task2,task3}.json`.

Compare variants:

```bash
$PY -m tool_l2p.compare \
  --no_replay_metrics tool_l2p_runs_full/metrics.json \
  --regularized_metrics tool_l2p_runs_regularized_no_replay/metrics.json \
  --replay_metrics tool_l2p_runs_replay/metrics.json
```

## Mini-Subset Train Base -> Task3

```bash
python -m tool_l2p.train \
  --stage base \
  --mini_subset \
  --max_base_tools 500 \
  --max_train_samples_per_tool 5 \
  --seed 42 \
  --split_path tool_l2p_runs/splits/mini_seed42.json

python -m tool_l2p.train \
  --stage task1 \
  --ckpt checkpoints/tool_l2p/base.pt \
  --mini_subset \
  --max_base_tools 500 \
  --max_incremental_tools 50 \
  --max_train_samples_per_tool 5 \
  --seed 42 \
  --split_path tool_l2p_runs/splits/mini_seed42.json

python -m tool_l2p.train \
  --stage task2 \
  --ckpt checkpoints/tool_l2p/task1.pt \
  --mini_subset \
  --max_base_tools 500 \
  --max_incremental_tools 50 \
  --max_train_samples_per_tool 5 \
  --seed 42 \
  --split_path tool_l2p_runs/splits/mini_seed42.json

python -m tool_l2p.train \
  --stage task3 \
  --ckpt checkpoints/tool_l2p/task2.pt \
  --mini_subset \
  --max_base_tools 500 \
  --max_incremental_tools 50 \
  --max_train_samples_per_tool 5 \
  --seed 42 \
  --split_path tool_l2p_runs/splits/mini_seed42.json
```

Default checkpoints are written to `checkpoints/tool_l2p/{stage}.pt`.
Metrics and diagnostics are appended to `tool_l2p_runs/metrics.json`.

## Evaluation

Global evaluation uses all visible tools for the checkpoint stage.
Local evaluation only uses the evaluated task's sampled tools and is diagnostic.

```bash
python -m tool_l2p.eval \
  --ckpt checkpoints/tool_l2p/task1.pt \
  --eval_task task1 \
  --mode global \
  --mini_subset \
  --split_path tool_l2p_runs/splits/mini_seed42.json

python -m tool_l2p.eval \
  --ckpt checkpoints/tool_l2p/task1.pt \
  --eval_task task1 \
  --mode local \
  --mini_subset \
  --split_path tool_l2p_runs/splits/mini_seed42.json
```

Reported metrics:

- Recall@1/3/5
- NDCG@1/3/5
- MRR

Training also prints:

- prompt selection histogram
- old/new classifier weight and bias diagnostics
- current-task local/global eval plus previous visible-task global eval

## Encoder Backend

`--encoder_backend auto` tries `sentence_transformers` with
`all-MiniLM-L6-v2`. If unavailable, it falls back to a frozen deterministic
hashing encoder so the mini baseline remains runnable. To force the fallback:

```bash
python -m tool_l2p.train --stage base --mini_subset --encoder_backend hashing
```
