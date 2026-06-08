import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import (
    ToolL2PDataset,
    collate_l2p_batch,
    create_or_load_sampled_split,
    load_stage_tool_keys,
    load_raw_samples,
    materialize_samples,
)
from .eval import run_l2p_evaluation
from .metrics import format_metrics
from .model import (
    ToolL2PModel,
    build_model,
    checkpoint_config,
    l2p_regularized_loss,
    load_checkpoint_model,
    save_checkpoint,
)
from .utils import (
    STAGES,
    append_metrics,
    check_stage,
    current_task_tool_ids,
    ensure_dir,
    previous_visible_tool_ids,
    print_json,
    resolve_device,
    set_seed,
    stage_index,
    visible_tool_ids,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal L2P-Tool baseline.")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--ckpt", default=None, help="Previous-stage checkpoint for incremental training.")
    parser.add_argument("--mini_subset", action="store_true", help="Use sampled mini-subset mode.")
    parser.add_argument("--full_data", action="store_true", help="Use all sampled/full task tools and all queries.")
    parser.add_argument("--max_base_tools", type=int, default=500)
    parser.add_argument("--max_incremental_tools", type=int, default=50)
    parser.add_argument("--max_train_samples_per_tool", type=int, default=5)
    parser.add_argument("--max_eval_samples_per_tool", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--mapping_path", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--save_path", default=None)

    parser.add_argument(
        "--encoder_backend",
        default="toolhcl_llm",
        choices=["auto", "sentence_transformer", "hashing", "toolhcl", "toolhcl_st", "toolhcl_llm"],
    )
    parser.add_argument("--encoder_model", default="auto")
    parser.add_argument("--encoder_model_path", default=None)
    parser.add_argument("--llm_pooling", default="embedding_only", choices=["embedding_only", "transformer_layer"])
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--prompt_pool_size", type=int, default=20)
    parser.add_argument("--prompt_length", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--top_k_prompt", type=int, default=None)
    parser.add_argument("--prompt_top_k", type=int, default=None)

    parser.add_argument("--l2p_variant", choices=["no_replay", "regularized_no_replay", "replay"], default="no_replay")
    parser.add_argument("--loss_type", choices=["global_ce", "local_ce", "sampled_ce"], default="global_ce")
    parser.add_argument("--freeze_old_classifier_rows", action="store_true")
    parser.add_argument("--freeze_fusion_after_base", action="store_true")
    parser.add_argument("--prompt_balance_loss_weight", type=float, default=0.0)
    parser.add_argument("--prompt_top_m", type=int, default=None)
    parser.add_argument("--prompt_sampling_train", action="store_true")
    parser.add_argument("--num_current_negatives", type=int, default=256)
    parser.add_argument("--num_old_negatives", type=int, default=64)
    parser.add_argument("--replay_strategy", choices=["none", "reservoir", "balanced_per_task", "balanced_per_tool"], default="none")
    parser.add_argument("--replay_buffer_size", type=int, default=0)
    parser.add_argument("--replay_samples_per_task", type=int, default=2000)
    parser.add_argument("--replay_samples_per_tool", type=int, default=1)
    parser.add_argument("--replay_ratio", type=float, default=0.0)
    parser.add_argument("--replay_batch_size", type=int, default=None)
    parser.add_argument("--replay_after_each_task", action="store_true")
    parser.add_argument("--save_replay_buffer", action="store_true")
    parser.add_argument("--load_replay_buffer", default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_key", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mini_subset and args.full_data:
        raise ValueError("Choose only one of --mini_subset or --full_data.")
    if not args.mini_subset:
        args.full_data = True
    if args.encoder_model_path:
        args.encoder_model = args.encoder_model_path
    if args.encoder_backend == "toolhcl":
        args.encoder_backend = "toolhcl_llm"
    if args.top_k_prompt is not None:
        args.top_n = args.top_k_prompt
    if args.prompt_top_k is not None:
        args.top_n = args.prompt_top_k
    if args.learning_rate is not None:
        args.lr = args.learning_rate
    if args.l2p_variant == "replay":
        if args.replay_strategy == "none":
            args.replay_strategy = "balanced_per_task"
        if args.replay_ratio <= 0 and args.replay_batch_size is None:
            args.replay_ratio = 0.25
    if args.full_data:
        args.max_base_tools = 0
        args.max_incremental_tools = 0
        args.max_train_samples_per_tool = 0
        args.max_eval_samples_per_tool = 0
    if args.split_path is None:
        if args.l2p_variant == "regularized_no_replay" and args.full_data:
            full_split_dir = "tool_l2p_runs_regularized_no_replay/splits"
        elif args.l2p_variant == "replay" and args.full_data:
            full_split_dir = "tool_l2p_runs_replay/splits"
        else:
            full_split_dir = "tool_l2p_runs_full/splits"
        args.split_path = (
            f"tool_l2p_runs/splits/mini_seed{args.seed}.json"
            if args.mini_subset
            else f"{full_split_dir}/full_seed{args.seed}.json"
        )
    if args.output_dir is None:
        if args.mini_subset:
            args.output_dir = "checkpoints/tool_l2p"
        elif args.l2p_variant == "regularized_no_replay":
            args.output_dir = "tool_l2p_runs_regularized_no_replay/checkpoints"
        elif args.l2p_variant == "replay":
            args.output_dir = "tool_l2p_runs_replay/checkpoints"
        else:
            args.output_dir = "checkpoints/tool_l2p_full"
    if args.metrics_path is None:
        if args.mini_subset:
            args.metrics_path = "tool_l2p_runs/metrics.json"
        elif args.l2p_variant == "regularized_no_replay":
            args.metrics_path = "tool_l2p_runs_regularized_no_replay/metrics.json"
        elif args.l2p_variant == "replay":
            args.metrics_path = "tool_l2p_runs_replay/metrics.json"
        else:
            args.metrics_path = "tool_l2p_runs_full/metrics.json"
    if args.run_dir is None:
        args.run_dir = os.path.dirname(args.metrics_path) or "."
    if args.prompt_top_m is not None:
        args.prompt_top_m = max(args.top_n, args.prompt_top_m)
    return args


def _load_or_create_split(args: argparse.Namespace):
    if os.path.exists(args.split_path):
        return create_or_load_sampled_split(
            train_samples_by_stage=None,
            split_path=args.split_path,
            mapping_path=args.mapping_path,
            seed=args.seed,
            max_base_tools=args.max_base_tools,
            max_incremental_tools=args.max_incremental_tools,
        )

    raw_by_stage = {
        stage: load_raw_samples(stage, "train", data_root=args.data_root)
        for stage in STAGES
    }
    if args.full_data:
        for stage in STAGES:
            observed = {str(sample["target_tool_key"]) for sample in raw_by_stage.get(stage, [])}
            tool_keys = load_stage_tool_keys(stage, data_root=args.data_root)
            missing = [key for key in tool_keys if key not in observed]
            raw_by_stage[stage].extend(
                {
                    "query_text": "",
                    "target_tool_name": key,
                    "target_tool_key": key,
                    "source_target_id": None,
                    "task_id": stage,
                    "_tool_only": True,
                }
                for key in missing
            )
            print(
                f"[tools] {stage}: dictionary_tools={len(tool_keys)} "
                f"observed_in_train={len(observed)} added_tool_only={len(missing)}"
            )
    return create_or_load_sampled_split(
        train_samples_by_stage=raw_by_stage,
        split_path=args.split_path,
        mapping_path=args.mapping_path,
        seed=args.seed,
        max_base_tools=args.max_base_tools,
        max_incremental_tools=args.max_incremental_tools,
    )


def _build_train_samples(args: argparse.Namespace, mapping: Dict[str, object]) -> List[Dict[str, object]]:
    raw = load_raw_samples(args.stage, "train", data_root=args.data_root)
    task_ids = current_task_tool_ids(args.stage, mapping["task_to_tool_ids"])
    return materialize_samples(
        raw,
        mapping["tool_to_id"],
        sampled_tool_ids=task_ids,
        max_samples_per_tool=args.max_train_samples_per_tool,
        seed=args.seed + stage_index(args.stage) * 4099,
    )


def _replay_enabled(args: argparse.Namespace) -> bool:
    return args.l2p_variant == "replay" and args.replay_strategy != "none"


def _replay_buffer_name(stage: str) -> str:
    return f"replay_buffer_after_{stage}.json"


def _replay_buffer_path(args: argparse.Namespace, stage: str) -> str:
    return os.path.join(args.run_dir, _replay_buffer_name(stage))


def _previous_stage_name(stage: str) -> str | None:
    idx = stage_index(stage)
    if idx == 0:
        return None
    return STAGES[idx - 1]


def _load_replay_buffer(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not _replay_enabled(args):
        return []
    path = args.load_replay_buffer
    if path is None:
        prev = _previous_stage_name(args.stage)
        if prev is None:
            return []
        path = _replay_buffer_path(args, prev)
    if not path or not os.path.exists(path):
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Replay buffer must be a list: {path}")
    out: List[Dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = item.get("query_text")
        tool_id = item.get("tool_id", item.get("target_tool_id"))
        if not isinstance(query, str) or tool_id is None:
            continue
        out.append(
            {
                "query_text": query,
                "target_tool_id": int(tool_id),
                "tool_id": int(tool_id),
                "target_tool_name": item.get("target_tool_name"),
                "task_id": str(item.get("task_id", "unknown")),
                "l1_id": item.get("l1_id"),
                "l2_id": item.get("l2_id"),
                "is_replay": True,
            }
        )
    print(f"[replay] loaded {len(out)} samples from {path}")
    return out


def _to_replay_record(sample: Dict[str, object]) -> Dict[str, object]:
    tool_id = int(sample["target_tool_id"])
    return {
        "query_text": str(sample["query_text"]),
        "tool_id": tool_id,
        "target_tool_id": tool_id,
        "target_tool_name": sample.get("target_tool_name"),
        "task_id": str(sample.get("task_id", "unknown")),
        "l1_id": sample.get("l1_id"),
        "l2_id": sample.get("l2_id"),
    }


def _stable_sample_records(records: List[Dict[str, object]], k: int, seed: int) -> List[Dict[str, object]]:
    if k <= 0 or len(records) <= k:
        return list(records)
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(records)), k))
    return [records[i] for i in idxs]


def _sample_current_for_replay(args: argparse.Namespace, train_samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    records = [_to_replay_record(sample) for sample in train_samples]
    seed = args.seed + stage_index(args.stage) * 7919
    if args.replay_strategy == "balanced_per_task":
        return _stable_sample_records(records, int(args.replay_samples_per_task), seed)
    if args.replay_strategy == "balanced_per_tool":
        grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for record in records:
            grouped[int(record["tool_id"])].append(record)
        out: List[Dict[str, object]] = []
        for offset, tool_id in enumerate(sorted(grouped)):
            out.extend(
                _stable_sample_records(
                    sorted(grouped[tool_id], key=lambda x: str(x["query_text"])),
                    int(args.replay_samples_per_tool),
                    seed + tool_id * 17 + offset,
                )
            )
        return out
    if args.replay_strategy == "reservoir":
        # Reservoir is applied after merging old + current; return all current candidates here.
        return records
    return []


def _merge_replay_buffer(
    args: argparse.Namespace,
    old_buffer: List[Dict[str, object]],
    train_samples: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not _replay_enabled(args):
        return []
    current = _sample_current_for_replay(args, train_samples)
    merged = list(old_buffer) + current
    if args.replay_strategy == "reservoir" and args.replay_buffer_size > 0:
        return _stable_sample_records(merged, int(args.replay_buffer_size), args.seed + 104729 + stage_index(args.stage))
    if args.replay_buffer_size > 0 and len(merged) > args.replay_buffer_size:
        return _stable_sample_records(merged, int(args.replay_buffer_size), args.seed + 104729 + stage_index(args.stage))
    return merged


def _replay_stats(
    buffer: List[Dict[str, object]],
    *,
    visible_ids: Sequence[int],
) -> Dict[str, object]:
    task_counts = Counter(str(item.get("task_id", "unknown")) for item in buffer)
    tool_counts = Counter(str(int(item.get("tool_id", item.get("target_tool_id", -1)))) for item in buffer)
    unique_tools = len(tool_counts)
    visible_count = len(set(int(x) for x in visible_ids))
    return {
        "buffer_size": len(buffer),
        "samples_per_task": dict(sorted(task_counts.items())),
        "samples_per_tool": dict(sorted(tool_counts.items(), key=lambda kv: int(kv[0]))),
        "unique_tools_covered": unique_tools,
        "visible_tools": visible_count,
        "tool_coverage_ratio": (float(unique_tools) / float(visible_count)) if visible_count else 0.0,
        "replay_strategy": "none" if not buffer else "active",
    }


def _write_replay_artifacts(
    args: argparse.Namespace,
    *,
    buffer: List[Dict[str, object]],
    stage: str,
    stats: Dict[str, object],
) -> None:
    if not _replay_enabled(args):
        return
    if args.save_replay_buffer:
        path = _replay_buffer_path(args, stage)
        save_json(buffer, path)
        print(f"[replay] saved buffer {path} samples={len(buffer)}")
    stats_path = os.path.join(args.run_dir, "replay_stats.json")
    all_stats = _load_json_if_exists(stats_path)
    all_stats[stage] = stats
    save_json(all_stats, stats_path)


def _build_or_load_model(args: argparse.Namespace, mapping: Dict[str, object], device: torch.device) -> ToolL2PModel:
    num_total_tools = int(mapping.get("num_total_tools") or len(mapping["tool_to_id"]))
    if args.ckpt:
        model, ckpt_config, _ = load_checkpoint_model(
            args.ckpt,
            device=device,
            encoder_backend_override=args.encoder_backend,
        )
        if model.num_tools != num_total_tools:
            raise ValueError(
                f"Checkpoint classifier size ({model.num_tools}) does not match sampled mapping "
                f"num_total_tools ({num_total_tools}). Use the same split_path/mapping_path."
            )
        print(f"[train] loaded checkpoint {args.ckpt} from stage={ckpt_config.get('stage')}")
        if model.top_n != args.top_n:
            print(f"[train] overriding checkpoint top_n={model.top_n} with requested top_n={args.top_n}")
            model.top_n = int(args.top_n)
    else:
        if args.stage != "base":
            print(f"[train] warning: no --ckpt supplied for {args.stage}; initializing from scratch.")
        model = build_model(
            encoder_backend=args.encoder_backend,
            encoder_model=args.encoder_model,
            hidden_dim=args.hidden_dim,
            num_tools=num_total_tools,
            prompt_pool_size=args.prompt_pool_size,
            prompt_length=args.prompt_length,
            top_n=args.top_n,
            device=device,
            llm_pooling=args.llm_pooling,
        )

    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("encoder.")
    return model


def _apply_freeze_controls(
    model: ToolL2PModel,
    *,
    stage: str,
    freeze_fusion_after_base: bool,
) -> None:
    if freeze_fusion_after_base and stage != "base":
        for param in model.fusion.parameters():
            param.requires_grad = False


def _parameter_trainability(model: ToolL2PModel) -> Dict[str, object]:
    trainable: List[str] = []
    frozen: List[str] = []
    trainable_params = 0
    frozen_params = 0
    encoder_trainable_params = 0
    encoder_frozen_params = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        if name.startswith("encoder."):
            if param.requires_grad:
                encoder_trainable_params += count
                trainable_params += count
            else:
                encoder_frozen_params += count
                frozen_params += count
            continue
        if param.requires_grad:
            trainable.append(name)
            trainable_params += count
        else:
            frozen.append(name)
            frozen_params += count
    return {
        "trainable_names": trainable,
        "frozen_names": frozen,
        "encoder_trainable_parameter_count": encoder_trainable_params,
        "encoder_frozen_parameter_count": encoder_frozen_params,
        "trainable_parameter_count": trainable_params,
        "frozen_parameter_count": frozen_params,
    }


def _prompt_usage_from_counts(counts: torch.Tensor, soft_prob_sum: torch.Tensor, soft_samples: int) -> Dict[str, object]:
    total = int(counts.sum().item())
    if total > 0:
        probs = counts.float() / float(total)
        nonzero = probs > 0
        hard_entropy = -torch.sum(probs[nonzero] * torch.log(probs[nonzero]))
        max_entropy = torch.log(torch.tensor(float(counts.numel())))
        hard_entropy_norm = hard_entropy / torch.clamp(max_entropy, min=1e-12)
    else:
        hard_entropy = torch.tensor(0.0)
        hard_entropy_norm = torch.tensor(0.0)

    if soft_samples > 0:
        mean_probs = soft_prob_sum / float(soft_samples)
        soft_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12))
        max_entropy = torch.log(torch.tensor(float(mean_probs.numel()), device=mean_probs.device))
        soft_entropy_norm = soft_entropy / torch.clamp(max_entropy, min=1e-12)
        soft_probs = [float(x) for x in mean_probs.detach().cpu().tolist()]
    else:
        soft_entropy = torch.tensor(0.0)
        soft_entropy_norm = torch.tensor(0.0)
        soft_probs = []

    top = torch.argsort(counts, descending=True)[: min(10, counts.numel())]
    return {
        "counts": [int(x) for x in counts.detach().cpu().tolist()],
        "total_selections": total,
        "nonzero_prompts": int((counts > 0).sum().item()),
        "top_selected_prompt_ids": [int(x) for x in top.detach().cpu().tolist() if int(counts[int(x)].item()) > 0],
        "hard_entropy": float(hard_entropy.detach().cpu()),
        "hard_entropy_normalized": float(hard_entropy_norm.detach().cpu()),
        "soft_mean_probs": soft_probs,
        "soft_entropy": float(soft_entropy.detach().cpu()),
        "soft_entropy_normalized": float(soft_entropy_norm.detach().cpu()),
    }


def _sample_replay_batch(
    replay_samples: List[Dict[str, object]],
    batch_size: int,
    rng: random.Random,
) -> Dict[str, object] | None:
    if not replay_samples or batch_size <= 0:
        return None
    rows = [replay_samples[rng.randrange(len(replay_samples))] for _ in range(batch_size)]
    return collate_l2p_batch(rows)


def _merge_batch_with_replay(
    batch: Dict[str, object],
    replay_batch: Dict[str, object] | None,
) -> Dict[str, object]:
    if replay_batch is None:
        return batch
    return {
        "query_text": list(batch["query_text"]) + list(replay_batch["query_text"]),
        "target_tool_id": torch.cat([batch["target_tool_id"], replay_batch["target_tool_id"]], dim=0),
        "task_id": list(batch.get("task_id", [])) + list(replay_batch.get("task_id", [])),
        "target_tool_name": list(batch.get("target_tool_name", [])) + list(replay_batch.get("target_tool_name", [])),
    }


def _effective_replay_batch_size(args: argparse.Namespace, current_batch_size: int, replay_samples: List[Dict[str, object]]) -> int:
    if not replay_samples or not _replay_enabled(args):
        return 0
    if args.replay_batch_size is not None:
        return max(0, int(args.replay_batch_size))
    ratio = max(0.0, min(float(args.replay_ratio), 0.95))
    if ratio <= 0:
        return 0
    replay_size = int(round(current_batch_size * ratio / max(1e-12, 1.0 - ratio)))
    return max(1, replay_size)


def _current_loader_batch_size(args: argparse.Namespace, replay_samples: List[Dict[str, object]]) -> int:
    if not replay_samples or not _replay_enabled(args):
        return int(args.batch_size)
    if args.replay_batch_size is not None:
        return max(1, int(args.batch_size) - max(0, int(args.replay_batch_size)))
    ratio = max(0.0, min(float(args.replay_ratio), 0.95))
    return max(1, int(round(float(args.batch_size) * (1.0 - ratio))))


def _train_one_epoch(
    model: ToolL2PModel,
    loader: DataLoader,
    optimizer: AdamW,
    visible_ids: Sequence[int],
    current_ids: Sequence[int],
    old_ids: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    replay_samples: List[Dict[str, object]] | None = None,
    epoch: int = 1,
) -> Dict[str, float]:
    model.train()
    sums = {
        "loss": 0.0,
        "loss_cls": 0.0,
        "loss_key": 0.0,
        "loss_balance": 0.0,
        "candidate_count_mean": 0.0,
        "prompt_soft_entropy": 0.0,
        "prompt_soft_entropy_normalized": 0.0,
        "prompt_soft_max_prob": 0.0,
        "prompt_soft_min_prob": 0.0,
    }
    total = 0
    total_current = 0
    total_replay = 0
    prompt_counts = torch.zeros(model.prompt_pool_size, dtype=torch.long)
    soft_prob_sum = torch.zeros(model.prompt_pool_size, dtype=torch.float64)
    soft_samples = 0
    freeze_old_rows = bool(args.freeze_old_classifier_rows and old_ids)
    old_tensor = (
        torch.tensor([int(x) for x in old_ids], dtype=torch.long, device=device)
        if freeze_old_rows
        else torch.empty(0, dtype=torch.long, device=device)
    )
    frozen_old_weight = None
    frozen_old_bias = None
    if freeze_old_rows:
        with torch.no_grad():
            frozen_old_weight = model.classifier.weight.index_select(0, old_tensor).detach().clone()
            frozen_old_bias = model.classifier.bias.index_select(0, old_tensor).detach().clone()

    replay_samples = replay_samples or []
    rng = random.Random(args.seed + stage_index(args.stage) * 100003 + epoch * 997)
    for batch in loader:
        current_batch_size = len(batch["query_text"])
        replay_batch_size = _effective_replay_batch_size(args, current_batch_size, replay_samples)
        replay_batch = _sample_replay_batch(replay_samples, replay_batch_size, rng)
        batch = _merge_batch_with_replay(batch, replay_batch)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["query_text"],
            prompt_sampling=args.prompt_sampling_train,
            prompt_top_m=args.prompt_top_m,
        )
        loss, parts = l2p_regularized_loss(
            outputs,
            batch["target_tool_id"].to(device),
            loss_type=args.loss_type,
            visible_ids=visible_ids,
            current_task_tool_ids=current_ids,
            old_tool_ids=old_ids,
            num_current_negatives=args.num_current_negatives,
            num_old_negatives=args.num_old_negatives,
            lambda_key=args.lambda_key,
            prompt_balance_loss_weight=args.prompt_balance_loss_weight,
            num_tools=model.num_tools,
        )
        loss.backward()
        if freeze_old_rows:
            if model.classifier.weight.grad is not None:
                model.classifier.weight.grad.index_fill_(0, old_tensor, 0.0)
            if model.classifier.bias.grad is not None:
                model.classifier.bias.grad.index_fill_(0, old_tensor, 0.0)
        optimizer.step()
        if freeze_old_rows and frozen_old_weight is not None and frozen_old_bias is not None:
            with torch.no_grad():
                model.classifier.weight.index_copy_(0, old_tensor, frozen_old_weight)
                model.classifier.bias.index_copy_(0, old_tensor, frozen_old_bias)

        batch_size = len(batch["query_text"])
        total += batch_size
        total_current += current_batch_size
        total_replay += replay_batch_size if replay_batch is not None else 0
        for key in sums:
            sums[key] += parts[key] * batch_size
        ids = outputs["selected_prompt_ids"].detach().cpu().reshape(-1)
        prompt_counts += torch.bincount(ids, minlength=model.prompt_pool_size)
        with torch.no_grad():
            probs = torch.softmax(outputs["prompt_similarities"].detach().float(), dim=-1)
            soft_prob_sum += probs.sum(dim=0).double().cpu()
            soft_samples += probs.shape[0]

    if total == 0:
        return sums
    averaged = {key: value / total for key, value in sums.items()}
    averaged["current_samples_seen"] = float(total_current)
    averaged["replay_samples_seen"] = float(total_replay)
    averaged["replay_fraction_seen"] = float(total_replay) / float(max(total, 1))
    usage = _prompt_usage_from_counts(prompt_counts, soft_prob_sum, soft_samples)
    averaged.update({f"prompt_usage_{key}": value for key, value in usage.items() if key != "soft_mean_probs"})
    return averaged


@torch.no_grad()
def prompt_selection_histogram(
    model: ToolL2PModel,
    samples: List[Dict[str, object]],
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    counts = torch.zeros(model.prompt_pool_size, dtype=torch.long)
    soft_prob_sum = torch.zeros(model.prompt_pool_size, dtype=torch.float64)
    soft_samples = 0
    loader = DataLoader(ToolL2PDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collate_l2p_batch)
    for batch in loader:
        outputs = model(batch["query_text"])
        ids = outputs["selected_prompt_ids"].detach().cpu().reshape(-1)
        counts += torch.bincount(ids, minlength=model.prompt_pool_size)
        probs = torch.softmax(outputs["prompt_similarities"].detach().float(), dim=-1)
        soft_prob_sum += probs.sum(dim=0).double().cpu()
        soft_samples += probs.shape[0]
    return _prompt_usage_from_counts(counts, soft_prob_sum, soft_samples)


@torch.no_grad()
def classifier_norm_diagnostics(
    model: ToolL2PModel,
    old_tool_ids: Iterable[int],
    new_tool_ids: Iterable[int],
    visible_ids: Iterable[int],
) -> Dict[str, object]:
    def stats(ids: Iterable[int]) -> Dict[str, float]:
        idx = [int(x) for x in ids]
        if not idx:
            return {
                "num_tools": 0,
                "weight_norm_mean": 0.0,
                "weight_norm_max": 0.0,
                "bias_mean": 0.0,
                "bias_max": 0.0,
                "bias_abs_max": 0.0,
            }
        tensor_idx = torch.tensor(idx, dtype=torch.long, device=model.classifier.weight.device)
        weights = model.classifier.weight.index_select(0, tensor_idx).detach()
        bias = model.classifier.bias.index_select(0, tensor_idx).detach()
        norms = weights.norm(dim=1)
        return {
            "num_tools": len(idx),
            "weight_norm_mean": float(norms.mean().cpu()),
            "weight_norm_max": float(norms.max().cpu()),
            "bias_mean": float(bias.mean().cpu()),
            "bias_max": float(bias.max().cpu()),
            "bias_abs_max": float(bias.abs().max().cpu()),
        }

    return {
        "old_tools": stats(old_tool_ids),
        "new_tools": stats(new_tool_ids),
        "visible_tools": stats(visible_ids),
    }


@torch.no_grad()
def logit_diagnostics(
    model: ToolL2PModel,
    samples: List[Dict[str, object]],
    old_tool_ids: Iterable[int],
    new_tool_ids: Iterable[int],
    visible_ids: Iterable[int],
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    groups = {
        "old_tools": [int(x) for x in old_tool_ids],
        "new_tools": [int(x) for x in new_tool_ids],
        "visible_tools": [int(x) for x in visible_ids],
    }
    states: Dict[str, Dict[str, float]] = {
        name: {
            "num_tools": float(len(ids)),
            "value_sum": 0.0,
            "value_count": 0.0,
            "max_sum": 0.0,
            "max_sumsq": 0.0,
            "min_sum": 0.0,
            "row_count": 0.0,
            "global_max": float("-inf"),
            "global_min": float("inf"),
        }
        for name, ids in groups.items()
    }
    gold_sum = 0.0
    gold_count = 0
    loader = DataLoader(ToolL2PDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collate_l2p_batch)
    for batch in loader:
        outputs = model(batch["query_text"])
        logits = outputs["logits"].detach()
        targets = batch["target_tool_id"].to(logits.device)
        row_ids = torch.arange(targets.numel(), device=logits.device)
        gold = logits[row_ids, targets]
        gold_sum += float(gold.sum().cpu())
        gold_count += int(gold.numel())

        for name, ids in groups.items():
            if not ids:
                continue
            tensor_ids = torch.tensor(ids, dtype=torch.long, device=logits.device)
            values = logits.index_select(1, tensor_ids)
            states[name]["value_sum"] += float(values.sum().cpu())
            states[name]["value_count"] += float(values.numel())
            row_max = values.max(dim=1).values
            states[name]["max_sum"] += float(row_max.sum().cpu())
            states[name]["max_sumsq"] += float((row_max.float() ** 2).sum().cpu())
            states[name]["min_sum"] += float(values.min(dim=1).values.sum().cpu())
            states[name]["row_count"] += float(values.shape[0])
            states[name]["global_max"] = max(states[name]["global_max"], float(values.max().cpu()))
            states[name]["global_min"] = min(states[name]["global_min"], float(values.min().cpu()))

    out: Dict[str, object] = {}
    for name, state in states.items():
        value_count = max(state["value_count"], 1.0)
        row_count = max(state["row_count"], 1.0)
        max_mean = state["max_sum"] / row_count if state["num_tools"] > 0 else 0.0
        max_var = max(0.0, (state["max_sumsq"] / row_count) - max_mean * max_mean) if state["num_tools"] > 0 else 0.0
        out[name] = {
            "num_tools": int(state["num_tools"]),
            "logit_mean": state["value_sum"] / value_count if state["num_tools"] > 0 else 0.0,
            "per_sample_max_logit_mean": max_mean,
            "per_sample_max_logit_std": max_var ** 0.5,
            "per_sample_min_logit_mean": state["min_sum"] / row_count if state["num_tools"] > 0 else 0.0,
            "logit_global_max": state["global_max"] if state["num_tools"] > 0 else 0.0,
            "logit_global_min": state["global_min"] if state["num_tools"] > 0 else 0.0,
        }
    out["gold_tools"] = {
        "logit_mean": gold_sum / max(gold_count, 1),
        "num_values": gold_count,
    }
    return out


def _post_train_eval(
    args: argparse.Namespace,
    model: ToolL2PModel,
    mapping: Dict[str, object],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    eval_results: Dict[str, Dict[str, float]] = {}
    eval_plan = [(args.stage, "local"), (args.stage, "global")]
    for prior in STAGES[: stage_index(args.stage)]:
        eval_plan.append((prior, "global"))

    for eval_task, mode in eval_plan:
        key = f"{eval_task}_{mode}"
        try:
            metrics = run_l2p_evaluation(
                model=model,
                checkpoint_stage=args.stage,
                eval_task=eval_task,
                mode=mode,
                mapping=mapping,
                data_root=args.data_root,
                max_eval_samples_per_tool=args.max_eval_samples_per_tool,
                seed=args.seed,
                batch_size=args.batch_size,
                device=device,
            )
            eval_results[key] = metrics
            print(f"[eval:{key}] {format_metrics(metrics)} candidates={metrics['num_candidates']}")
        except FileNotFoundError as exc:
            eval_results[key] = {"error": str(exc), "valid_samples": 0}
            print(f"[eval:{key}] skipped: {exc}")
    return eval_results


def _mean_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["recall@1", "recall@3", "recall@5", "ndcg@5", "mrr"]
    if not items:
        return {key: 0.0 for key in keys} | {"num_tasks": 0}
    return {
        key: sum(float(item.get(key, 0.0)) for item in items) / len(items)
        for key in keys
    } | {"num_tasks": len(items)}


def build_continual_summary(metrics_path: str, stage: str, eval_results: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    current_key = f"{stage}_global"
    current = eval_results.get(current_key, {})
    old_items = [
        value
        for key, value in eval_results.items()
        if key.endswith("_global") and key != current_key and "error" not in value
    ]
    summary: Dict[str, object] = {
        "current_task_global": current,
        "old_task_global_average": _mean_metrics(old_items),
    }

    if os.path.exists(metrics_path):
        history = load_json(metrics_path)
    else:
        history = {}

    forgetting_items: List[Dict[str, float]] = []
    forgetting_by_task: Dict[str, Dict[str, float]] = {}
    for old_stage in STAGES[: stage_index(stage)]:
        current_old = eval_results.get(f"{old_stage}_global")
        learned_payload = history.get(old_stage, {})
        learned = learned_payload.get("eval", {}).get(f"{old_stage}_global")
        if not current_old or not learned:
            continue
        diffs = {
            key: float(learned.get(key, 0.0)) - float(current_old.get(key, 0.0))
            for key in ["recall@1", "recall@3", "recall@5", "ndcg@5", "mrr"]
        }
        forgetting_items.append(diffs)
        forgetting_by_task[old_stage] = diffs

    summary["forgetting_by_task"] = forgetting_by_task
    summary["average_forgetting"] = _mean_metrics(forgetting_items)
    return summary


def _load_json_if_exists(path: str) -> Dict[str, object]:
    if os.path.exists(path):
        data = load_json(path)
        if isinstance(data, dict):
            return data
    return {}


def _append_text(path: str, text: str) -> None:
    from .utils import ensure_parent

    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _args_config(args: argparse.Namespace) -> Dict[str, object]:
    keys = [
        "stage",
        "ckpt",
        "mini_subset",
        "full_data",
        "seed",
        "split_path",
        "mapping_path",
        "data_root",
        "output_dir",
        "metrics_path",
        "run_dir",
        "encoder_backend",
        "encoder_model",
        "llm_pooling",
        "hidden_dim",
        "prompt_pool_size",
        "prompt_length",
        "top_n",
        "l2p_variant",
        "loss_type",
        "freeze_old_classifier_rows",
        "freeze_fusion_after_base",
        "prompt_balance_loss_weight",
        "prompt_top_m",
        "prompt_sampling_train",
        "num_current_negatives",
        "num_old_negatives",
        "replay_strategy",
        "replay_buffer_size",
        "replay_samples_per_task",
        "replay_samples_per_tool",
        "replay_ratio",
        "replay_batch_size",
        "replay_after_each_task",
        "save_replay_buffer",
        "load_replay_buffer",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "lambda_key",
        "device",
    ]
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def _write_run_artifacts(
    *,
    args: argparse.Namespace,
    stage_payload: Dict[str, object],
    prompt_payload: Dict[str, object],
    classifier_payload: Dict[str, object],
    trainability: Dict[str, object],
) -> None:
    ensure_dir(args.run_dir)

    config_path = os.path.join(args.run_dir, "config.json")
    config = _load_json_if_exists(config_path)
    config["variant"] = args.l2p_variant
    config["replay"] = "enabled" if _replay_enabled(args) else "disabled"
    config.setdefault("distillation", "disabled")
    stages = config.setdefault("stages", {})
    if isinstance(stages, dict):
        stages[args.stage] = {
            "args": _args_config(args),
            "trainability": trainability,
        }
    save_json(config, config_path)

    prompt_path = os.path.join(args.run_dir, "prompt_usage.json")
    prompt_usage = _load_json_if_exists(prompt_path)
    prompt_usage[args.stage] = prompt_payload
    save_json(prompt_usage, prompt_path)

    classifier_path = os.path.join(args.run_dir, "classifier_stats.json")
    classifier_stats = _load_json_if_exists(classifier_path)
    classifier_stats[args.stage] = classifier_payload
    save_json(classifier_stats, classifier_path)

    log_path = os.path.join(args.run_dir, "train.log")
    evals = stage_payload.get("eval", {})
    current_key = f"{args.stage}_global"
    current = evals.get(current_key, {}) if isinstance(evals, dict) else {}
    log_line = (
        f"stage={args.stage} variant={args.l2p_variant} loss_type={args.loss_type} "
        f"replay={'enabled' if _replay_enabled(args) else 'disabled'} distillation=disabled "
        f"current_R@1={current.get('recall@1', 0.0) if isinstance(current, dict) else 0.0:.4f} "
        f"current_R@5={current.get('recall@5', 0.0) if isinstance(current, dict) else 0.0:.4f} "
        f"current_MRR={current.get('mrr', 0.0) if isinstance(current, dict) else 0.0:.4f}\n"
    )
    _append_text(log_path, log_line)


def main() -> None:
    args = normalize_args(parse_args())
    args.stage = check_stage(args.stage)

    set_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir)

    split, mapping, mapping_path = _load_or_create_split(args)
    task_to_tool_ids = mapping["task_to_tool_ids"]
    visible_ids = visible_tool_ids(args.stage, task_to_tool_ids)
    current_ids = current_task_tool_ids(args.stage, task_to_tool_ids)
    old_ids = previous_visible_tool_ids(args.stage, task_to_tool_ids)

    print(
        f"[split] loaded split={args.split_path} mapping={mapping_path} "
        f"num_total_tools={mapping.get('num_total_tools', len(mapping['tool_to_id']))}"
    )
    if args.l2p_variant == "regularized_no_replay":
        variant_name = "L2P-Tool-Regularized-NoReplay"
    elif args.l2p_variant == "replay":
        variant_name = "L2P-Tool-Replay"
    else:
        variant_name = "L2P-Tool-NoReplay"
    print(f"[mode] {'mini_subset' if args.mini_subset else 'full_data'} encoder={args.encoder_backend}")
    print(f"[variant] Variant: {variant_name}")
    print(f"[variant] Replay: {'enabled' if _replay_enabled(args) else 'disabled'}")
    print("[variant] Distillation: disabled")
    print(
        f"[loss] loss_type={args.loss_type} current_negatives={args.num_current_negatives} "
        f"old_negatives={args.num_old_negatives} prompt_balance_weight={args.prompt_balance_loss_weight}"
    )
    if _replay_enabled(args):
        print(
            f"[replay] strategy={args.replay_strategy} buffer_size={args.replay_buffer_size} "
            f"samples_per_task={args.replay_samples_per_task} samples_per_tool={args.replay_samples_per_tool} "
            f"ratio={args.replay_ratio} replay_batch_size={args.replay_batch_size}"
        )
    print(
        f"[stage] {args.stage}: current_task_tool_ids={len(current_ids)} "
        f"visible_tool_ids={len(visible_ids)} old_tool_ids={len(old_ids)}"
    )

    train_samples = _build_train_samples(args, mapping)
    if not train_samples:
        raise RuntimeError(
            f"No train samples for stage={args.stage} after mini-subset filtering. "
            "Check retrieval files and sampled split."
        )
    sample_mode = "mini-subset" if args.mini_subset else "full-data"
    print(f"[train] materialized {len(train_samples)} {sample_mode} samples")
    replay_samples = _load_replay_buffer(args)
    replay_stats_before = _replay_stats(replay_samples, visible_ids=old_ids) if replay_samples else {
        "buffer_size": 0,
        "samples_per_task": {},
        "samples_per_tool": {},
        "unique_tools_covered": 0,
        "visible_tools": len(set(old_ids)),
        "tool_coverage_ratio": 0.0,
        "replay_strategy": args.replay_strategy,
    }
    if _replay_enabled(args):
        print_json("[replay] buffer before training", replay_stats_before)

    model = _build_or_load_model(args, mapping, device)
    _apply_freeze_controls(
        model,
        stage=args.stage,
        freeze_fusion_after_base=args.freeze_fusion_after_base,
    )
    if args.freeze_old_classifier_rows and old_ids:
        print(f"[freeze] freeze_old_classifier_rows enabled: {len(old_ids)} old classifier rows will be restored after each optimizer step")
    elif args.freeze_old_classifier_rows:
        print("[freeze] freeze_old_classifier_rows enabled but no old rows exist for base stage")
    if args.freeze_fusion_after_base and args.stage != "base":
        print("[freeze] freeze_fusion_after_base enabled: fusion parameters are frozen")
    trainability = _parameter_trainability(model)
    print_json("[trainable] parameter status", trainability)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    current_batch_size = _current_loader_batch_size(args, replay_samples)
    if _replay_enabled(args):
        print(
            f"[replay] current_batch_size={current_batch_size} "
            f"effective_replay_batch_size={_effective_replay_batch_size(args, current_batch_size, replay_samples)}"
        )
    loader = DataLoader(
        ToolL2PDataset(train_samples),
        batch_size=current_batch_size,
        shuffle=True,
        collate_fn=collate_l2p_batch,
    )

    epoch_logs: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        metrics = _train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            visible_ids=visible_ids,
            current_ids=current_ids,
            old_ids=old_ids,
            args=args,
            device=device,
            replay_samples=replay_samples,
            epoch=epoch,
        )
        epoch_logs.append(metrics)
        print(
            f"[train] epoch={epoch}/{args.epochs} "
            f"loss={metrics['loss']:.4f} cls={metrics['loss_cls']:.4f} "
            f"key={metrics['loss_key']:.4f} balance={metrics.get('loss_balance', 0.0):.4f} "
            f"prompt_entropy={metrics.get('prompt_usage_hard_entropy_normalized', 0.0):.4f} "
            f"soft_entropy={metrics.get('prompt_usage_soft_entropy_normalized', 0.0):.4f} "
            f"top_prompts={metrics.get('prompt_usage_top_selected_prompt_ids', [])}"
        )

    prompt_hist = prompt_selection_histogram(model, train_samples, batch_size=args.batch_size)
    norm_diag = classifier_norm_diagnostics(model, old_ids, current_ids, visible_ids)
    logit_diag = logit_diagnostics(
        model,
        train_samples,
        old_tool_ids=old_ids,
        new_tool_ids=current_ids,
        visible_ids=visible_ids,
        batch_size=args.batch_size,
    )
    replay_buffer_after = _merge_replay_buffer(args, replay_samples, train_samples)
    replay_stats_after = (
        _replay_stats(replay_buffer_after, visible_ids=visible_ids)
        if _replay_enabled(args)
        else {
            "buffer_size": 0,
            "samples_per_task": {},
            "samples_per_tool": {},
            "unique_tools_covered": 0,
            "visible_tools": len(set(visible_ids)),
            "tool_coverage_ratio": 0.0,
            "replay_strategy": "none",
        }
    )
    if _replay_enabled(args):
        replay_stats_after["replay_strategy"] = args.replay_strategy
        print_json("[replay] buffer after update", replay_stats_after)
        _write_replay_artifacts(args, buffer=replay_buffer_after, stage=args.stage, stats=replay_stats_after)
    print_json("[diagnostic] prompt selection histogram", prompt_hist)
    print_json("[diagnostic] classifier norm old/new", norm_diag)
    print_json("[diagnostic] logit stats old/new", logit_diag)

    save_path = args.save_path or os.path.join(args.output_dir, f"{args.stage}.pt")
    ensure_dir(os.path.dirname(save_path))
    config = checkpoint_config(
        model,
        stage=args.stage,
        encoder_backend=args.encoder_backend,
        encoder_model=args.encoder_model,
        llm_pooling=args.llm_pooling,
        split_path=args.split_path,
        mapping_path=mapping_path,
        l2p_variant=args.l2p_variant,
        loss_type=args.loss_type,
    )
    save_checkpoint(
        save_path,
        model,
        config,
        extra={
            "train_epochs": epoch_logs,
            "prompt_histogram": prompt_hist,
            "classifier_norm": norm_diag,
            "logit_stats": logit_diag,
            "l2p_variant": args.l2p_variant,
            "loss_type": args.loss_type,
            "replay_stats_before": replay_stats_before,
            "replay_stats_after": replay_stats_after,
            "split": split,
        },
    )
    print(f"[ckpt] saved {save_path}")

    eval_results = _post_train_eval(args, model, mapping, device)
    cl_summary = build_continual_summary(args.metrics_path, args.stage, eval_results)
    print_json("[continual] summary", cl_summary)
    prompt_payload = {
        "epoch_usage": epoch_logs,
        "final_deterministic_usage": prompt_hist,
    }
    classifier_payload = {
        "classifier_norm": norm_diag,
        "logit_stats": logit_diag,
    }
    stage_payload = {
        "checkpoint": save_path,
        "split_path": args.split_path,
        "mapping_path": mapping_path,
        "variant": args.l2p_variant,
        "loss_type": args.loss_type,
        "replay": "enabled" if _replay_enabled(args) else "disabled",
        "distillation": "disabled",
        "train": {
            "num_samples": len(train_samples),
            "epochs": epoch_logs,
            "visible_tool_ids": len(visible_ids),
            "current_task_tool_ids": len(current_ids),
            "old_tool_ids": len(old_ids),
            "replay_samples_available": len(replay_samples),
            "replay_stats_before": replay_stats_before,
            "replay_stats_after": replay_stats_after,
            "trainability": trainability,
        },
        "diagnostics": {
            "prompt_selection_histogram": prompt_hist,
            "classifier_norm": norm_diag,
            "logit_stats": logit_diag,
        },
        "replay_stats": replay_stats_after,
        "eval": eval_results,
        "continual_summary": cl_summary,
    }
    _write_run_artifacts(
        args=args,
        stage_payload=stage_payload,
        prompt_payload=prompt_payload,
        classifier_payload=classifier_payload,
        trainability=trainability,
    )
    append_metrics(
        args.metrics_path,
        args.stage,
        stage_payload,
    )
    print(f"[metrics] updated {args.metrics_path}")
    print(f"[artifacts] updated {args.run_dir}/config.json prompt_usage.json classifier_stats.json train.log")


if __name__ == "__main__":
    main()
