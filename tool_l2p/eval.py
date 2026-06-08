import argparse
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from .data import (
    ToolL2PDataset,
    collate_l2p_batch,
    load_raw_samples,
    load_split_and_mapping,
    materialize_samples,
)
from .metrics import format_metrics, merge_metric_batches, ranking_metrics
from .model import ToolL2PModel, load_checkpoint_model
from .utils import STAGES, current_task_tool_ids, default_mapping_path, resolve_device, visible_tool_ids


@torch.no_grad()
def evaluate_samples(
    model: ToolL2PModel,
    samples: List[Dict[str, object]],
    candidate_tool_ids: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dataset = ToolL2PDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_l2p_batch)
    metric_batches: List[Dict[str, float]] = []
    candidate_ids = [int(x) for x in candidate_tool_ids]
    for batch in loader:
        outputs = model(batch["query_text"])
        metrics = ranking_metrics(
            outputs["logits"],
            batch["target_tool_id"].to(device),
            candidate_ids,
            ks=(1, 3, 5),
        )
        metric_batches.append(metrics)
    return merge_metric_batches(metric_batches)


def prepare_eval_samples(
    *,
    eval_task: str,
    mapping: Dict[str, object],
    data_root: str,
    max_eval_samples_per_tool: int,
    seed: int,
) -> List[Dict[str, object]]:
    raw = load_raw_samples(eval_task, "eval", data_root=data_root)
    task_to_tool_ids = mapping["task_to_tool_ids"]
    return materialize_samples(
        raw,
        mapping["tool_to_id"],
        sampled_tool_ids=task_to_tool_ids.get(eval_task, []),
        max_samples_per_tool=max_eval_samples_per_tool,
        seed=seed + 50000 + STAGES.index(eval_task) * 1291,
    )


def candidate_ids_for_mode(
    *,
    mode: str,
    checkpoint_stage: str,
    eval_task: str,
    mapping: Dict[str, object],
) -> List[int]:
    task_to_tool_ids = mapping["task_to_tool_ids"]
    if mode == "global":
        return visible_tool_ids(checkpoint_stage, task_to_tool_ids)
    if mode == "local":
        return current_task_tool_ids(eval_task, task_to_tool_ids)
    raise ValueError("mode must be global or local")


def run_l2p_evaluation(
    *,
    model: ToolL2PModel,
    checkpoint_stage: str,
    eval_task: str,
    mode: str,
    mapping: Dict[str, object],
    data_root: str,
    max_eval_samples_per_tool: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    samples = prepare_eval_samples(
        eval_task=eval_task,
        mapping=mapping,
        data_root=data_root,
        max_eval_samples_per_tool=max_eval_samples_per_tool,
        seed=seed,
    )
    candidates = candidate_ids_for_mode(
        mode=mode,
        checkpoint_stage=checkpoint_stage,
        eval_task=eval_task,
        mapping=mapping,
    )
    metrics = evaluate_samples(model, samples, candidates, batch_size=batch_size, device=device)
    metrics["num_eval_samples"] = len(samples)
    metrics["num_candidates"] = len(candidates)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the minimal L2P-Tool baseline.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval_task", choices=STAGES, required=True)
    parser.add_argument("--mode", choices=["global", "local"], default="global")
    parser.add_argument("--mini_subset", action="store_true")
    parser.add_argument("--full_data", action="store_true")
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--mapping_path", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--max_eval_samples_per_tool", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encoder_backend",
        default=None,
        choices=["auto", "sentence_transformer", "hashing", "toolhcl", "toolhcl_st", "toolhcl_llm"],
        help="Optional checkpoint-load override.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mini_subset and args.full_data:
        raise ValueError("Choose only one of --mini_subset or --full_data.")
    if not args.mini_subset:
        args.full_data = True
    if args.full_data:
        args.max_eval_samples_per_tool = 0
    if args.encoder_backend == "toolhcl":
        args.encoder_backend = "toolhcl_llm"
    if args.split_path is None:
        args.split_path = (
            f"tool_l2p_runs/splits/mini_seed{args.seed}.json"
            if args.mini_subset
            else f"tool_l2p_runs_full/splits/full_seed{args.seed}.json"
        )
    return args


def main() -> None:
    args = normalize_args(parse_args())

    device = resolve_device(args.device)
    model, ckpt_config, _ = load_checkpoint_model(args.ckpt, device=device, encoder_backend_override=args.encoder_backend)
    checkpoint_stage = str(ckpt_config.get("stage", "base"))
    mapping_path = args.mapping_path or str(ckpt_config.get("mapping_path") or default_mapping_path(args.split_path, args.seed))
    _, mapping = load_split_and_mapping(args.split_path, mapping_path, args.seed)

    metrics = run_l2p_evaluation(
        model=model,
        checkpoint_stage=checkpoint_stage,
        eval_task=args.eval_task,
        mode=args.mode,
        mapping=mapping,
        data_root=args.data_root,
        max_eval_samples_per_tool=args.max_eval_samples_per_tool,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )
    print(
        f"[eval] ckpt_stage={checkpoint_stage} eval_task={args.eval_task} mode={args.mode} "
        f"candidates={metrics['num_candidates']} samples={metrics['num_eval_samples']}"
    )
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
