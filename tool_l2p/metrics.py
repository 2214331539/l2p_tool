import math
from typing import Dict, Iterable, List

import torch


def ranking_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_tool_ids: Iterable[int],
    ks: Iterable[int] = (1, 3, 5),
) -> Dict[str, float]:
    candidate_ids = [int(x) for x in candidate_tool_ids]
    ks = [int(k) for k in ks]
    out = {f"recall@{k}": 0.0 for k in ks}
    out.update({f"ndcg@{k}": 0.0 for k in ks})
    out["mrr"] = 0.0
    out["valid_samples"] = 0

    if logits.numel() == 0 or not candidate_ids:
        return out

    device = logits.device
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    candidate_logits = logits.index_select(1, candidate_tensor)
    order = torch.argsort(candidate_logits, dim=1, descending=True)

    id_to_local = {tool_id: i for i, tool_id in enumerate(candidate_ids)}
    max_k = min(max(ks), len(candidate_ids))

    for row_idx, target in enumerate(targets.tolist()):
        target = int(target)
        if target not in id_to_local:
            continue
        out["valid_samples"] += 1
        target_local = id_to_local[target]
        ranked = order[row_idx]
        matches = (ranked == target_local).nonzero(as_tuple=False)
        if matches.numel() == 0:
            continue
        rank = int(matches[0].item()) + 1
        out["mrr"] += 1.0 / rank
        for k in ks:
            k_eff = min(k, max_k)
            if rank <= k_eff:
                out[f"recall@{k}"] += 1.0
                out[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1)

    valid = out["valid_samples"]
    if valid > 0:
        for k in ks:
            out[f"recall@{k}"] = out[f"recall@{k}"] / valid * 100.0
            out[f"ndcg@{k}"] = out[f"ndcg@{k}"] / valid * 100.0
        out["mrr"] = out["mrr"] / valid * 100.0
    return out


def merge_metric_batches(metric_batches: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_batches:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "ndcg@1": 0.0,
            "ndcg@3": 0.0,
            "ndcg@5": 0.0,
            "mrr": 0.0,
            "valid_samples": 0,
        }

    count = sum(int(m.get("valid_samples", 0)) for m in metric_batches)
    keys = [k for k in metric_batches[0] if k != "valid_samples"]
    out = {"valid_samples": count}
    for key in keys:
        if count == 0:
            out[key] = 0.0
            continue
        out[key] = sum(float(m.get(key, 0.0)) * int(m.get("valid_samples", 0)) for m in metric_batches) / count
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr"]
    parts = [f"{k}={metrics.get(k, 0.0):.2f}" for k in keys]
    parts.append(f"valid={int(metrics.get('valid_samples', 0))}")
    return " ".join(parts)

