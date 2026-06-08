import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


STAGES = ["base", "task1", "task2", "task3"]


def check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {STAGES}.")
    return stage


def stage_index(stage: str) -> int:
    return STAGES.index(check_stage(stage))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def default_mapping_path(split_path: str, seed: int) -> str:
    split_dir = Path(split_path).parent
    return str(split_dir / f"tool_mapping_seed{seed}.json")


def visible_tool_ids(stage: str, task_to_tool_ids: Dict[str, List[int]]) -> List[int]:
    idx = stage_index(stage)
    ids: List[int] = []
    for s in STAGES[: idx + 1]:
        ids.extend(int(x) for x in task_to_tool_ids.get(s, []))
    return ids


def previous_visible_tool_ids(stage: str, task_to_tool_ids: Dict[str, List[int]]) -> List[int]:
    idx = stage_index(stage)
    ids: List[int] = []
    for s in STAGES[:idx]:
        ids.extend(int(x) for x in task_to_tool_ids.get(s, []))
    return ids


def current_task_tool_ids(stage: str, task_to_tool_ids: Dict[str, List[int]]) -> List[int]:
    check_stage(stage)
    return [int(x) for x in task_to_tool_ids.get(stage, [])]


def stable_sample(items: Iterable[Any], max_items: Optional[int], seed: int) -> List[Any]:
    values = list(items)
    if max_items is None or max_items <= 0 or len(values) <= max_items:
        return values
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(values)), max_items))
    return [values[i] for i in idxs]


def append_metrics(metrics_path: str, stage: str, payload: Dict[str, Any]) -> None:
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
    else:
        metrics = {}
    metrics[stage] = payload
    save_json(metrics, metrics_path)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def print_json(title: str, payload: Dict[str, Any]) -> None:
    print(title)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

