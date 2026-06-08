import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .utils import STAGES, check_stage, default_mapping_path, ensure_parent, load_json, save_json, stage_index


QUERY_FIELDS = ("query_text", "query", "instruction", "input", "text", "utterance")
TARGET_NAME_FIELDS = ("target_tool_name", "target_name", "tool_name", "tool", "api")
TARGET_ID_FIELDS = ("target_tool_id", "target_id", "tool_id", "label")
TARGET_PATTERN = re.compile(r"<<(.+?)&&(.+?)>>")
TOOL_TEXT_PATTERN = re.compile(
    r"Tool:\s*(.*?)\.\.?\s*Description:.*?\bAPI:\s*(.*?)\.\s*(?:API Description|$)",
    re.IGNORECASE | re.DOTALL,
)


def canonical_tool_name(tool_name: str, api_name: Optional[str] = None) -> str:
    tool_name = str(tool_name).strip()
    if api_name is None:
        match = TARGET_PATTERN.match(tool_name)
        if match:
            return f"<<{match.group(1).strip()}&&{match.group(2).strip()}>>"
        return tool_name
    return f"<<{tool_name}&&{str(api_name).strip()}>>"


def _load_json_or_pt(path: str) -> Any:
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu", weights_only=False)
    return load_json(path)


def _as_record_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("target_tool_id", key)
                records.append(item)
        return records
    return []


def _extract_query(item: Dict[str, Any]) -> Optional[str]:
    for field in QUERY_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    conversations = item.get("conversations")
    if isinstance(conversations, list):
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            if conv.get("role") == "user" and isinstance(conv.get("content"), str):
                text = conv["content"].strip()
                if text:
                    return text
    return None


def _assistant_target_from_conversations(item: Dict[str, Any]) -> Optional[str]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        if conv.get("role") == "assistant" and isinstance(conv.get("content"), str):
            text = conv["content"].strip()
            if text:
                return text
    return None


def _extract_target(
    item: Dict[str, Any],
    id_to_name: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[Any]]:
    target_name: Optional[str] = None
    target_id: Optional[Any] = None

    assistant_target = _assistant_target_from_conversations(item)
    if assistant_target:
        match = TARGET_PATTERN.search(assistant_target)
        target_name = canonical_tool_name(match.group(0) if match else assistant_target)

    if target_name is None:
        for field in TARGET_NAME_FIELDS:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                if field == "tool_name" and isinstance(item.get("api_name"), str):
                    target_name = canonical_tool_name(value, item["api_name"])
                else:
                    target_name = canonical_tool_name(value)
                break

    for field in TARGET_ID_FIELDS:
        if field in item and item[field] is not None:
            target_id = item[field]
            break

    if target_name is None and target_id is not None and id_to_name:
        target_name = id_to_name.get(str(target_id))

    if target_name is None and target_id is not None:
        target_name = f"id:{target_id}"

    return target_name, target_id


def _base_tool_lookup(data_root: str) -> Dict[str, str]:
    path = Path(data_root) / "train" / "raw" / "train_tools_with_id.json"
    if not path.exists():
        return {}
    data = load_json(path)
    lookup: Dict[str, str] = {}
    if isinstance(data, list):
        for tool in data:
            if not isinstance(tool, dict) or "tool_id" not in tool:
                continue
            name = tool.get("tool_name") or tool.get("name")
            api = tool.get("api_name") or tool.get("api")
            if name is None:
                continue
            lookup[str(tool["tool_id"])] = canonical_tool_name(str(name), str(api) if api is not None else None)
    return lookup


def _tool_name_from_tool_record(tool: Dict[str, Any]) -> Optional[str]:
    name = tool.get("tool_name") or tool.get("name")
    api = tool.get("api_name") or tool.get("api")
    if name is not None:
        return canonical_tool_name(str(name), str(api) if api is not None else None)

    text = tool.get("text") or tool.get("raw_intent") or tool.get("description")
    if isinstance(text, str):
        match = TOOL_TEXT_PATTERN.search(text)
        if match:
            return canonical_tool_name(match.group(1).strip(), match.group(2).strip())
    return None


def load_stage_tool_keys(stage: str, data_root: str = "data") -> List[str]:
    stage = check_stage(stage)
    root = Path(data_root)
    candidates: List[Path]
    if stage == "base":
        candidates = [
            root / "train" / "raw" / "train_tools_with_id.json",
            root / "base" / "raw" / "train_tools_with_id.json",
        ]
    else:
        candidates = [
            root / stage / "raw" / f"{stage}_tools_with_id.json",
            root / stage / "clusters" / f"{stage}_tools_with_id.json",
        ]

    for path in candidates:
        if not path.exists():
            continue
        data = load_json(path)
        records = _as_record_list(data)
        keys = []
        for record in records:
            key = _tool_name_from_tool_record(record)
            if key:
                keys.append(key)
        if keys:
            return sorted(set(keys))
    return []


def _candidate_paths(stage: str, split: str, data_root: str) -> List[str]:
    check_stage(stage)
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")

    filename = "retrieval_train.json" if split == "train" else "retrieval_eval.json"
    root = Path(data_root)
    paths: List[Path] = []

    if stage == "base":
        paths.extend(
            [
                root / "base" / "raw" / filename,
                root / "train" / "raw" / filename,
                root / "base" / "clusters" / "base_retrieval_clean.json",
                root / "train" / "clusters" / "base_retrieval_clean.json",
            ]
        )
        if split == "train":
            paths.append(root / "train" / "cache" / "training_samples_cache.pt")
        else:
            paths.extend(
                [
                    root / "base" / "raw" / "retrieval_train.json",
                    root / "train" / "raw" / "retrieval_train.json",
                    root / "train" / "cache" / "training_samples_cache.pt",
                ]
            )
    else:
        task_dir = root / stage
        clean_name = f"{stage}_retrieval_clean.json"
        paths.extend(
            [
                task_dir / "raw" / filename,
                task_dir / "clusters" / clean_name,
            ]
        )
        if split == "eval":
            paths.append(task_dir / "raw" / "retrieval_train.json")
        paths.append(task_dir / "raw" / "memorization_train.json")

    return [str(p) for p in paths]


def resolve_retrieval_path(stage: str, split: str, data_root: str = "data") -> str:
    candidates = _candidate_paths(stage, split, data_root)
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No retrieval file found for stage={stage} split={split}. Tried: {candidates}"
    )


def load_raw_samples(
    stage: str,
    split: str,
    data_root: str = "data",
    path_override: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    stage = check_stage(stage)
    path = path_override or resolve_retrieval_path(stage, split, data_root)
    data = _load_json_or_pt(path)
    records = _as_record_list(data)
    id_lookup = _base_tool_lookup(data_root) if stage == "base" else {}

    samples: List[Dict[str, Any]] = []
    for item in records:
        query_text = _extract_query(item)
        target_name, source_target_id = _extract_target(item, id_lookup)
        if not query_text or not target_name:
            continue
        samples.append(
            {
                "query_text": query_text,
                "target_tool_name": target_name,
                "target_tool_key": target_name,
                "source_target_id": source_target_id,
                "task_id": stage,
            }
        )

    if verbose:
        fallback_note = ""
        expected = Path(data_root) / ("train" if stage == "base" else stage) / "raw"
        expected_file = expected / ("retrieval_train.json" if split == "train" else "retrieval_eval.json")
        if str(path) != str(expected_file) and not (stage == "base" and path.endswith("data/base/raw/" + expected_file.name)):
            fallback_note = " (fallback source)"
        print(f"[data] {stage}/{split}: loaded {len(samples)} samples from {path}{fallback_note}")
    return samples


def group_by_tool(samples: Iterable[Dict[str, Any]], key: str = "target_tool_key") -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        tool_key = sample.get(key)
        if tool_key is not None:
            grouped[str(tool_key)].append(sample)
    return dict(grouped)


def _sample_tool_keys(tool_keys: List[str], max_tools: int, seed: int) -> List[str]:
    keys = sorted(set(tool_keys))
    if max_tools <= 0 or len(keys) <= max_tools:
        return keys
    rng = random.Random(seed)
    return sorted(rng.sample(keys, max_tools))


def _normalize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    tool_to_id = {str(k): int(v) for k, v in mapping["tool_to_id"].items()}
    id_to_tool = {int(k): str(v) for k, v in mapping["id_to_tool"].items()}
    task_to_tool_ids = {
        stage: [int(x) for x in mapping.get("task_to_tool_ids", {}).get(stage, [])]
        for stage in STAGES
    }
    return {
        **mapping,
        "tool_to_id": tool_to_id,
        "id_to_tool": id_to_tool,
        "task_to_tool_ids": task_to_tool_ids,
    }


def load_split_and_mapping(split_path: str, mapping_path: Optional[str], seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    split = load_json(split_path)
    effective_seed = int(split.get("seed", seed))
    mapping_path = mapping_path or default_mapping_path(split_path, effective_seed)
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Split exists at {split_path}, but mapping file is missing: {mapping_path}")
    mapping = _normalize_mapping(load_json(mapping_path))
    return split, mapping


def create_or_load_sampled_split(
    train_samples_by_stage: Optional[Dict[str, List[Dict[str, Any]]]],
    split_path: str,
    mapping_path: Optional[str],
    seed: int,
    max_base_tools: int,
    max_incremental_tools: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if os.path.exists(split_path):
        split, mapping = load_split_and_mapping(split_path, mapping_path, seed)
        return split, mapping, mapping_path or default_mapping_path(split_path, int(split.get("seed", seed)))

    if train_samples_by_stage is None:
        raise ValueError("train_samples_by_stage is required when creating a new split.")

    split_tool_keys: Dict[str, List[str]] = {}
    for stage in STAGES:
        grouped = group_by_tool(train_samples_by_stage.get(stage, []))
        max_tools = max_base_tools if stage == "base" else max_incremental_tools
        split_tool_keys[stage] = _sample_tool_keys(
            list(grouped.keys()),
            max_tools=max_tools,
            seed=seed + stage_index(stage) * 1009,
        )

    tool_to_id: Dict[str, int] = {}
    id_to_tool: Dict[int, str] = {}
    task_to_tool_ids: Dict[str, List[int]] = {stage: [] for stage in STAGES}
    next_id = 0
    for stage in STAGES:
        for tool_key in split_tool_keys[stage]:
            if tool_key not in tool_to_id:
                tool_to_id[tool_key] = next_id
                id_to_tool[next_id] = tool_key
                next_id += 1
            task_to_tool_ids[stage].append(tool_to_id[tool_key])

    split = {
        "seed": seed,
        "base": task_to_tool_ids["base"],
        "task1": task_to_tool_ids["task1"],
        "task2": task_to_tool_ids["task2"],
        "task3": task_to_tool_ids["task3"],
        "tool_keys": split_tool_keys,
        "params": {
            "max_base_tools": max_base_tools,
            "max_incremental_tools": max_incremental_tools,
        },
    }
    mapping = {
        "seed": seed,
        "num_total_tools": next_id,
        "tool_to_id": tool_to_id,
        "id_to_tool": {str(k): v for k, v in id_to_tool.items()},
        "task_to_tool_ids": task_to_tool_ids,
    }

    mapping_path = mapping_path or default_mapping_path(split_path, seed)
    ensure_parent(split_path)
    save_json(split, split_path)
    save_json(mapping, mapping_path)
    return split, _normalize_mapping(mapping), mapping_path


def materialize_samples(
    raw_samples: List[Dict[str, Any]],
    tool_to_id: Dict[str, int],
    sampled_tool_ids: Iterable[int],
    max_samples_per_tool: int,
    seed: int,
) -> List[Dict[str, Any]]:
    sampled_ids = {int(x) for x in sampled_tool_ids}
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in raw_samples:
        tool_key = str(sample["target_tool_key"])
        if tool_key not in tool_to_id:
            continue
        tool_id = int(tool_to_id[tool_key])
        if tool_id not in sampled_ids:
            continue
        converted = {
            "query_text": sample["query_text"],
            "target_tool_id": tool_id,
            "target_tool_name": sample.get("target_tool_name"),
            "task_id": sample.get("task_id"),
        }
        grouped[tool_id].append(converted)

    out: List[Dict[str, Any]] = []
    for offset, tool_id in enumerate(sorted(grouped)):
        rows = sorted(grouped[tool_id], key=lambda x: x["query_text"])
        if max_samples_per_tool > 0 and len(rows) > max_samples_per_tool:
            rng = random.Random(seed + tool_id * 9176 + offset)
            rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), max_samples_per_tool))]
        out.extend(rows)
    return out


class ToolL2PDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_l2p_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query_text": [item["query_text"] for item in batch],
        "target_tool_id": torch.tensor([int(item["target_tool_id"]) for item in batch], dtype=torch.long),
        "task_id": [item.get("task_id") for item in batch],
        "target_tool_name": [item.get("target_tool_name") for item in batch],
    }
