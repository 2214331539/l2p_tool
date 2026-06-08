import argparse
import json
import os
from typing import Dict, Iterable, List

from .utils import STAGES


METRIC_KEYS = ["recall@1", "recall@5", "mrr"]


def _load(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _avg_metric(rows: List[Dict[str, object]], key: str) -> float:
    return _mean(float(row.get(key, 0.0)) for row in rows)


def _current_rows(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for stage in STAGES:
        payload = metrics.get(stage, {})
        if not isinstance(payload, dict):
            continue
        evals = payload.get("eval", {})
        if not isinstance(evals, dict):
            continue
        current = evals.get(f"{stage}_global", {})
        if isinstance(current, dict):
            rows.append(current)
    return rows


def _old_rows(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for stage in STAGES:
        payload = metrics.get(stage, {})
        if not isinstance(payload, dict):
            continue
        evals = payload.get("eval", {})
        if not isinstance(evals, dict):
            continue
        for key, value in evals.items():
            if key.endswith("_global") and key != f"{stage}_global" and isinstance(value, dict):
                rows.append(value)
    return rows


def _final_old_rows(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    payload = metrics.get("task3", {})
    if not isinstance(payload, dict):
        return []
    evals = payload.get("eval", {})
    if not isinstance(evals, dict):
        return []
    return [
        value
        for key, value in evals.items()
        if key.endswith("_global") and key != "task3_global" and isinstance(value, dict)
    ]


def _prompt_entropy(metrics: Dict[str, object]) -> float:
    vals = []
    for stage in STAGES:
        payload = metrics.get(stage, {})
        if not isinstance(payload, dict):
            continue
        diag = payload.get("diagnostics", {})
        if not isinstance(diag, dict):
            continue
        prompt = diag.get("prompt_selection_histogram", {})
        if isinstance(prompt, dict):
            vals.append(float(prompt.get("hard_entropy_normalized", 0.0)))
    return _mean(vals)


def _old_new_logit_gap(metrics: Dict[str, object]) -> float:
    gaps = []
    for stage in STAGES[1:]:
        payload = metrics.get(stage, {})
        if not isinstance(payload, dict):
            continue
        diag = payload.get("diagnostics", {})
        if not isinstance(diag, dict):
            continue
        logits = diag.get("logit_stats", {})
        if not isinstance(logits, dict):
            continue
        old = logits.get("old_tools", {})
        new = logits.get("new_tools", {})
        if isinstance(old, dict) and isinstance(new, dict):
            gaps.append(float(old.get("per_sample_max_logit_mean", 0.0)) - float(new.get("per_sample_max_logit_mean", 0.0)))
    return _mean(gaps)


def _final_forgetting(metrics: Dict[str, object]) -> Dict[str, float]:
    payload = metrics.get("task3", {})
    if not isinstance(payload, dict):
        return {key: 0.0 for key in METRIC_KEYS}
    summary = payload.get("continual_summary", {})
    if not isinstance(summary, dict):
        return {key: 0.0 for key in METRIC_KEYS}
    forgetting = summary.get("average_forgetting", {})
    if not isinstance(forgetting, dict):
        return {key: 0.0 for key in METRIC_KEYS}
    return {key: float(forgetting.get(key, 0.0)) for key in METRIC_KEYS}


def _final_replay(metrics: Dict[str, object]) -> tuple[int, float]:
    payload = metrics.get("task3", {})
    if not isinstance(payload, dict):
        return 0, 0.0
    replay = payload.get("replay_stats", {})
    if not isinstance(replay, dict):
        train = payload.get("train", {})
        replay = train.get("replay_stats_after", {}) if isinstance(train, dict) else {}
    if not isinstance(replay, dict):
        return 0, 0.0
    return int(replay.get("buffer_size", 0)), float(replay.get("tool_coverage_ratio", 0.0))


def summarize_version(name: str, metrics: Dict[str, object]) -> List[object]:
    current = _current_rows(metrics)
    old = _old_rows(metrics)
    final_old = _final_old_rows(metrics)
    forgetting = _final_forgetting(metrics)
    replay_size, replay_coverage = _final_replay(metrics)
    return [
        name,
        _avg_metric(current, "recall@1"),
        _avg_metric(current, "recall@5"),
        _avg_metric(current, "mrr"),
        _avg_metric(old, "recall@1"),
        _avg_metric(old, "recall@5"),
        _avg_metric(old, "mrr"),
        _avg_metric(final_old, "recall@1"),
        _avg_metric(final_old, "recall@5"),
        _avg_metric(final_old, "mrr"),
        forgetting["recall@1"],
        forgetting["recall@5"],
        forgetting["mrr"],
        _prompt_entropy(metrics),
        _old_new_logit_gap(metrics),
        replay_size,
        replay_coverage,
    ]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare L2P-Tool variants.")
    parser.add_argument("--no_replay_metrics", default="tool_l2p_runs_full/metrics.json")
    parser.add_argument("--regularized_metrics", default="tool_l2p_runs_regularized_no_replay/metrics.json")
    parser.add_argument("--replay_metrics", default="tool_l2p_runs_replay/metrics.json")
    args = parser.parse_args()

    rows = [
        summarize_version("L2P-Tool-NoReplay", _load(args.no_replay_metrics)),
        summarize_version("L2P-Tool-Regularized-NoReplay", _load(args.regularized_metrics)),
        summarize_version("L2P-Tool-Replay", _load(args.replay_metrics)),
    ]
    headers = [
        "Version",
        "CurAvg_R@1",
        "CurAvg_R@5",
        "CurAvg_MRR",
        "OldAvg_R@1",
        "OldAvg_R@5",
        "OldAvg_MRR",
        "FinalOld_R@1",
        "FinalOld_R@5",
        "FinalOld_MRR",
        "Forget_R@1",
        "Forget_R@5",
        "Forget_MRR",
        "PromptEntropy",
        "OldNewLogitGap",
        "ReplayBufferSize",
        "ReplayToolCoverage",
    ]
    print(", ".join(headers))
    for row in rows:
        print(", ".join(_fmt(item) for item in row))


if __name__ == "__main__":
    main()
