import argparse
import json
from typing import Dict

from .utils import STAGES


def _fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.2f}"
    return str(value)


def print_table(title: str, rows: list[list[object]], headers: list[str]) -> None:
    print(f"\n{title}")
    print(", ".join(headers))
    for row in rows:
        print(", ".join(_fmt(x) for x in row))


def _mean_current_rows(rows: list[list[object]]) -> list[list[object]]:
    if not rows:
        return []
    metric_cols = [3, 4, 5, 6, 7]
    means = []
    for col in metric_cols:
        values = [float(row[col]) for row in rows]
        means.append(sum(values) / len(values))
    return [["avg_current", len(rows), *means]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize L2P-Tool metrics.json.")
    parser.add_argument("--metrics_path", default="tool_l2p_runs_full/metrics.json")
    args = parser.parse_args()

    with open(args.metrics_path, "r", encoding="utf-8") as f:
        metrics: Dict[str, object] = json.load(f)

    current_rows = []
    old_rows = []
    summary_rows = []
    prompt_rows = []
    logit_rows = []
    for stage in STAGES:
        payload = metrics.get(stage)
        if not isinstance(payload, dict):
            continue
        evals = payload.get("eval", {})
        if not isinstance(evals, dict):
            continue
        current = evals.get(f"{stage}_global", {})
        if isinstance(current, dict):
            current_rows.append(
                [
                    stage,
                    current.get("num_candidates", 0),
                    current.get("num_eval_samples", 0),
                    current.get("recall@1", 0.0),
                    current.get("recall@3", 0.0),
                    current.get("recall@5", 0.0),
                    current.get("ndcg@5", 0.0),
                    current.get("mrr", 0.0),
                ]
            )
        for key, value in evals.items():
            if not key.endswith("_global") or key == f"{stage}_global" or not isinstance(value, dict):
                continue
            old_rows.append(
                [
                    stage,
                    key,
                    value.get("num_candidates", 0),
                    value.get("num_eval_samples", 0),
                    value.get("recall@1", 0.0),
                    value.get("recall@3", 0.0),
                    value.get("recall@5", 0.0),
                    value.get("mrr", 0.0),
                ]
            )
        summary = payload.get("continual_summary", {})
        if isinstance(summary, dict):
            old_avg = summary.get("old_task_global_average", {})
            forgetting = summary.get("average_forgetting", {})
            if isinstance(old_avg, dict) and isinstance(forgetting, dict):
                summary_rows.append(
                    [
                        stage,
                        old_avg.get("recall@1", 0.0),
                        old_avg.get("recall@3", 0.0),
                        old_avg.get("recall@5", 0.0),
                        old_avg.get("mrr", 0.0),
                        forgetting.get("recall@1", 0.0),
                        forgetting.get("recall@3", 0.0),
                        forgetting.get("recall@5", 0.0),
                        forgetting.get("mrr", 0.0),
                    ]
                )
        diagnostics = payload.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            prompt = diagnostics.get("prompt_selection_histogram", {})
            if isinstance(prompt, dict):
                prompt_rows.append(
                    [
                        stage,
                        prompt.get("nonzero_prompts", 0),
                        prompt.get("hard_entropy_normalized", 0.0),
                        prompt.get("soft_entropy_normalized", 0.0),
                        prompt.get("top_selected_prompt_ids", []),
                    ]
                )
            logits = diagnostics.get("logit_stats", {})
            if isinstance(logits, dict):
                old_logits = logits.get("old_tools", {})
                new_logits = logits.get("new_tools", {})
                if isinstance(old_logits, dict) and isinstance(new_logits, dict):
                    logit_rows.append(
                        [
                            stage,
                            old_logits.get("per_sample_max_logit_mean", 0.0),
                            new_logits.get("per_sample_max_logit_mean", 0.0),
                            old_logits.get("logit_mean", 0.0),
                            new_logits.get("logit_mean", 0.0),
                        ]
                    )

    print("metrics_path", args.metrics_path)
    print_table(
        "CURRENT TASK GLOBAL",
        current_rows,
        ["stage", "candidates", "samples", "R@1", "R@3", "R@5", "NDCG@5", "MRR"],
    )
    print_table(
        "AVERAGE CURRENT TASK GLOBAL",
        _mean_current_rows(current_rows),
        ["name", "num_stages", "R@1", "R@3", "R@5", "NDCG@5", "MRR"],
    )
    print_table(
        "OLD TASK GLOBAL",
        old_rows,
        ["checkpoint", "eval_key", "candidates", "samples", "R@1", "R@3", "R@5", "MRR"],
    )
    print_table(
        "CONTINUAL SUMMARY",
        summary_rows,
        ["stage", "old_avg_R@1", "old_avg_R@3", "old_avg_R@5", "old_avg_MRR", "forget_R@1", "forget_R@3", "forget_R@5", "forget_MRR"],
    )
    print_table(
        "PROMPT USAGE",
        prompt_rows,
        ["stage", "nonzero", "hard_entropy_norm", "soft_entropy_norm", "top_prompt_ids"],
    )
    print_table(
        "OLD/NEW LOGITS",
        logit_rows,
        ["stage", "old_max_mean", "new_max_mean", "old_mean", "new_mean"],
    )


if __name__ == "__main__":
    main()
