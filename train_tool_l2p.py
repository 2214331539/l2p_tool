import argparse
import os
import subprocess
import sys
from pathlib import Path


STAGES = ["base", "task1", "task2", "task3"]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run L2P-Tool training stages into one experiment directory.",
        add_help=True,
    )
    parser.add_argument("--stages", default="base,task1,task2,task3")
    parser.add_argument("--output_dir", default="tool_l2p_runs_regularized_no_replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Expected subset of {STAGES}.")

    run_dir = Path(args.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    split_path = run_dir / "splits" / f"full_seed{args.seed}.json"
    metrics_path = run_dir / "metrics.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    previous_ckpt: str | None = None
    for stage in stages:
        cmd = [
            args.python,
            "-m",
            "tool_l2p.train",
            "--stage",
            stage,
            "--seed",
            str(args.seed),
            "--output_dir",
            str(ckpt_dir),
            "--run_dir",
            str(run_dir),
            "--metrics_path",
            str(metrics_path),
            "--split_path",
            str(split_path),
            *passthrough,
        ]
        if stage != "base":
            previous_stage = STAGES[STAGES.index(stage) - 1]
            previous_ckpt = str(ckpt_dir / f"{previous_stage}.pt")
            cmd.extend(["--ckpt", previous_ckpt])
        print("[train_tool_l2p]", " ".join(cmd), flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
