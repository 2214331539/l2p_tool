#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"

"$PY" -m tool_l2p.compare \
  --no_replay_metrics tool_l2p_runs_replay/metrics.json \
  --regularized_metrics tool_l2p_runs_replay_ep3/metrics.json \
  --replay_metrics tool_l2p_runs_replay_ep5/metrics.json

