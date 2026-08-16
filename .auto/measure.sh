#!/usr/bin/env bash
set -euo pipefail
# OrbitQuant autoresearch benchmark: honest rate-distortion + estimator quality.
# Runs the benchmark 3x and reports the median per metric (smooths fwht_us).
cd "$(dirname "$0")/.."

exec "$PWD/.venv/bin/python" - <<'PY'
import collections
import re
import statistics
import subprocess
import sys

lines = []
for _ in range(3):
    out = subprocess.run(
        [sys.executable, "-m", "baseline.benchmark_tq"],
        capture_output=True, text=True,
    ).stdout
    lines.append([(m.group(1), float(m.group(2)))
                  for m in re.finditer(r"^METRIC (\w+)=([0-9.eE+-]+)$", out, re.M)])

by = collections.defaultdict(list)
for run in lines:
    for k, v in run:
        by[k].append(v)

order = ["tq_mse", "tq_bytes_b2", "tq_bias_b1", "tq_bias_raw_b1", "tq_bias_raw", "tq_var_b2", "tq_attn_kl_b2", "tq_attn_kl_b2_p95", "fwht_us"]
for k in order:
    med = statistics.median(sorted(by[k]))
    print(f"METRIC {k}={med:.6g}")
PY
