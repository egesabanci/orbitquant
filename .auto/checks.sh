#!/usr/bin/env bash
set -euo pipefail
# Correctness/backpressure gate. Deterministic. Runs after every passing bench.
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"

# 1) byte-compile every baseline module (catches syntax errors instantly)
"$PY" -m py_compile baseline/*.py

# 2) benchmark invariants (FWHT identities, orthogonality, metric ranges, 136-bit check)
"$PY" -m baseline.benchmark_tq --selfcheck

# 3) full statistical theorem checks (seeded => deterministic, exits nonzero on fail)
"$PY" -m baseline.run_baseline --d 64 --nrot 2000 >/dev/null 2>&1
