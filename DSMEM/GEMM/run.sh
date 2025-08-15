#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building matmul programs..."
make clean
make all

# Vary matrix sizes; fixed cluster=8 and block=16 sizes are in code
sizes=(512 1024 2048 4096)

OUT_CSV="results_matmul_comparison.csv"
echo "N,regular_ms,dsmem_ms" > "$OUT_CSV"

for N in "${sizes[@]}"; do
  echo -n "$N,"
  ./matmul "$N" | cut -d',' -f2 | tr -d '\n'
  echo -n ","
  ./matmul_h100 "$N" | cut -d',' -f2
done >> "$OUT_CSV"

echo "Wrote $OUT_CSV"

# Plot comparison if matplotlib available
CSV_PATH="$OUT_CSV" python3 - <<'PY' || true
import os, csv
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('Skipping plot (matplotlib not available):', e)
    raise SystemExit(0)
Ns, reg, dsm = [], [], []
with open(os.environ['CSV_PATH']) as f:
    r = csv.DictReader(f)
    for row in r:
        Ns.append(int(row['N']))
        reg.append(float(row['regular_ms']))
        dsm.append(float(row['dsmem_ms']))
import numpy as np
x = np.arange(len(Ns))
width = 0.35
plt.figure(figsize=(8,4))
plt.bar(x - width/2, reg, width, label='regular')
plt.bar(x + width/2, dsm, width, label='dsmem cluster=8')
plt.xticks(x, [str(n) for n in Ns])
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (ms)')
plt.title('MatMul: Regular vs DSMEM Performance')
plt.legend()
plt.tight_layout()
img = os.path.splitext(os.environ['CSV_PATH'])[0] + '.png'
plt.savefig(img, dpi=150)
print('Saved', img)
PY
