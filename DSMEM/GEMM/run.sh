#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building matmul programs..."
make clean
make all

# Vary matrix sizes and cluster sizes
sizes=(512 1024 2048)
cluster_sizes=(2 4 8)
OUT_CSV="results_matmul_dsm_comparison.csv"
echo "N,cluster_size,regular_ms,dsm_optimized_ms" > "$OUT_CSV"

nvidia-smi -pm 1 # Enable persistent mode
# run this to find the supported clock combinations
# nvidia-smi --query-supported-clocks=mem,gr --format=csv
# nvidia-smi -ac 1410,1830 # h100 does not support this combination

for N in "${sizes[@]}"; do
  for C in "${cluster_sizes[@]}"; do
    echo -n "$N,$C,"
    # Regular baseline (original implementation)
    ./matmul "$N" | cut -d',' -f2 | tr -d '\n'
    echo -n ","
    # DSM optimized with specified cluster size
    ./matmul_dsm "$N" "$C"
  done
done >> "$OUT_CSV"

echo "Wrote $OUT_CSV"

# Plot comparison if matplotlib available
CSV_PATH="$OUT_CSV" python3 - <<'PY' || true
import os, csv
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    print('Skipping plot (matplotlib not available):', e)
    raise SystemExit(0)
data = {}
with open(os.environ['CSV_PATH']) as f:
    r = csv.DictReader(f)
    for row in r:
        n = int(row['N'])
        c = int(row['cluster_size'])
        if n not in data: data[n] = {}
        data[n][c] = {'reg': float(row['regular_ms']), 'dsm': float(row['dsm_optimized_ms'])}
Ns = sorted(data.keys())
clusters = sorted(data[Ns[0]].keys())
x = np.arange(len(Ns))
width = 0.25
plt.figure(figsize=(12,6))
for i, c in enumerate(clusters):
    dsm_times = [data[n][c]['dsm'] for n in Ns]
    plt.bar(x + i*width, dsm_times, width, label=f'DSM cluster={c}')
plt.xticks(x + width, [str(n) for n in Ns])
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (ms)')
plt.title('DSM-Optimized Matrix Multiplication Performance by Cluster Size')
plt.legend()
plt.tight_layout()
img = os.path.splitext(os.environ['CSV_PATH'])[0] + '.png'
plt.savefig(img, dpi=150)
print('Saved', img)
PY
