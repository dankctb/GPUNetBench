#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building matmul H100 program..."
make clean
make matmul_h100

GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)

if [[ "$GPU_INFO" != *"H100"* ]]; then
  echo "No Hopper GPU Architecture detected" >&2
  exit 1
fi

# Fixed input size, vary cluster sizes
N=${1:-4096}
clusters=(0 1 2 4 8 16)

OUT_CSV="results_matmul_h100_N${N}.csv"
echo "N,cluster_size,latency_ms" > "$OUT_CSV"

for c in "${clusters[@]}"; do
  ./matmul_h100 $N $c | tee -a "$OUT_CSV"
done

echo "Wrote $OUT_CSV"

# Plot bar chart using python + matplotlib if available
CSV_PATH="$OUT_CSV" python3 - <<'PY' || true
import os, csv
csv_path = os.environ.get('CSV_PATH')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('Skipping plot (matplotlib not available):', e)
    raise SystemExit(0)
xs, ys = [], []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(int(row['cluster_size']))
        ys.append(float(row['latency_ms']))
import os
import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
plt.bar([str(x) for x in xs], ys)
plt.xlabel('cluster_size')
plt.ylabel('latency (ms)')
plt.title('MatMul H100 DSMEM sweep')
plt.tight_layout()
img = os.path.splitext(csv_path)[0] + '.png'
plt.savefig(img, dpi=150)
print('Saved', img)
PY
