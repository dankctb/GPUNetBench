#!/usr/bin/env bash
set -euo pipefail

# Build the executable
make

# Create output directories
mkdir -p benchmark_log benchmark_plots

# Common parameters
ITER=1                    # number of measurement iterations
L2_LOOPS=1000             # inner loops for L2-cache stress
SIZE_L2=1                 # sizeMultiple=1 for L2
SIZE_HBM=1000

# Range of CTAs and WARPs to sweep
CTAS=$(seq 1 32)
WARPS=$(seq 1 32)

echo "===== L2 Cache Experiments (loopCount=${L2_LOOPS}, sizeMultiple=${SIZE_L2}) ====="
> benchmark_log/results_L2.log
for cta in $CTAS; do
  for warp in $WARPS; do
    echo "L2 experiment: CTA: $cta, WARP: $warp"
    ./BW $cta $warp $ITER $L2_LOOPS $SIZE_L2 >> benchmark_log/results_L2.log
  done
  echo "" >> benchmark_log/results_L2.log
done
echo "L2 results written to benchmark_log/results_L2.log"

python3 plot_2d_colormap.py benchmark_log/results_L2.log -o benchmark_plots/fixed_bw_2d_colormap_L2.png
python3 plot_per_CTA.py results_L2.log -s L2
python3 plot_per_warp.py results_L2.log -s L2


echo
echo "===== HBM Experiments (loopCount=1, sizeMultiple=${SIZE_HBM}) ====="
> benchmark_log/results_HBM.log
for cta in $CTAS; do
  for warp in $WARPS; do
    echo "HBM experiment: CTA: $cta, WARP: $warp"
    ./BW $cta $warp $ITER 1 $SIZE_HBM >> benchmark_log/results_HBM.log
  done
  echo "" >> benchmark_log/results_HBM.log
done
echo "HBM results written to benchmark_log/results_HBM.log"


python3 plot_2d_colormap.py benchmark_log/results_HBM.log -o benchmark_plots/fixed_bw_2d_colormap_HBM.png
python3 plot_per_CTA.py results_HBM.log -s HBM
python3 plot_per_warp.py results_HBM.log -s HBM