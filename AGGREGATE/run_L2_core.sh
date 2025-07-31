#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_L2_core.sh [GPU_ARCH]
# Get GPU architecture from command line argument (default: a100)
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <v100|a100|h100>"
    echo "Please specify the GPU architecture"
    exit 1
fi

GPU_ARCH=$1

echo "Using GPU architecture: $GPU_ARCH"

# Build the executable with parameters: ARCH=v100, a100, h100
make ARCH=$GPU_ARCH

# Create output directories
mkdir -p benchmark_log benchmark_plots

# Common parameters
ITER=1                    # number of measurement iterations
L2_LOOPS=1000             # inner loops for L2-cache stress
SIZE_L2=1                 # sizeMultiple=1 for L2
NUM_L2_ACCESS=$((120*32*32*32))   # number of L2 accesses (80 * 32^3 = 2621440)
LOG_FILE="benchmark_log/${GPU_ARCH}_${NUM_L2_ACCESS}Access_${L2_LOOPS}loop_results_L2.log"  # output log file path
PLOT_FILE="benchmark_plots/${GPU_ARCH}_${NUM_L2_ACCESS}Access_${L2_LOOPS}loop_2d_colormap_L2.png"

# Range of CTAs and WARPs to sweep
CTAS=$(seq 1 32)
WARPS=$(seq 1 32)

nvidia-smi -pm 1
nvidia-smi -ac 877,1380

echo "===== L2 Cache Experiments (loopCount=${L2_LOOPS}, sizeMultiple=${SIZE_L2}) ====="
> $LOG_FILE
for cta in $CTAS; do
  for warp in $WARPS; do
    echo "L2 experiment: CTA: $cta, WARP: $warp"
    ./BW $cta $warp $ITER $L2_LOOPS $SIZE_L2 $NUM_L2_ACCESS >> "$LOG_FILE"
  done
  echo "" >> "$LOG_FILE"
done
echo "L2 results written to $LOG_FILE"

# Plot the results using 2d colormap
python3 plot_2d_colormap.py $LOG_FILE -o $PLOT_FILE


