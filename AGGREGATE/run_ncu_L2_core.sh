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
LOG_FILE="benchmark_log/${GPU_ARCH}_${NUM_L2_ACCESS}Access_${L2_LOOPS}loop_results_L2_ncu.log"  # output log file path
NCU_RAW_LOG="benchmark_log/${GPU_ARCH}_${NUM_L2_ACCESS}Access_${L2_LOOPS}loop_ncu_raw.log"  # raw ncu output log
PLOT_FILE="benchmark_plots/${GPU_ARCH}_${NUM_L2_ACCESS}Access_${L2_LOOPS}loop_2d_colormap_L2_ncu.png"

# Range of CTAs and WARPs to sweep
CTAS=$(seq 1 32)
WARPS=$(seq 1 32)

echo "===== L2 Cache Experiments (loopCount=${L2_LOOPS}, sizeMultiple=${SIZE_L2}) ====="

# Initialize log files
> $LOG_FILE
> $NCU_RAW_LOG

# Run all ncu experiments and save to raw log
for cta in $CTAS; do
  for warp in $WARPS; do
    echo "L2 experiment: CTA: $cta, WARP: $warp"
    
    # Add experiment marker to raw log
    echo "===== CTA: $cta, WARP: $warp =====" >> "$NCU_RAW_LOG"
    
    # Run ncu and append output to raw log
    ncu --metrics lts__t_bytes.sum.per_second --target-processes all ./BW $cta $warp $ITER $L2_LOOPS $SIZE_L2 $NUM_L2_ACCESS >> "$NCU_RAW_LOG" 2>&1
    
    # Add separator to raw log
    echo "" >> "$NCU_RAW_LOG"
  done
done

echo "Raw ncu results written to $NCU_RAW_LOG"

# Parse bandwidth values from raw log file
echo "Parsing bandwidth values from raw log..."
current_cta=1
current_warp=1
bandwidth_count=0

while IFS= read -r line; do
  if [[ $line == *"===== CTA:"* ]]; then
    # Extract CTA and WARP from marker line
    current_cta=$(echo "$line" | sed 's/.*CTA: \([0-9]*\).*/\1/')
    current_warp=$(echo "$line" | sed 's/.*WARP: \([0-9]*\).*/\1/')
    bandwidth_count=0  # Reset counter for each experiment
  elif [[ $line == *"lts__t_bytes.sum.per_second"* ]]; then
    bandwidth_count=$((bandwidth_count + 1))
    
    # Only process the second occurrence (actual kernel, not warm-up)
    if [ $bandwidth_count -eq 2 ]; then
      # Extract unit (column 2) and value (column 3)
      BANDWIDTH_UNIT=$(echo "$line" | awk '{print $2}' | tr -d ' ')
      BANDWIDTH_VALUE=$(echo "$line" | awk '{print $3}' | tr -d ' ')
      
      if [ -n "$BANDWIDTH_VALUE" ] && [ "$BANDWIDTH_VALUE" != "N/A" ]; then
        # Check unit and convert accordingly
        if [[ "$BANDWIDTH_UNIT" == "Tbyte/second" ]]; then
          # Convert Tbyte/second to GB/second (multiply by 1000)
          BANDWIDTH_GB=$(awk "BEGIN {print $BANDWIDTH_VALUE * 1000}")
        elif [[ "$BANDWIDTH_UNIT" == "Gbyte/second" ]]; then
          # Already in GB/second, use as-is
          BANDWIDTH_GB="$BANDWIDTH_VALUE"
        else
          echo "Warning: Unknown unit $BANDWIDTH_UNIT for CTA=$current_cta, WARP=$current_warp"
          BANDWIDTH_GB="N/A"
        fi
        echo "$BANDWIDTH_GB" >> "$LOG_FILE"
      else
        echo "N/A" >> "$LOG_FILE"
      fi
      
      # Add newline after each CTA row is complete
      if [ "$current_warp" -eq 32 ]; then
        echo "" >> "$LOG_FILE"
      fi
    fi
  fi
done < "$NCU_RAW_LOG"

echo "L2 results written to $LOG_FILE"


# Plot the results using 2d colormap
python3 plot_2d_colormap.py $LOG_FILE -o $PLOT_FILE


