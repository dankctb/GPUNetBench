#!/bin/bash
#
# The code uses the following compile‑time definitions:
#   - ILP unrolling factor is set via: -DILP_FACTOR
#   - Measurement mode: either -DCALC_LATENCY or -DCALC_BW
#
# The executable expects four runtime parameters:
#   <rt_destSM> <rt_srcSM> <numClusters> <blockSize>
#
# In this script we use:
#   - CLUSTER_SIZE = 16   (ranks per cluster)
#   - NUM_CLUSTERS = 1
#   - BLOCK_SIZE   = 1024
#   - ILP_FACTOR   = 8
#
# Output files are named using the mode, block size, number of clusters, and ILP factor:
#   <mode>_<blockSize>_<numClusters>_<ILP_FACTOR>.log
#
# The script loops over all source/destination rank combinations
# (excluding cases where src == dest).
#

# Fixed parameters
CLUSTER_SIZE=16
NUM_CLUSTERS=1
BLOCK_SIZE=1024
ILP_FACTOR=8

# Loop over the two measurement modes: latency and bandwidth.
for mode in latency bw; do
    if [ "$mode" == "latency" ]; then
       MODE_FLAG="-DCALC_LATENCY"
       MODE_NAME="Latency"
    else
       MODE_FLAG="-DCALC_BW"
       MODE_NAME="Bandwidth"
    fi

    echo "Compiling with:"
    echo "  ILP_FACTOR = ${ILP_FACTOR}"
    echo "  BLOCK_SIZE = ${BLOCK_SIZE}"
    echo "  Mode       = ${MODE_NAME}"
    echo

    # Set compile definitions via NVCC_DEFS:
    export NVCC_DEFS="-DILP_FACTOR=${ILP_FACTOR} ${MODE_FLAG}"

    # Clean and compile the target.
    make clean
    make

    # Create an output file name that identifies the configuration.
    outfile="${mode}_${BLOCK_SIZE}_${NUM_CLUSTERS}_${ILP_FACTOR}.log"
    rm -f "$outfile"

    # Run the benchmark for all src/dest combinations, skipping src == dest.
    for dest in $(seq 0 $((CLUSTER_SIZE - 1))); do
        for src in $(seq 0 $((CLUSTER_SIZE - 1))); do
            if [ "$dest" -eq "$src" ]; then
                continue
            fi
            echo "${MODE_NAME} Mode: rt_destSM=${dest}, rt_srcSM=${src}" | tee -a "$outfile"
            ./SM2SM ${dest} ${src} ${NUM_CLUSTERS} ${BLOCK_SIZE} >> "$outfile"
        done
    done

    echo "${MODE_NAME} tests complete. Results in ${outfile}"
    echo
done

echo "All SM‑to‑SM tests finished."
