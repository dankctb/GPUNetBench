#!/bin/bash
#
# It uses the following compile‑time definitions:
#   - ILP unrolling factor: -DILP_FACTOR
#   - Measurement mode: either -DCALC_LATENCY or -DCALC_BW
#   - Traffic pattern selection: one of the following:
#         - TRAFFIC_RANDPERM    : -DTRAFFIC_RANDPERM
#         - TRAFFIC_ROUNDROBIN  : -DTRAFFIC_ROUNDROBIN
#         - TRAFFIC_UNIFORM     : -DTRAFFIC_UNIFORM
#         - TRAFFIC_ALL2ALL     : -DTRAFFIC_ALL2ALL
#         - ALL2ALL full mode   : -DTRAFFIC_ALL2ALL -DREAD_ALL2ALL_FULL
#
# The executable now expects two runtime parameters:
#   <numClusters> <blockSize>
#
# Fixed parameters:
#   - CLUSTER_SIZE = 16   (used for partner map, if needed)
#   - NUM_CLUSTERS = 1
#   - ILP_FACTOR   = 8
#
# The script loops over a set of block sizes:
#   BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)
#
# Output files are named using the measurement mode, traffic pattern,
# block size, number of clusters, and ILP factor:
#   <mode>_<traffic>_<blockSize>_<numClusters>_<ILP_FACTOR>.log
#

# Fixed parameters
CLUSTER_SIZE=16
NUM_CLUSTERS=1
ILP_FACTOR=8
BLOCK_SIZES=(32 64 128 256 512 1024)

# Specify the traffic patterns to run.
# Valid options: "randperm", "roundrobin", "uniform", "all2all", "all2all_full"
TRAFFIC_PATTERNS=("randperm" "roundrobin" "uniform" "all2all" "all2all_full")

# Loop over each traffic pattern.
for traffic in "${TRAFFIC_PATTERNS[@]}"; do
    # Set traffic pattern compile definitions.
    if [ "$traffic" == "randperm" ]; then
        TRAFFIC_FLAG="-DTRAFFIC_RANDPERM"
    elif [ "$traffic" == "roundrobin" ]; then
        TRAFFIC_FLAG="-DTRAFFIC_ROUNDROBIN"
    elif [ "$traffic" == "uniform" ]; then
        TRAFFIC_FLAG="-DTRAFFIC_UNIFORM"
    elif [ "$traffic" == "all2all" ]; then
        TRAFFIC_FLAG="-DTRAFFIC_ALL2ALL"
    elif [ "$traffic" == "all2all_full" ]; then
        TRAFFIC_FLAG="-DTRAFFIC_ALL2ALL -DREAD_ALL2ALL_FULL"
    else
        echo "Unknown traffic pattern: $traffic"
        exit 1
    fi

    # Loop over the two measurement modes: latency and bandwidth.
    for mode in latency bw; do
        if [ "$mode" == "latency" ]; then
           MODE_FLAG="-DCALC_LATENCY"
           MODE_NAME="Latency"
        else
           MODE_FLAG="-DCALC_BW"
           MODE_NAME="Bandwidth"
        fi

        # Loop over the specified block sizes.
        for bs in "${BLOCK_SIZES[@]}"; do
            echo "Compiling with:"
            echo "  ILP_FACTOR = ${ILP_FACTOR}"
            echo "  BLOCK_SIZE = ${bs}"
            echo "  Mode       = ${MODE_NAME}"
            echo "  Traffic    = ${traffic}"
            echo

            # Set compile definitions via NVCC_DEFS.
            export NVCC_DEFS="-DILP_FACTOR=${ILP_FACTOR} ${MODE_FLAG} ${TRAFFIC_FLAG}"

            # Clean and compile the target.
            make clean
            make

            # Create an output file name that identifies the configuration.
            outfile="${mode}_${traffic}_${bs}_${NUM_CLUSTERS}_${ILP_FACTOR}.log"
            rm -f "$outfile"

            echo "${MODE_NAME} Mode: Running with NUM_CLUSTERS=${NUM_CLUSTERS} and BLOCK_SIZE=${bs}" | tee -a "$outfile"
            # Run the benchmark with runtime parameters: numClusters and blockSize.
            ./SM2SM ${NUM_CLUSTERS} ${bs} >> "$outfile"

            echo "${MODE_NAME} test complete for traffic pattern '${traffic}' with block size ${bs}. Results in ${outfile}"
            echo
        done
    done
done

echo "All SM‑to‑SM tests finished."
