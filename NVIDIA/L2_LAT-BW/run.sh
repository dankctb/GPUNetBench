#!/bin/bash
# run.sh - Runs a series of executions using a selectable access pattern
# and optional random delay, with varying threads per CTA. Uses 2 CTAs per SM.
#
# Command-line arguments:
#   $1: run_mode                (ncu or normal)
#   $2: access_pattern          (stream, strided, random) [default: stream]
#   $3: random_delay            (0 or 1) [default: 0]
#   $4: random_delay_method     (thread or warp) [default: thread]
#   $5: random_delay_steps      (integer) [default: 32]
#
# When using profiling, only ncu is used.
#
# Examples:
#   ./run.sh normal stream 0
#   ./run.sh ncu random 1 warp 64

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <ncu|normal> [access_pattern: stream|strided|random] [random_delay: 0|1] [random_delay_method: thread|warp] [random_delay_steps]"
    exit 1
fi

run_mode=$1
access_pattern=${2:-stream}
random_delay=${3:-0}
random_delay_method=${4:-thread}
random_delay_steps=${5:-32}

# Build the NVCC_DEFS variable.
COMMON_FLAGS=""

# Set access pattern flags.
if [ "$access_pattern" == "stream" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} -DUSE_STREAM_ACCESS"
elif [ "$access_pattern" == "strided" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} -DUSE_STRIDED_ACCESS"
elif [ "$access_pattern" == "random" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} -DUSE_RANDOM_ACCESS"
else
    echo "Error: Unknown access pattern '$access_pattern'. Choose stream, strided, or random."
    exit 1
fi

# Set random delay flags.
if [ "$random_delay" -eq 1 ]; then
    COMMON_FLAGS="${COMMON_FLAGS} -DENABLE_RANDOM_DELAY -DRANDOM_DELAY_STEPS=${random_delay_steps}"
    if [ "$random_delay_method" == "warp" ]; then
        COMMON_FLAGS="${COMMON_FLAGS} -DWARP_RANDOM_DELAY"
    fi
fi

# Enable latency measurement only for normal execution.
if [ "$run_mode" = "normal" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} -DENABLE_LATENCY_MEASUREMENT"
fi

export NVCC_DEFS="${COMMON_FLAGS}"

# Set architecture.
ARCH=v100

# Clean and build.
make clean
make ARCH=${ARCH}

# Set output mode.
OUTPUT_MODE=b

CTAs_per_SM=2

#THREADS_LIST=(1 32 64 512 1024)
THREADS_LIST=(1 32 64 96 128 160 192 224 256 288 320 384 448 512 576 640 704 768 832 896 960 1024)


echo "===================================================================================="
echo "Bandwidth measurement, Latency distribution and average latency measurement"
echo "===================================================================================="


for threads in "${THREADS_LIST[@]}"; do
    echo "Running with threads per CTA: $threads, CTAs per SM: ${CTAs_per_SM}"
    
    # Build log file name.
    logFileName="ncu_${threads}_${CTAs_per_SM}_${access_pattern}"
    if [ "$random_delay" -eq 1 ]; then
         logFileName="${logFileName}_rand_delay_${random_delay_method}_${random_delay_steps}"
    fi
    logFileName="${logFileName}.log"
    
    if [ "$run_mode" = "ncu" ]; then
         ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_l1tex2xbar_write_bytes.sum.per_second,gpu__time_duration.avg \
             --log-file "$logFileName" \
             ./L2_LAT-BW -t ${threads} -c ${CTAs_per_SM} -o ${OUTPUT_MODE}
    elif [ "$run_mode" = "normal" ]; then
         ./L2_LAT-BW -t ${threads} -c ${CTAs_per_SM} -o ${OUTPUT_MODE}
    fi
done

echo "=============================================="
echo "All runs complete."

