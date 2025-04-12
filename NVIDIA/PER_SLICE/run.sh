#!/bin/bash
# run.sh - Runs all possible SM and L2 slice combinations in both Direct SM and GPC modes for V100 as example.
# Each execution is either profiled with nvprof, profiled with ncu, or run normally.
#
# Requirements:
#   - This script assumes a makefile that accepts:
#         ARCH=<v100|a100|h100> and MODE=<direct|gpc>
#     where in direct mode the usage is:
#         ./SLICE <SMid> <slice_index>
#     and in GPC mode the usage is:
#         ./SLICE <GPCselectedList> <SMmax> <slice_index>
#
#   - For profiling:
#         nvprof uses:
#           --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput
#
#         ncu uses:
#           --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_l1tex2xbar_write_bytes.sum.per_second
#
# Adjust NUM_DIRECT_SMS and NUM_SLICES as necessary for the chosen arch.
#
# Usage: ./run.sh [nvprof|ncu|normal]

set -e  # Exit immediately if a command exits with a non-zero status.

# Verify that the user has provided an argument.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [nvprof|ncu|normal]"
    exit 1
fi

profile_mode=$1
if [[ "$profile_mode" != "nvprof" && "$profile_mode" != "ncu" && "$profile_mode" != "normal" ]]; then
    echo "Error: Unknown profiling mode '$profile_mode'. Choose one of: nvprof, ncu, normal."
    exit 1
fi

# ---------------------------------------------------------
# DIRECT SM MODE (Single SM selection)
# ---------------------------------------------------------
ARCH=v100
NUM_DIRECT_SMS=80
NUM_SLICES=32

echo "=============================================="
echo "Direct SM Mode ($profile_mode) for V100"
echo "=============================================="

# Clean and compile in direct SM mode.
make clean
make ARCH=$ARCH MODE=direct

# Loop over every SM id and every L2 slice.
for (( slice=0; slice<NUM_SLICES; slice++ )); do
    for (( sm=0; sm<NUM_DIRECT_SMS; sm++ )); do
        if [ "$profile_mode" = "nvprof" ]; then
            logFileName="${ARCH}_direct_SM${sm}_slice${slice}.log"
            nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
                --log-file "$logFileName" \
                ./SLICE ${sm} ${slice}
        elif [ "$profile_mode" = "ncu" ]; then
            logFileName="${ARCH}_direct_SM${sm}_slice${slice}.log"
            ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_l1tex2xbar_write_bytes.sum.per_second \
                --log-file "$logFileName" \
                ./SLICE ${sm} ${slice}
        elif [ "$profile_mode" = "normal" ]; then
            # For normal mode, accumulate all output in one log file.
            logFileName="${ARCH}.log"
            ./SLICE ${sm} ${slice} >> "$logFileName"
        fi
    done
    # For normal mode only, add a newline after each slice iteration.
    if [ "$profile_mode" = "normal" ]; then
         echo "" >> "$logFileName"
    fi
done

# ---------------------------------------------------------
# GPC MODE (GPC mapping selection)
# ---------------------------------------------------------
echo "=============================================="
echo "GPC Mode ($profile_mode) for V100"
echo "=============================================="

# Clean and compile in GPC mode.
make clean
make ARCH=$ARCH MODE=gpc

# In GPC mode for V100, there are 6 possible GPC indices (0..5).
for (( slice=0; slice<NUM_SLICES; slice++ )); do
    for gpc in 0 1 2 3 4 5; do
        if [ $gpc -lt 4 ]; then
            maxSMIndex=14
        else
            maxSMIndex=12
        fi

        if [ "$profile_mode" = "nvprof" ]; then
            logFileName="${ARCH}_gpc_GPC${gpc}_slice${slice}.log"
            nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
                --log-file "$logFileName" \
                ./SLICE ${gpc} ${maxSMIndex} ${slice}
        elif [ "$profile_mode" = "ncu" ]; then
            logFileName="${ARCH}_gpc_GPC${gpc}_slice${slice}.log"
            ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_l1tex2xbar_write_bytes.sum.per_second \
                --log-file "$logFileName" \
                ./SLICE ${gpc} ${maxSMIndex} ${slice}
        elif [ "$profile_mode" = "normal" ]; then
            logFileName="${ARCH}_gpc.log"
            ./SLICE ${gpc} ${maxSMIndex} ${slice} >> "$logFileName"
        fi
    done
    if [ "$profile_mode" = "normal" ]; then
         echo "" >> "$logFileName"
    fi
done

echo "=============================================="
echo "All runs complete."
