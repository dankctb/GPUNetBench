#!/bin/bash
#
# Script: parse_logs.sh
#
# Description:
#   Processes two sets of log files:
#
#   1. Files named as (<arch>)_gpc_GPC<num>_slice<num>.log:
#      - Greps for the line containing:
#            l1tex__m_xbar2l1tex_read_bytes.sum.per_second
#        and extracts the last numeric value on that line.
#      - If not found, it greps for the line containing:
#            l2_read_throughput
#        and extracts the first numeric value from that line.
#      - Each numeric value is written (one per file) to an output file named <arch>_gpc.log.
#
#   2. Files named as ${ARCH}_direct_SM${sm}_slice${slice}.log:
#      - Performs the same extraction as above.
#      - Each numeric value is written to an output file named <arch>.log.
#
# Usage:
#       ./parse_logs.sh
#

# Function to extract a numeric value from the given file.
extract_value() {
    local file="$1"
    local value

    # Try the primary metric line first.
    value=$(grep -m 1 "l1tex__m_xbar2l1tex_read_bytes.sum.per_second" "$file" | awk '{print $NF}')

    # If no value found, try the fallback metric line.
    if [ -z "$value" ]; then
        local fallback_line
        fallback_line=$(grep -m 1 "l2_read_throughput" "$file")
        if [ -n "$fallback_line" ]; then
            # Extract only one numeric value (the first found) from the fallback line.
            value=$(echo "$fallback_line" | grep -oE "[0-9]+(\.[0-9]+)?" | head -n 1)
        fi
    fi

    echo "$value"
}

# Process the first group of log files: (<arch>)_gpc_GPC<num>_slice<num>.log
# First, clear/create the output files for each distinct arch.
for file in *_gpc_GPC*_slice*.log; do
    [ -e "$file" ] || continue
    arch="${file%%_*}"
    output_file="${arch}_gpc.log"
    > "$output_file"
done

# Loop over the first group and extract values.
for file in *_gpc_GPC*_slice*.log; do
    if [ -f "$file" ]; then
        arch="${file%%_*}"
        output_file="${arch}_gpc.log"
        value=$(extract_value "$file")
        if [ -n "$value" ]; then
            echo "$value" >> "$output_file"
        else
            echo "Warning: Value not found in $file" >&2
        fi
    fi
done

# Process the second group of log files: ${ARCH}_direct_SM${sm}_slice${slice}.log
# Clear/create the output file for each distinct arch.
for file in *_direct_SM*_slice*.log; do
    [ -e "$file" ] || continue
    arch="${file%%_*}"
    output_file="${arch}.log"
    > "$output_file"
done

# Loop over the second group and extract values.
for file in *_direct_SM*_slice*.log; do
    if [ -f "$file" ]; then
        arch="${file%%_*}"
        output_file="${arch}.log"
        value=$(extract_value "$file")
        if [ -n "$value" ]; then
            echo "$value" >> "$output_file"
        else
            echo "Warning: Value not found in $file" >&2
        fi
    fi
done

echo "Log parsing complete."
