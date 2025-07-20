#!/bin/bash
#
# parse_logs.sh
#
# Description:
#   Processes files with names formatted as:
#
#       ncu_<threads_per_CTA>_<CTAs_per_SM>_<accesspattern>[_rand_delay_<method>_<steps>].log
#
#   For each file (sorted in ascending order by threads_per_CTA), two metrics are extracted:
#
#   1. From the **second** line containing:
#          l1tex__m_xbar2l1tex_read_bytes.sum.per_second
#      the script extracts the last numeric value. It then checks if this line contains
#      "GByte/s". If not – if it contains "TByte/s" – it converts the value to GByte/s.
#
#   2. From the **second** line containing:
#          gpu__time_duration.avg
#      the script extracts the last numeric value. It then checks if this line already has the unit:
#         - If "us" is present, no conversion is done.
#         - If "ms" is present, the value is converted to µs (multiplied by 1e3).
#         - If "s" is present (and not "ms" or "us"), it is converted to µs (multiplied by 1e6).
#
#   The converted results are written into two output files:
#
#       BW_<accesspattern>_<CTAs_per_SM>[_rand_delay_<method>_<steps>].log
#       KET_<accesspattern>_<CTAs_per_SM>[_rand_delay_<method>_<steps>].log
#
#   Each output file is built so that each line (ordered by increasing threads_per_CTA)
#   contains the converted value.
#
# Usage:
#       ./parse_logs.sh
#

# Function to extract the last numeric value from a string.
extract_last_number() {
    grep -oE "[0-9]+(\.[0-9]+)?" <<< "$1" | tail -n 1
}

# Make sure we have at least one .log to work on
shopt -s nullglob
all_logs=( ncu_*_*.log )
if [ ${#all_logs[@]} -eq 0 ]; then
    echo "No ncu log files found."
    exit 1
fi

# Build a list of unique "<CTAs>_<accesspattern>[_rand_delay_…]" suffixes
# by stripping off "ncu_<threads>_" and ".log"
mapfile -t pattern_suffixes < <(
    for f in "${all_logs[@]}"; do
        basename="$f"
        # remove prefix up to second underscore, then strip .log
        suffix="${basename#ncu_[0-9]*_}"
        suffix="${suffix%.log}"
        echo "$suffix"
    done | sort -u
)

# Now process each pattern in turn
for suffix in "${pattern_suffixes[@]}"; do
    # split suffix into CTAs, accesspattern, and optional rand_delay tail
    CTAs="${suffix%%_*}"
    rest="${suffix#${CTAs}_}"
    accesspattern="${rest%%_*}"
    rand_extra="${rest#${accesspattern}_}"
    if [ "$rand_extra" != "$rest" ]; then
        rand_suffix="_${rand_extra}"
    else
        rand_suffix=""
    fi

    bw_output="BW_${accesspattern}_${CTAs}${rand_suffix}.log"
    ket_output="KET_${accesspattern}_${CTAs}${rand_suffix}.log"

    # start fresh
    > "$bw_output"
    > "$ket_output"

    echo "=== Processing pattern: CTAs=${CTAs}, accesspattern=${accesspattern}${rand_suffix} ==="

    # find and sort matching files by threads_per_CTA (the 2nd field)
    sorted_files=( $(ls ncu_*_"${suffix}".log 2>/dev/null | sort -t '_' -k2,2n) )
    if [ ${#sorted_files[@]} -eq 0 ]; then
        echo "  (no files matched ncu_*_${suffix}.log)"
        continue
    fi

    for file in "${sorted_files[@]}"; do
        echo "  → $file"

        # --- Bandwidth metric (second 'l1tex__m_xbar2l1tex_read_bytes.sum.per_second') ---
        line1=$(grep " l1tex__m_xbar2l1tex_read_bytes.sum.per_second " "$file" | sed -n '2p')
        if [ -z "$line1" ]; then
            echo "    Warning: primary metric line #2 not found in $file" >&2
            continue
        fi
        num1=$(extract_last_number "$line1")
        if [[ "$line1" == *"Gbyte/s"* ]]; then
            gbps="$num1"
        else
            # assume Tbyte/s
            gbps=$(awk -v v="$num1" 'BEGIN{ printf "%.3f", v*1e3 }')
        fi

        # --- Kernel time metric (second 'gpu__time_duration.avg') ---
        line2=$(grep "gpu__time_duration.avg" "$file" | sed -n '2p')
        if [ -z "$line2" ]; then
            echo "    Warning: time-duration line #2 not found in $file" >&2
            continue
        fi
        num2=$(extract_last_number "$line2")
        if [[ "$line2" == *"us"* ]]; then
            us="$num2"
        elif [[ "$line2" == *"ms"* ]]; then
            us=$(awk -v v="$num2" 'BEGIN{ printf "%.3f", v*1e3 }')
        else
            # plain seconds or anything else → µs
            us=$(awk -v v="$num2" 'BEGIN{ printf "%.3f", v*1e6 }')
        fi

        echo "$gbps" >> "$bw_output"
        echo "$us"   >> "$ket_output"
    done

    echo "  → Saved  BW → $bw_output"
    echo "  → Saved KET → $ket_output"
    echo
done

echo "All patterns processed."
