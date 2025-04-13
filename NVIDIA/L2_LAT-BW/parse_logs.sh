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
#   1. From the line containing:
#          l1tex__m_xbar2l1tex_read_bytes.sum.per_second
#      the script extracts the last numeric value. It then checks if this line contains
#      "GB/s". If not - if it contains "TB/s" - it converts the value to GB/s.
#
#   2. From the line containing:
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

# Find files matching the expected pattern.
files=( ncu_*_*.log )
if [ ${#files[@]} -eq 0 ]; then
    echo "No ncu log files found."
    exit 1
fi

# Use a sample filename to extract the common fields.
sample="${files[0]}"
# Expected sample: ncu_<threads>_<CTAs>_<accesspattern>[...].log
IFS='_' read -r _ threads CTAs rest <<< "$sample"
accesspattern=$(echo "$rest" | cut -d'.' -f1 | cut -d'_' -f1)
# Check for optional random delay suffix (e.g. _rand_delay_warp_32)
if [[ "$rest" =~ rand_delay ]]; then
    rand_suffix=$(echo "$rest" | sed 's/\.[^.]*$//')
    rand_suffix=${rand_suffix#$accesspattern}
else
    rand_suffix=""
fi

# Build the output file names.
bw_output="BW_${accesspattern}_${CTAs}${rand_suffix}.log"
ket_output="KET_${accesspattern}_${CTAs}${rand_suffix}.log"

# Clear the output files.
> "$bw_output"
> "$ket_output"

# List files matching the pattern and sort by the threads_per_CTA field (second underscore field) numerically.
sorted_files=$(ls ncu_*_"${CTAs}"_"${accesspattern}"*".log" 2>/dev/null | sort -t '_' -k2,2n)
if [ -z "$sorted_files" ]; then
    echo "No files matching pattern ncu_*_${CTAs}_${accesspattern}*.log found."
    exit 1
fi

echo "Processing files in order:"
for file in $sorted_files; do
    echo "  $file"
done

# Process each sorted file.
for file in $sorted_files; do
    # For ordering, we assume the file name's second field is the threads_per_CTA,
    # but we no longer output that value.
    
    # Extract the primary metric line.
    line1=$(grep -m 1 "l1tex__m_xbar2l1tex_read_bytes.sum.per_second" "$file")
    if [ -z "$line1" ]; then
        echo "Warning: Primary metric not found in $file" >&2
        continue
    fi
    num1=$(extract_last_number "$line1")
    # Convert to GB/s if necessary.
    if [[ "$line1" == *"GB/s"* ]]; then
        gbps="$num1"
    elif [[ "$line1" == *"TB/s"* ]]; then
        gbps=$(awk -v val="$num1" 'BEGIN { printf "%.3f", val*1e3 }')
    else
        # Assume B/s if no unit is found.
        gbps=$(awk -v val="$num1" 'BEGIN { printf "%.3f", val/1e9 }')
    fi

    # Extract the second metric line.
    line2=$(grep -m 1 "gpu__time_duration.avg" "$file")
    if [ -z "$line2" ]; then
        echo "Warning: Second metric not found in $file" >&2
        continue
    fi
    num2=$(extract_last_number "$line2")
    # Convert to microseconds if needed.
    if [[ "$line2" == *"us"* ]]; then
        us="$num2"
    elif [[ "$line2" == *"ms"* ]]; then
        us=$(awk -v val="$num2" 'BEGIN { printf "%.3f", val*1e3 }')
    elif [[ "$line2" == *"s"* ]]; then
        us=$(awk -v val="$num2" 'BEGIN { printf "%.3f", val*1e6 }')
    else
        # Default to seconds conversion.
        us=$(awk -v val="$num2" 'BEGIN { printf "%.3f", val*1e6 }')
    fi

    # Write only the converted values in order.
    echo "$gbps" >> "$bw_output"
    echo "$us" >> "$ket_output"
done

echo "Parsing complete."
echo "Bandwidth results saved to: $bw_output"
echo "Kernel execution time results saved to: $ket_output"
