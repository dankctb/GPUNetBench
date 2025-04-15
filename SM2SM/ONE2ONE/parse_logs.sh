#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for file in *.log; do
  # skip already‐generated hist files
  [[ "$file" == *_hist.log ]] && continue

  mode="${file%%_*}"
  out="${file%.log}_hist.log"

  case "$mode" in
    latency)
      # 1) pick only lines with “Avg Latency”
      # 2) from those lines, pull out every float (one per line)
      grep "Avg Latency" "$file" \
        | grep -oE '[0-9]+\.[0-9]+' \
        > "$out"
      ;;
    bw)
      # same for “Bandwidth”
      grep "Bandwidth" "$file" \
        | grep -oE '[0-9]+\.[0-9]+' \
        > "$out"
      ;;
    *)
      echo "Skipping '$file': unknown mode '$mode'" >&2
      continue
      ;;
  esac

  echo "Wrote all ${mode} values to $out"
done
