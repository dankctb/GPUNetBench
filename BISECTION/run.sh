#!/bin/bash

# ───────────────────────────────────────────────────────────────
# User‑configurable parameters
# ───────────────────────────────────────────────────────────────

# GPU architecture: "a100" or "h100"
if [ -z "$ARCH" ]; then
  ARCH="h100"
fi

# L2 partition: 0 or 1
PARTITION=0

ITERATION=1

# CTAs per SM (fixed to 32 as requested)
CTAS_PER_SM=32

# List of GPC IDs to sweep over incrementally
GPC_LIST=(0 2 3 4) # Example GPC IDs in one partition in H100 PCIe

# ───────────────────────────────────────────────────────────────
# Build the program with the selected ARCH and PARTITION
# ───────────────────────────────────────────────────────────────

echo "=== Building for ARCH=${ARCH}, PARTITION=${PARTITION}, ITERATION=${ITERATION} ==="
make clean
make ARCH=${ARCH} PARTITION=${PARTITION} ITERATION=${ITERATION}


if [[ ! -x BISECTION ]]; then
  echo "ERROR: BISECTION binary not found or not executable."
  exit 1
fi

# ───────────────────────────────────────────────────────────────
# Run the binary with incremental GPC sets
# ───────────────────────────────────────────────────────────────

for (( i=1; i<=${#GPC_LIST[@]}; i++ )); do
  # take the first $i entries from GPC_LIST
  GPC_IDS=( "${GPC_LIST[@]:0:$i}" )
  echo
  echo ">>> Running with GPC IDs: ${GPC_IDS[*]}"
  ./BISECTION ${CTAS_PER_SM} "${GPC_IDS[@]}" >> BISECTION.log
done
