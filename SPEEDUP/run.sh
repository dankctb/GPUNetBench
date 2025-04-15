#!/usr/bin/env bash
#
#
# Usage:
#   ./run.sh [operation] [architecture] [sort] [mode]
#
#   operation:  read | write       (default: write)
#   architecture: v100 | a100 | h100 | h100cpc (default: v100)
#   sort:       gpc | tpc         (default: gpc)
#   mode:       run | profile | both   (default: both)

set -e

OP=${1:-write}
ARCH=${2:-v100}
SORT=${3:-gpc}
MODE=${4:-both}

echo ">>> Compiling: OP=${OP}, ARCH=${ARCH}, SORT=${SORT}"
make clean
make OP=${OP} ARCH=${ARCH} SORT=${SORT}

BIN=SPEEDUP
CTA=2
WARP=32
ITER=1
GPC=0

for SM in $(seq 1 14); do
  if [[ "$MODE" == "run" || "$MODE" == "both" ]]; then
    echo ">>> Normal run: OP=${OP}, ARCH=${ARCH}, SORT=${SORT}, GPC=${GPC}, SM=${SM}"
    OUTFILE="out_${ARCH}_${OP}_${SORT}_gpc${GPC}_sm${SM}.txt"
    ./${BIN} $CTA $WARP $ITER $GPC $SM >> "$OUTFILE"
  fi

  if [[ "$MODE" == "profile" || "$MODE" == "both" ]]; then
    echo ">>> Profiling run: OP=${OP}, ARCH=${ARCH}, SORT=${SORT}, GPC=${GPC}, SM=${SM}"
    LOGFILE="prof_${ARCH}_${OP}_${SORT}_gpc${GPC}_sm${SM}.log"
    nvprof \
      --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
      --log-file "$LOGFILE" \
      ./${BIN} $CTA $WARP $ITER $GPC $SM
  fi
done
