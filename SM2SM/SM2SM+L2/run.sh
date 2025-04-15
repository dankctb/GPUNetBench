#!/usr/bin/env bash
#
# run.sh – Sweep ILP factors and traffic path configurations using the Makefile.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh [default|custom] [--dsmdest 1|0] [--readall 1|0]
#
# Parameters:
#   Role map selection:
#       default : Uses built‑in mapping.
#       custom  : Uses custom mapping (CUSTOM_ROLES).
#
#   Optional flags:
#       --dsmdest 1|0 : Enable (1) or disable (0) DSM_DEST_OTHER_CPC flag.
#       --readall 1|0 : Enable (1) or disable (0) READ_FULL flag.
#
# Configuration:
#   - Block size:     1024
#   - Cluster number: 1
#   - Cluster size:   16 (compile‑time constant)
#
# Experiment groups:
#   1) SM2SM‑only:       Vary ILP_DSM in {1,2,4,8}, disable L2 (ENABLE_L2=0)
#   2) L2‑only:          Vary ILP_L2 in {1,2,4,8}, disable SM2SM (ENABLE_SM2SM=0)
#   3) BOTH (L2 ILP fixed to 8): Vary ILP_DSM in {1,2,4,8}
#   4) BOTH (SM2SM ILP fixed to 8): Vary ILP_L2 in {1,2,4,8}
#
# Testing both metrics: CALC_BW and CALC_LATENCY.
#
# CPC / RANK MAP (H100 16‑rank GPC):
#   - CPC0 : ranks { 4, 5, 10, 11 }
#   - CPC1 : ranks { 2, 3, 8, 9, 14, 15 }
#   - CPC2 : ranks { 0, 1, 6, 7, 12, 13 }
#
# The logs for every run are stored in the results directory.
#

# -------- USER‑SELECTABLE RANK MAPS ----------------------------------------
DEFAULT_ROLES=""
CUSTOM_ROLES="2,1,0,0,0,0,0,0,0,0,0,0,2,1,0,0"
#-------------0-1-2-3-4-5-6-7-8-9-A-B-C-D-E-F--------------------------------

# Process the first argument: role map selection.
ROLESET=${1:-default}
if [[ "${ROLESET}" == "custom" ]]; then
    ROLES="${CUSTOM_ROLES}"
else
    ROLES="${DEFAULT_ROLES}"
fi
shift

# -------- OPTIONAL FLAGS ----------------------------------------------------
# Default values: disabled (0)
DSM_DEST_OPTION=0
READ_ALL_OPTION=0

while (( "$#" )); do
    case "$1" in
        --dsmdest)
            DSM_DEST_OPTION="$2"
            shift 2
            ;;
        --readall)
            READ_ALL_OPTION="$2"
            shift 2
            ;;
        *) # Unrecognized option.
            echo "Warning: Unrecognized option '$1'"
            shift
            ;;
    esac
done

# -------- FIXED PARAMETERS -------------------------------------------------
BLOCKSIZE=1024
NCLUSTERS=1
ILP_LIST=(1 2 4 8)

# Output directory for logs and executables.
OUTDIR="results"
mkdir -p "${OUTDIR}"

# -------- Helper Function: build_and_run -----------------------------------
build_and_run () {
    local tag=$1         # Descriptive tag for this run (e.g. SM2SMonly)
    local ilp_l2=$2      # ILP for L2 branch
    local ilp_dsm=$3     # ILP for DSM branch
    local enable_l2=$4   # 0 or 1
    local enable_dsm=$5  # 0 or 1
    local metric=$6      # 'bw' or 'lat'
    local exe="${OUTDIR}/${tag}_${metric}_ILPL2${ilp_l2}_ILPDSM${ilp_dsm}"
    
    # Set compile‑time flags via NVCC_DEFS.
    export NVCC_DEFS="-DILP_L2=${ilp_l2} -DILP_DSM=${ilp_dsm} -DENABLE_L2=${enable_l2} -DENABLE_SM2SM=${enable_dsm}"
    
    if [ "$DSM_DEST_OPTION" -eq 1 ]; then
        export NVCC_DEFS="${NVCC_DEFS} -DDSM_DEST_OTHER_CPC"
    fi
    if [ "$READ_ALL_OPTION" -eq 1 ]; then
        export NVCC_DEFS="${NVCC_DEFS} -DREAD_ALL2ALL_FULL"
    fi
    
    if [[ "${metric}" == "lat" ]]; then
        export NVCC_DEFS="${NVCC_DEFS} -DCALC_LATENCY -U CALC_BW"
    fi

    echo "==== Building ${exe} ===="
    # Build the target using the Makefile. Your Makefile produces an executable named SM2SM+L2.
    make clean >/dev/null 2>&1
    make || exit 1

    # Rename the produced binary.
    mv SM2SM+L2 "${exe}"
    
    echo "==== Running ${exe} ===="
    # Run the executable: pass NCLUSTERS, BLOCKSIZE, and optionally the role mapping.
    if [[ -z "${ROLES}" ]]; then
        ./"${exe}" ${NCLUSTERS} ${BLOCKSIZE} > "${exe}.log"
    else
        ./"${exe}" ${NCLUSTERS} ${BLOCKSIZE} "${ROLES}" > "${exe}.log"
    fi
    echo "Log stored in ${exe}.log"
    echo
}

# ==================== 1) SM2SM‑ONLY ========================================
for ilp in "${ILP_LIST[@]}"; do
    for met in bw lat; do
        build_and_run "SM2SMonly" 8 "${ilp}" 0 1 "${met}"
    done
done

# ==================== 2) L2‑ONLY ===========================================
for ilp in "${ILP_LIST[@]}"; do
    for met in bw lat; do
        build_and_run "L2only" "${ilp}" 8 1 0 "${met}"
    done
done

# ==================== 3) BOTH (L2 fixed=8, vary DSM ILP) =======================
for ilp in "${ILP_LIST[@]}"; do
    for met in bw lat; do
        build_and_run "Both_L2fix8" 8 "${ilp}" 1 1 "${met}"
    done
done

# ==================== 4) BOTH (DSM fixed=8, vary L2 ILP) =======================
for ilp in "${ILP_LIST[@]}"; do
    for met in bw lat; do
        build_and_run "Both_SM2SMfix8" "${ilp}" 8 1 1 "${met}"
    done
done

echo "All runs finished. Check the ${OUTDIR} directory for logs and executables."
