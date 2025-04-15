#!/bin/bash
#
# This script will run the Local SMem benchmark compiled from main.cu.
#
# It uses the following compile‑time definitions:
#   - Access pattern is set via: -DSTRIDE (if set to 1, it is a stream access, otherwise it is a strided access)
#   - The ILP unrolling factor is set via -DILP_FACTOR
#   - Measurement mode: either -DCALC_LATENCY or -DCALC_BW
#
# The executable expects two runtime parameters: <blockSize> and <numBlocks>.
# In this script we use 1 block.
#
# Output files are named using the mode, access pattern, stride, block size, and number of blocks.
#
# Define arrays for ILP factors, block sizes (threads per block) or stride.
ILP_FACTORS=(1 2 4 8)
BLOCK_SIZES=(32 64 128 256 512 1024)
STRIDE=(1 2 4 8 16 32)
BLOCKS=1

# Loop over the two measurement modes: latency and bandwidth.
for mode in latency bw; do
    if [ "$mode" == "latency" ]; then
       MODE_FLAG="-DCALC_LATENCY"
    else
       MODE_FLAG="-DCALC_BW"
    fi
    for ilp in "${ILP_FACTORS[@]}"; do
      for block in "${BLOCK_SIZES[@]}"; do
        for stride in "${STRIDE[@]}"; do
         echo "Compiling with:"
         echo "  ILP_FACTOR = $ilp"
         echo "  BLOCK_SIZE = $block"
         echo "  Mode       = $mode "
         echo "  STRIDE     = $stride"
         
         # Set compile definitions via NVCC_DEFS:
         export NVCC_DEFS="-DSTRIDE=${stride} -DILP_FACTOR=${ilp} ${MODE_FLAG}"
         
         # Clean and compile the target.
         make clean
         make
         
         # Create an output file name that identifies the configuration.
         outfile="smem_${mode}_stride${stride}_${block}_${BLOCKS}.txt"
         
         # Run the benchmark (executable SM) with block size (threads per block) and 1 block.
         ./SMEM $block $BLOCKS >> "$outfile"
      done
    done
done
