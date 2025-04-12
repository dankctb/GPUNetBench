#!/bin/bash
# This script profiles the CUDA program ("MP") with nvprof and plot the results.
# The program expects the following arguments:
#   ./MP <GPCselected> <SMmax> <numMemoryPartitions> <MP_id[0]> ... <MP_id[numMemoryPartitions-1]>
#
# Explanation of parameters:
#   - GPCselected: a comma-separated list of Graphics Processing Cluster IDs (0..5) || 6 has 28 interleaved SMs from different GPCs (with no TPC contention) 
#   - SMmax: The maximum Streaming Multiprocessor (SM) index active within the selected GPC.
#   - numMemoryPartitions: The number of memory partitions (MPs) selected.
#                         (Each MP contributes up to 8 slices; total slices = numMemoryPartitions * slicesPerMp.)
#   - slicesPerMP: number of slices to extract per MP (up to 8 based on current mapping)
#   - MP_id[i]: List of memory partition (MP) IDs (from 0 to 3) corresponding to each selected MP.
#

nvcc -Xptxas -dlcm=cg -O3 -DITERATION=2000 -arch=sm_70 -o MP main.cu

# Perform 1 GPC to increasing L2 MP

GPCselected=0
maxSMIndex=14
slicesPerMP=8

for ((numMemoryPartitions = 1; numMemoryPartitions < 5; numMemoryPartitions++)); do
    selectedMPIDs=""
    for ((mpIndex = 0; mpIndex < numMemoryPartitions; mpIndex++)); do
        selectedMPIDs+="$mpIndex "
    done
    
    # Compose the log file name such that it reflects the current parameters.
    logFileName="GPC${GPCselected}_SM${maxSMIndex}_MP${numMemoryPartitions}.log"
    
    nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
            --log-file "$logFileName" \
            ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs

    #nvprof --aggregate-mode off -e all --log-file "$logFileName" \
    #    ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs
done


# Select 28 interleaved SMs.
# Perform 14 and 28 SMs to increasing L2 MP

GPCselected=6

for ((maxSMIndex = 14; maxSMIndex < 29; maxSMIndex+=14)); do
    for ((numMemoryPartitions = 1; numMemoryPartitions < 5; numMemoryPartitions++)); do
        selectedMPIDs=""
        for ((mpIndex = 0; mpIndex < numMemoryPartitions; mpIndex++)); do
            selectedMPIDs+="$mpIndex "
        done
        
        # Compose the log file name such that it reflects the current parameters.
        logFileName="distributedSM${maxSMIndex}_MP${numMemoryPartitions}.log"
        
        # Run nvprof with the proper set of metrics and redirect the output to the log file.
        nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
                --log-file "$logFileName" \
                ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs

        #nvprof --aggregate-mode off -e all --log-file "$logFileName" \
    #        ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs
    done
done

# Select 2 GPCs with 28 SMs.
# Perform 28 SMs to increasing L2 MP

GPCselected="0,1"
maxSMIndex=28

for ((numMemoryPartitions = 1; numMemoryPartitions < 5; numMemoryPartitions++)); do
    selectedMPIDs=""
    for ((mpIndex = 0; mpIndex < numMemoryPartitions; mpIndex++)); do
        selectedMPIDs+="$mpIndex "
    done
    
    # Compose the log file name such that it reflects the current parameters.
    logFileName="contiguousSM${maxSMIndex}_MP${numMemoryPartitions}.log"
    
    # Run nvprof with the proper set of metrics and redirect the output to the log file.
    nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
            --log-file "$logFileName" \
            ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs

    #nvprof --aggregate-mode off -e all --log-file "$logFileName" \
#        ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $selectedMPIDs
done


# Select all the SMs
nvcc -Xptxas -dlcm=cg -O3 -arch=sm_70 -DENABLE_ALL_SMS -DITERATION=2000 -o MP main.cu

# Perform all SMs to increasing L2 slices in MP 0

GPCselected="0,1,2,3,4,5"
maxSMIndex=80
numMemoryPartitions=1
mpIndex=1

for ((slicesPerMP = 1; slicesPerMP < 9; slicesPerMP++)); do
    
    # Compose the log file name such that it reflects the current parameters.
    logFileName="SM${maxSMIndex}_MP${numMemoryPartitions}_slices${slicesPerMP}.log"
    
    nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
            --log-file "$logFileName" \
            ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $mpIndex

    #nvprof --aggregate-mode off -e all --log-file "$logFileName" \
#        ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $mpIndex
done


# Select all the SMs.
# Perform all SMs to increasing L2 slices in interleaved MPs

GPCselected="0,1,2,3,4,5"
maxSMIndex=80
numMemoryPartitions=1
mpIndex=4

for ((slicesPerMP = 1; slicesPerMP < 9; slicesPerMP++)); do
    
    # Compose the log file name such that it reflects the current parameters.
    logFileName="SM${maxSMIndex}_interleavedMP_slices${slicesPerMP}.log"
    
    nvprof --metrics l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,l2_write_throughput \
            --log-file "$logFileName" \
            ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $mpIndex

    #nvprof --aggregate-mode off -e all --log-file "$logFileName" \
#        ./MP $GPCselected $maxSMIndex $numMemoryPartitions $slicesPerMP $mpIndex
done
