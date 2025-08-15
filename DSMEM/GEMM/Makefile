NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
# CUDA_ARCH_H100: Use compute capability 9.0 (H100) and enable device linking (-rdc=true) for cooperative groups/DSMEM
CUDA_ARCH_H100 = -arch=sm_90 -rdc=true
# executable file name
TARGET_H100 = matmul_h100 
TARGET_SAMPLE = matmul

# 'all' target: build both H100 and regular matmul executables
all: $(TARGET_H100) $(TARGET_SAMPLE) 

# build the H100 matmul executable from matmul_h100.cu
# with NVCC, using flags and H100 architecture options 
$(TARGET_H100): matmul_h100.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_H100) -o $@ $^

$(TARGET_SAMPLE): matmul.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_H100) -o $@ $^

# 'clean' target: removes all generated executables and object files to ensure a fresh build
clean:
	rm -f $(TARGET_H100) $(TARGET_SAMPLE) *.o

# Declare 'all' and 'clean' as phony targets (not actual files), 
# so make always runs their recipes
.PHONY: all clean
