NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
CUDA_ARCH_H100 = -arch=sm_90 -rdc=true -Xptxas -dlcm=cg
CUDA_ARCH_REGULAR = -arch=sm_75

TARGET_H100 = matmul_h100
TARGET_REGULAR = matmul

all: $(TARGET_H100) $(TARGET_REGULAR)

$(TARGET_H100): matmul_h100.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_H100) -o $@ $^

$(TARGET_REGULAR): matmul.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_REGULAR) -o $@ $^

clean:
	rm -f $(TARGET_H100) $(TARGET_REGULAR) *.o

.PHONY: all clean
