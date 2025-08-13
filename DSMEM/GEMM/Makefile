NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
CUDA_ARCH_H100 = -arch=sm_90 -rdc=true -Xptxas -dlcm=cg -DCOMPILE_H100

TARGET_H100 = matmul_h100

all: $(TARGET_H100)

$(TARGET_H100): matmul_h100.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_H100) -o $@ $^

clean:
	rm -f $(TARGET_H100) *.o

.PHONY: all clean
