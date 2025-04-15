NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := SM2SM

ARCH_FLAGS := -arch=sm_90

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
