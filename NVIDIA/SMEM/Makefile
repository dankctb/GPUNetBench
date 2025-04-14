NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := SMEM

# Default architecture; set ARCH to v100, a100, or h100 as needed.
ARCH ?= v100

ifeq ($(ARCH),v100)
    ARCH_FLAGS := -arch=sm_70
endif
ifeq ($(ARCH),a100)
    ARCH_FLAGS := -arch=sm_80
endif
ifeq ($(ARCH),h100)
    ARCH_FLAGS := -arch=sm_90
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
