NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := L2_LAT-BW

# Default architecture; can be set to v100, a100, or h100.
ARCH ?= v100

ifeq ($(ARCH),v100)
    ARCH_FLAGS := -arch=sm_70
    L2_FLAG  := -DUSABLE_BYTES=5242880        # 5 * 1024 * 1024
endif
ifeq ($(ARCH),a100)
    ARCH_FLAGS := -arch=sm_80
    L2_FLAG  := -DUSABLE_BYTES=36700160       # 35 * 1024 * 1024
endif
ifeq ($(ARCH),h100)
    ARCH_FLAGS := -arch=sm_90
    L2_FLAG  := -DUSABLE_BYTES=47185920       # 45 * 1024 * 1024
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(L2_FLAG) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
