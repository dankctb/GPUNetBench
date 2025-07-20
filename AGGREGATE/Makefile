NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

# Default architecture; can be set to v100, a100, or h100.
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

.PHONY: all clean main all_arch
all: BW

# Build original main.cu
main: BW
BW: main.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

# Build all_arch_main.cu
all_arch: all_arch_BW
all_arch_BW: all_arch_main.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f BW analysis_BW
