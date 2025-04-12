# Usage:
#   make ARCH=<v100|a100|h100> MODE=<direct|gpc>
# Example:
#   make ARCH=a100 MODE=direct

NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := SLICE

# Default values
ARCH ?= v100
MODE ?= gpc

# Architecture flags
ifeq ($(ARCH),v100)
    ARCH_FLAGS := -arch=sm_70
endif
ifeq ($(ARCH),a100)
    ARCH_FLAGS := -arch=sm_80
    NVCC_DEFS  += -DUSE_A100
endif
ifeq ($(ARCH),h100)
    ARCH_FLAGS := -arch=sm_90
    NVCC_DEFS  += -DUSE_H100
endif

# Mode flag: direct (for direct SM selection) or gpc (default)
ifeq ($(MODE),direct)
    NVCC_DEFS += -DUSE_DIRECT_SM
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
