
# Usage: make OP=<read|write> ARCH=<v100|a100|h100> SORT=<gpc|tpc>
#   e.g. make OP=read ARCH=v100 SORT=tpc

NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := SPEEDUP

# Defaults
OP   ?= write
ARCH ?= v100
SORT ?= gpc

# Operation flag
ifeq ($(OP),read)
  NVCC_DEFS += -DUSE_READ
endif

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
ifeq ($(ARCH),h100cpc)
  ARCH_FLAGS := -arch=sm_90
  NVCC_DEFS  += -DUSE_H100CPC
endif

# SORT flag
#  - tpc : default ascending SM IDs
#  - gpc : even SMs first, then odd  (requires SORT_GPC in code)
ifeq ($(SORT),gpc)
  NVCC_DEFS += -DSORT_GPC
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
