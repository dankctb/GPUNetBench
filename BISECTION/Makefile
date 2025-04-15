NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := BISECTION

# Default architecture; set ARCH to a100, or h100 as needed.
ARCH ?= a100

NVCC_DEFS := -DPARTITION=$(PARTITION) -DITERATION=$(ITERATION)
ifeq ($(ARCH),a100)
  ARCH_FLAGS := -arch=sm_80
  NVCC_DEFS += -DUSE_A100
endif
ifeq ($(ARCH),h100)
  ARCH_FLAGS := -arch=sm_90
  NVCC_DEFS += -DUSE_H100
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(NVCC_DEFS) -o $@ $<

clean:
	rm -f $(TARGET)
