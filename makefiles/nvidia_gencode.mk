#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 BAAI. All rights reserved.
#

# -----------------
# This file is included by Makefile when USE_NVIDIA=1.
# It detects the CUDA version and sets the DEVICE_COMPILER_GENCODE variable accordingly.
# ----------------
# Purpose: detect CUDA version and set DEVICE_COMPILER_GENCODE

NVCC ?= $(DEVICE_HOME)/bin/nvcc
CUDA_LIB ?= $(DEVICE_HOME)/lib64
CUDA_INC ?= $(DEVICE_HOME)/include

# Detect CUDA version from nvcc
CUDA_VERSION := $(strip $(shell \
	which $(NVCC) >/dev/null 2>&1 && \
	$(NVCC) --version | grep release | sed 's/.*release //' | sed 's/,.*//' \
))

CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

# ----------------
# GENCODE tables
# ----------------
CUDA8_GENCODE   = -gencode=arch=compute_60,code=sm_60 \
                  -gencode=arch=compute_61,code=sm_61
CUDA9_GENCODE   = -gencode=arch=compute_70,code=sm_70
CUDA10_GENCODE  = -gencode=arch=compute_75,code=sm_75
CUDA11_GENCODE  = -gencode=arch=compute_80,code=sm_80
CUDA12_GENCODE  = -gencode=arch=compute_90,code=sm_90
CUDA12_8_GENCODE = -gencode=arch=compute_100,code=sm_100 \
                   -gencode=arch=compute_120,code=sm_120
CUDA13_GENCODE  = -gencode=arch=compute_110,code=sm_110

CUDA8_PTX  = -gencode=arch=compute_61,code=compute_61
CUDA9_PTX  = -gencode=arch=compute_70,code=compute_70
CUDA11_PTX = -gencode=arch=compute_80,code=compute_80
CUDA12_PTX = -gencode=arch=compute_90,code=compute_90
CUDA13_PTX = -gencode=arch=compute_120,code=compute_120

# ----------------
# Decide DEVICE_COMPILER_GENCODE
# ----------------
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA10_GENCODE) \
    $(CUDA11_GENCODE) \
    $(CUDA12_GENCODE) \
    $(CUDA12_8_GENCODE) \
    $(CUDA13_GENCODE) \
    $(CUDA13_PTX)

else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8; echo $$?),0)
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA8_GENCODE) \
    $(CUDA9_GENCODE) \
    $(CUDA11_GENCODE) \
    $(CUDA12_GENCODE) \
    $(CUDA12_8_GENCODE) \
    $(CUDA13_PTX)

else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 11 -a "0$(CUDA_MINOR)" -ge 8 -o "0$(CUDA_MAJOR)" -gt 11; echo $$?),0)
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA8_GENCODE) \
    $(CUDA9_GENCODE) \
    $(CUDA11_GENCODE) \
    $(CUDA12_GENCODE) \
    $(CUDA12_PTX)

else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA8_GENCODE) \
    $(CUDA9_GENCODE) \
    $(CUDA11_GENCODE) \
    $(CUDA11_PTX)

else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 9; echo $$?),0)
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA8_GENCODE) \
    $(CUDA9_GENCODE) \
    $(CUDA9_PTX)

else
  DEVICE_COMPILER_GENCODE ?= \
    $(CUDA8_GENCODE) \
    $(CUDA8_PTX)
endif

# ----------------
# Multicast primitive support check (requires sm_90+)
# Keep sm_80+ for A100 compatibility, multicast kernels use __CUDA_ARCH__ guards
# ----------------

# Define all sm_90+ gencodes for detection
SM90_PLUS_GENCODES = $(CUDA12_GENCODE) $(CUDA12_8_GENCODE) $(CUDA13_GENCODE) $(CUDA12_PTX) $(CUDA13_PTX)

# Check if any sm_90+ gencode is present in DEVICE_COMPILER_GENCODE
HAS_SM90_PLUS := $(strip $(foreach gc,$(SM90_PLUS_GENCODES),$(findstring $(gc),$(DEVICE_COMPILER_GENCODE))))

ifneq ($(HAS_SM90_PLUS),)
  # Has sm_90+ support: keep sm_80+ for compatibility
  # Multicast kernels use #if __CUDA_ARCH__ >= 900 guards in source code
  DEVICE_COMPILER_GENCODE := $(filter-out $(CUDA8_GENCODE),$(DEVICE_COMPILER_GENCODE))
  DEVICE_COMPILER_GENCODE := $(filter-out $(CUDA9_GENCODE),$(DEVICE_COMPILER_GENCODE))
  DEVICE_COMPILER_GENCODE := $(filter-out $(CUDA10_GENCODE),$(DEVICE_COMPILER_GENCODE))
  DEVICE_COMPILER_GENCODE := $(filter-out $(CUDA8_PTX),$(DEVICE_COMPILER_GENCODE))
  DEVICE_COMPILER_GENCODE := $(filter-out $(CUDA9_PTX),$(DEVICE_COMPILER_GENCODE))
  $(info Multicast support enabled (sm_90+). Using DEVICE_COMPILER_GENCODE: ${DEVICE_COMPILER_GENCODE})
else
  # No sm_90+ support: keep all gencodes and disable multicast
  NVCC_GENCODE_MULTICAST_UNSUPPORTED := 1
  $(info WARNING: No sm_90+ architecture detected. Multicast primitives disabled. Custom AllReduce will fallback to ncclAllReduce.)
  $(info Using DEVICE_COMPILER_GENCODE: ${DEVICE_COMPILER_GENCODE})
endif