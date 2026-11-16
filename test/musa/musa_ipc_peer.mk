# test/musa/musa_ipc_peer.mk
#
# Small standalone build for the two-process MUSA IPC peer gate. It mirrors the
# working repo build split:
#   * kernel translation unit (.mu)  -> mcc, -fPIC, --offload-arch=mp_31
#   * host translation unit (.cc)    -> g++ + MPI + FlagCX/MUSA includes
#   * link                           -> g++ + libflagcx + MUSA runtime + MPI
#
# `mcc -dlink` is intentionally absent (unsupported on this toolchain).
#
# Typical invocation:
#   make -f musa_ipc_peer.mk BUILD_DIR=... BIN=... MUSA_HOME=/usr/local/musa

MUSA_IPC_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(MUSA_IPC_DIR)/../..)

MUSA_HOME ?= /usr/local/musa
MUSA_ARCH ?= mp_31
MCC := $(MUSA_HOME)/bin/mcc
HOST_CXX ?= g++

BUILD_DIR ?= $(ROOT)/build/musa_device_api_results/ipc_peer
OBJ_DIR := $(BUILD_DIR)/obj
BIN ?= $(BUILD_DIR)/bin/musa_ipc_peer

FLAGCX_BUILD_LIB ?= $(ROOT)/build/lib

MPI_CFLAGS := $(shell mpicxx --showme:compile 2>/dev/null)
MPI_LDFLAGS := $(shell mpicxx --showme:link 2>/dev/null)
ifeq ($(strip $(MPI_CFLAGS)),)
MPI_CFLAGS := -I/usr/include/x86_64-linux-gnu/mpi
MPI_LDFLAGS := -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
endif

HOST_CFLAGS := -std=c++17 -g -O2 -fPIC -pthread -Wall -Wno-unused-function
HOST_DEFINES := -DUSE_MUSA_ADAPTOR -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
HOST_INCLUDES := -I$(ROOT)/flagcx/include \
                 -I$(ROOT)/flagcx/adaptor/include \
                 -I$(MUSA_HOME)/include

KERNEL_OBJ := $(OBJ_DIR)/musa_ipc_peer_kernels.o
HOST_OBJ := $(OBJ_DIR)/musa_ipc_peer_test.o

.PHONY: all clean print-config

all: $(BIN)

$(KERNEL_OBJ): $(MUSA_IPC_DIR)/musa_ipc_peer_kernels.mu
	@mkdir -p $(OBJ_DIR)
	$(MCC) -std=c++17 --offload-arch=$(MUSA_ARCH) -fPIC \
		-I$(MUSA_HOME)/include -c $< -o $@

$(HOST_OBJ): $(MUSA_IPC_DIR)/musa_ipc_peer_test.cc
	@mkdir -p $(OBJ_DIR)
	$(HOST_CXX) $(HOST_CFLAGS) $(HOST_DEFINES) $(MPI_CFLAGS) $(HOST_INCLUDES) \
		-c $< -o $@

$(BIN): $(KERNEL_OBJ) $(HOST_OBJ)
	@mkdir -p $(dir $(BIN))
	$(HOST_CXX) -std=c++17 -fPIC -pthread $^ -o $@ \
		-L$(FLAGCX_BUILD_LIB) -Wl,--no-as-needed \
		-Wl,-rpath,$(FLAGCX_BUILD_LIB) -lflagcx \
		$(MPI_LDFLAGS) \
		-L$(MUSA_HOME)/lib -Wl,-rpath,$(MUSA_HOME)/lib -lmusart -lmusa

print-config:
	@echo "BIN=$(BIN)"
	@echo "MCC=$(MCC)"
	@echo "MUSA_ARCH=$(MUSA_ARCH)"
	@echo "FLAGCX_BUILD_LIB=$(FLAGCX_BUILD_LIB)"
	@echo "MPI_CFLAGS=$(MPI_CFLAGS)"
	@echo "MPI_LDFLAGS=$(MPI_LDFLAGS)"

clean:
	@rm -f $(KERNEL_OBJ) $(HOST_OBJ) $(BIN)
