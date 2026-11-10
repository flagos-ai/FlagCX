# Standalone CANN/MPI probe: no FlagCX library or test fixture dependency.
DEVICE_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
MPI_HOME ?= /usr/local/mpi
USE_ASCEND ?= 1
include ../../make.inc
MPIRUN ?= mpirun
DEVICE_PAIR ?= 0,1
MPI_ARGS ?=
SOC_VERSION ?=
ASCENDC_CMAKE_DIR ?=
IB_DEVICE ?=
IB_PORT ?= 1
IB_GID_INDEX ?= 0
STREAM_MODE ?= valuewrite
IB_MODE ?= wait

CXXFLAGS += -O0 -Wextra -Werror
CPPFLAGS += -I$(DEVICE_HOME)/include
LDFLAGS += -L$(DEVICE_HOME)/lib64 -Wl,-rpath,$(DEVICE_HOME)/lib64
LDLIBS += -lascendcl

.PHONY: all stream-build stream-run-cross stream-run-same ib-build ib-run-cross run-cross run-same clean

all: build/vmm_probe

build/vmm_probe: vmm_probe.cpp
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(MPI_INCLUDE) $(CXXFLAGS) $< -o $@ \
	  $(LDFLAGS) $(MPI_LINK) $(LDLIBS)

run-cross: build/vmm_probe
	$(MPIRUN) $(MPI_ARGS) -np 2 ./build/vmm_probe --devices $(DEVICE_PAIR)

run-same: build/vmm_probe
	$(MPIRUN) $(MPI_ARGS) -np 2 ./build/vmm_probe --devices 0,0

stream-build:
	ASCEND_HOME_PATH=$(DEVICE_HOME) SOC_VERSION=$(SOC_VERSION) \
	  ASCENDC_CMAKE_DIR=$(ASCENDC_CMAKE_DIR) \
	  cmake -S . -B build/stream -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/stream --target stream_visibility_probe -j

stream-run-cross: stream-build
	timeout 90s $(MPIRUN) $(MPI_ARGS) -np 2 ./build/stream/stream_visibility_probe --devices $(DEVICE_PAIR) --mode $(STREAM_MODE)

stream-run-same: stream-build
	timeout 90s $(MPIRUN) $(MPI_ARGS) -np 2 ./build/stream/stream_visibility_probe --devices 0,0 --mode $(STREAM_MODE)

ib-build: stream-build
	cmake --build build/stream --target ib_visibility_probe -j

ib-run-cross: ib-build
	IB_DEVICE=$(IB_DEVICE) IB_PORT=$(IB_PORT) IB_GID_INDEX=$(IB_GID_INDEX) \
	  timeout 90s $(MPIRUN) $(MPI_ARGS) -np 2 ./build/stream/ib_visibility_probe --devices $(DEVICE_PAIR) --mode $(IB_MODE)

clean:
	rm -f build/vmm_probe
