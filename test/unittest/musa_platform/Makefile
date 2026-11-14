# Standalone MUSA platform probes. This target does not require MPI or a
# completed test/kernel/musa backend.
.DEFAULT_GOAL := all
PROJECT_ROOT := $(abspath ../../..)
include $(PROJECT_ROOT)/makefiles/musa.mk

BUILD_DIR := $(CURDIR)/build
INCLUDES := -I$(PROJECT_ROOT)/flagcx/adaptor/include \
	-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
	-I$(DEVICE_INCLUDE)
PROBE := $(BUILD_DIR)/musa_platform_probe
HOST_LAYOUT := $(BUILD_DIR)/musa_host_layout
PROBE_DEVICE_OBJ := $(BUILD_DIR)/musa_platform_probe.device.o
PROBE_DEVICE_IMAGE := $(BUILD_DIR)/musa_platform_probe.mubin
PROBE_HOST_OBJ := $(BUILD_DIR)/musa_platform_probe.host.o
RESOURCE_PROBE := $(BUILD_DIR)/musa_launch_resources
RESOURCE_DEVICE_OBJ := $(RESOURCE_PROBE).device.o
RESOURCE_DEVICE_IMAGE := $(RESOURCE_PROBE).mubin
RESOURCE_HOST_OBJ := $(RESOURCE_PROBE).host.o

# Follow MCCL src/device/Makefile: device-only compile, explicit device link,
# then host-only compile embedding the linked image. The mcc 5.1.0 automatic
# link driver was observed to invoke two device links for the combined object.
# DEVICE_HOME may expose runtime libraries while bin/mcc resolves into a
# versioned SDK. Use that compiler's toolchain, matching its verbose link trace.
MUSA_TOOLCHAIN_ROOT ?= $(abspath $(dir $(realpath $(DEVICE_COMPILER)))/..)
MUSA_DEVICE_LD ?= $(MUSA_TOOLCHAIN_ROOT)/bin/lld
# Match the argument emitted by mcc. This path is not necessarily an installed
# file: the server's driver used it successfully in its first device link even
# though LLVMgold.so is absent. Let the SDK linker handle the driver argument;
# do not impose a separate Makefile file-existence requirement.
MUSA_LTO_PLUGIN ?= $(MUSA_TOOLCHAIN_ROOT)/lib/LLVMgold.so
MUSA_RESOURCE_DIR ?= $(shell $(DEVICE_COMPILER) -print-resource-dir)
MUSA_TRAP_HANDLER ?= $(MUSA_RESOURCE_DIR)/lib/linux/trap_handler.o

.DELETE_ON_ERROR:

# Independent regression probe; mcc owns this object's image registration.
.PHONY: device-error
device-error: $(BUILD_DIR)/musa_device_error

$(BUILD_DIR)/musa_device_error.o: musa_device_error.cu Makefile | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(filter-out -fgpu-rdc,$(DEVICE_COMPILE_FLAG)) \
		$(ADAPTOR_FLAG) $(INCLUDES) -I$(PROJECT_ROOT)/flagcx/include \
		-I$(PROJECT_ROOT)/flagcx/core/include -I$(PROJECT_ROOT)/flagcx/runner/include \
		-I$(PROJECT_ROOT)/flagcx/service/include -o $@ $<

$(BUILD_DIR)/musa_device_error: $(BUILD_DIR)/musa_device_error.o \
    $(PROJECT_ROOT)/build/lib/libflagcx.so
	$(CXX) -o $@ $< -L$(PROJECT_ROOT)/build/lib -lflagcx \
		-L$(DEVICE_LIB) -lmusart -lmusa

.PHONY: all run run-groups run-atomics run-cross-block clean barrier-control named-barrier-repro native-trap-repro launch-resources

all: $(HOST_LAYOUT) $(PROBE)

# Host-only diagnostic calling libflagcx's production native stream wrappers.
# Separate from all: it requires the main library to have been built first.
.PHONY: stream-value64
stream-value64: $(BUILD_DIR)/musa_stream_value64

$(BUILD_DIR)/musa_stream_value64: musa_stream_value64.cc \
    $(PROJECT_ROOT)/flagcx/adaptor/include/flagcx_device_adaptor.h \
    $(PROJECT_ROOT)/build/lib/libflagcx.so Makefile | $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -pthread $(INCLUDES) \
		-I$(PROJECT_ROOT)/flagcx/include -o $@ $< \
		-L$(PROJECT_ROOT)/build/lib -lflagcx -L$(DEVICE_LIB) -lmusart -lmusa

$(BUILD_DIR):
	@mkdir -p $@

$(PROBE_DEVICE_OBJ): musa_platform_probe.cu Makefile | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) $(ADAPTOR_FLAG) $(INCLUDES) \
		--cuda-device-only -o $@ $<

$(PROBE_DEVICE_IMAGE): $(PROBE_DEVICE_OBJ) $(MUSA_TRAP_HANDLER) Makefile
	$(MUSA_DEVICE_LD) -flavor gnu --no-undefined \
		-plugin $(MUSA_LTO_PLUGIN) -plugin-opt=mcpu=$(MUSA_ARCH) -plugin-opt=O2 \
		$(PROBE_DEVICE_OBJ) $(MUSA_TRAP_HANDLER) -o $@

$(PROBE_HOST_OBJ): musa_platform_probe.cu $(PROBE_DEVICE_IMAGE) Makefile | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) $(ADAPTOR_FLAG) $(INCLUDES) \
		--cuda-host-only -Xclang -fcuda-include-gpubinary \
		-Xclang $(PROBE_DEVICE_IMAGE) -o $@ $<

# The device image is already linked and registered by PROBE_HOST_OBJ.
# Use the host linker; do not send it through the offload link driver again.
$(PROBE): $(PROBE_HOST_OBJ) Makefile
	$(CXX) -o $@ $(PROBE_HOST_OBJ) \
		-L$(DEVICE_LIB) -lmusart -lmusa

$(HOST_LAYOUT): musa_host_layout.cc | $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -MMD -MP $(ADAPTOR_FLAG) $(INCLUDES) -o $@ $<

# SDK-only controls use the exact same device-link and embedding pipeline.
launch-resources: $(RESOURCE_PROBE)
	$(RESOURCE_PROBE)

$(RESOURCE_DEVICE_OBJ): musa_launch_resources.cu Makefile | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -I$(DEVICE_INCLUDE) \
		--cuda-device-only -o $@ $<

$(RESOURCE_DEVICE_IMAGE): $(RESOURCE_DEVICE_OBJ) $(MUSA_TRAP_HANDLER) Makefile
	$(MUSA_DEVICE_LD) -flavor gnu --no-undefined \
		-plugin $(MUSA_LTO_PLUGIN) -plugin-opt=mcpu=$(MUSA_ARCH) -plugin-opt=O2 \
		$(RESOURCE_DEVICE_OBJ) $(MUSA_TRAP_HANDLER) -o $@

$(RESOURCE_HOST_OBJ): musa_launch_resources.cu $(RESOURCE_DEVICE_IMAGE) Makefile | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -I$(DEVICE_INCLUDE) \
		--cuda-host-only -Xclang -fcuda-include-gpubinary \
		-Xclang $(RESOURCE_DEVICE_IMAGE) -o $@ $<

$(RESOURCE_PROBE): $(RESOURCE_HOST_OBJ) Makefile
	$(CXX) -o $@ $(RESOURCE_HOST_OBJ) -L$(DEVICE_LIB) -lmusart -lmusa

barrier-control: $(BUILD_DIR)/musa_barrier_control.o

named-barrier-repro: $(BUILD_DIR)/musa_named_barrier_repro.o

native-trap-repro: $(BUILD_DIR)/musa_native_trap_repro.o

$(BUILD_DIR)/musa_native_trap_repro.o: musa_native_trap_repro.cu | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -I$(DEVICE_INCLUDE) -o $@ $<

$(BUILD_DIR)/musa_barrier_control.o: musa_named_barrier_repro.cu | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -I$(DEVICE_INCLUDE) \
		-DFLAGCX_MUSA_NAMED_BARRIER_REPRO=0 -o $@ $<

$(BUILD_DIR)/musa_named_barrier_repro.o: musa_named_barrier_repro.cu | $(BUILD_DIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -I$(DEVICE_INCLUDE) \
		-DFLAGCX_MUSA_NAMED_BARRIER_REPRO=1 -o $@ $<

run: all
	$(HOST_LAYOUT)
	$(PROBE)

run-groups: all
	$(HOST_LAYOUT)
	$(PROBE) --groups-only

run-atomics: all
	$(HOST_LAYOUT)
	$(PROBE) --atomics-only

run-cross-block: all
	$(HOST_LAYOUT)
	$(PROBE) --cross-block-only

clean:
	$(RM) $(BUILD_DIR)/musa_stream_value64
	$(RM) $(BUILD_DIR)/musa_platform_probe.o $(PROBE) $(HOST_LAYOUT) \
		$(PROBE_DEVICE_OBJ) $(PROBE_DEVICE_IMAGE) $(PROBE_HOST_OBJ) \
		$(PROBE_DEVICE_OBJ:.o=.d) $(PROBE_HOST_OBJ:.o=.d) \
		$(RESOURCE_PROBE) $(RESOURCE_DEVICE_OBJ) $(RESOURCE_DEVICE_IMAGE) \
		$(RESOURCE_HOST_OBJ) $(RESOURCE_DEVICE_OBJ:.o=.d) $(RESOURCE_HOST_OBJ:.o=.d) \
		$(BUILD_DIR)/musa_platform_probe.d $(BUILD_DIR)/musa_host_layout.d \
		$(BUILD_DIR)/musa_barrier_control.o $(BUILD_DIR)/musa_barrier_control.d \
		$(BUILD_DIR)/musa_named_barrier_repro.o \
		$(BUILD_DIR)/musa_named_barrier_repro.d \
		$(BUILD_DIR)/musa_native_trap_repro.o $(BUILD_DIR)/musa_native_trap_repro.d

-include $(BUILD_DIR)/musa_platform_probe.d $(BUILD_DIR)/musa_host_layout.d
-include $(PROBE_DEVICE_OBJ:.o=.d) $(PROBE_HOST_OBJ:.o=.d)
-include $(RESOURCE_DEVICE_OBJ:.o=.d) $(RESOURCE_HOST_OBJ:.o=.d)
-include $(BUILD_DIR)/musa_barrier_control.d $(BUILD_DIR)/musa_named_barrier_repro.d
-include $(BUILD_DIR)/musa_native_trap_repro.d
