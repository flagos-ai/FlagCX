include ../../make.inc

ifneq ($(USE_METAX),1)
$(error rdma_atomic_probe is a MetaX-only diagnostic)
endif

.DEFAULT_GOAL := all
NATIVE := $(BINDIR)/rdma_atomic_native_probe
ADAPTOR := $(BINDIR)/rdma_atomic_adaptor_probe
SDK_INVENTORY := $(BINDIR)/rdma_atomic_sdk_inventory
COMMON := $(OBJDIR)/common/inventory.o $(OBJDIR)/common/gdr.o
NATIVE_OBJS := $(OBJDIR)/native/main.o $(OBJDIR)/native/native.o $(COMMON)
ADAPTOR_OBJS := $(OBJDIR)/adaptor/main.o $(OBJDIR)/adaptor/adaptor.o $(COMMON)
CONFIG := $(BUILDDIR)/sdk/allocation_modes.h
INCLUDES := $(HOST_ADAPTOR_DEFINE) $(FLAGCX_INCLUDES) $(MPI_INCLUDE) \
            $(DEVICE_INCLUDE) -I$(BUILDDIR)/sdk
LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK) -L$(DEVICE_HOME)/lib \
        -Wl,-rpath,$(DEVICE_HOME)/lib -Wl,-rpath,$(DEVICE_HOME)/lib64 -lpthread -ldl

.PHONY: all native adaptor sdk-inventory clean FORCE
all: native adaptor sdk-inventory
native: $(NATIVE)
adaptor: $(ADAPTOR)
sdk-inventory: $(SDK_INVENTORY)

FORCE:

$(CONFIG): configure.py FORCE
	python3 $< --cxx '$(CXX)' --sdk '$(DEVICE_HOME)' --output '$(BUILDDIR)/sdk'

$(OBJDIR)/native/%.o: %.cpp probe.h diagnostics.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DFLAGCX_PROBE_NATIVE -MMD -MP -c $< -o $@

$(OBJDIR)/adaptor/%.o: %.cpp probe.h diagnostics.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR)/common/%.o: %.cpp probe.h diagnostics.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(NATIVE): $(NATIVE_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LINK) -libverbs

$(ADAPTOR): $(ADAPTOR_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LINK)

# Do not use LINK here: the control must not load FlagCX, MPI or libibverbs.
# It also needs no allocation-mode configure step.
$(SDK_INVENTORY): sdk_inventory.cpp diagnostics.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEVICE_INCLUDE) -I$(PROJECT_ROOT)/third-party/json/single_include \
		$< -o $@ $(DEVICE_LINK) -L$(DEVICE_HOME)/lib \
		-Wl,-rpath,$(DEVICE_HOME)/lib -Wl,-rpath,$(DEVICE_HOME)/lib64 -lpthread -ldl

clean:
	rm -rf $(BUILDDIR)

-include $(sort $(NATIVE_OBJS:.o=.d) $(ADAPTOR_OBJS:.o=.d))
