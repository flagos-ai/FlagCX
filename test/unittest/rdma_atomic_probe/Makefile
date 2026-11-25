include ../../make.inc

ifneq ($(USE_METAX),1)
$(error rdma_atomic_probe is a MetaX-only diagnostic)
endif

.DEFAULT_GOAL := all
NATIVE := $(BINDIR)/rdma_atomic_native_probe
ADAPTOR := $(BINDIR)/rdma_atomic_adaptor_probe
COMMON := $(OBJDIR)/common/inventory.o $(OBJDIR)/common/gdr.o
NATIVE_OBJS := $(OBJDIR)/native/main.o $(OBJDIR)/native/native.o $(COMMON)
ADAPTOR_OBJS := $(OBJDIR)/adaptor/main.o $(OBJDIR)/adaptor/adaptor.o $(COMMON)
CONFIG := $(BUILDDIR)/sdk/allocation_modes.h
INCLUDES := $(HOST_ADAPTOR_DEFINE) $(FLAGCX_INCLUDES) $(MPI_INCLUDE) \
            $(DEVICE_INCLUDE) -I$(BUILDDIR)/sdk
LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK) -L$(DEVICE_HOME)/lib \
        -Wl,-rpath,$(DEVICE_HOME)/lib -Wl,-rpath,$(DEVICE_HOME)/lib64 -lpthread -ldl

.PHONY: all native adaptor clean FORCE
all: native adaptor
native: $(NATIVE)
adaptor: $(ADAPTOR)

FORCE:

$(CONFIG): configure.py FORCE
	python3 $< --cxx '$(CXX)' --sdk '$(DEVICE_HOME)' --output '$(BUILDDIR)/sdk'

$(OBJDIR)/native/%.o: %.cpp probe.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DFLAGCX_PROBE_NATIVE -MMD -MP -c $< -o $@

$(OBJDIR)/adaptor/%.o: %.cpp probe.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR)/common/%.o: %.cpp probe.h $(CONFIG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(NATIVE): $(NATIVE_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LINK) -libverbs

$(ADAPTOR): $(ADAPTOR_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LINK)

clean:
	rm -rf $(BUILDDIR)

-include $(sort $(NATIVE_OBJS:.o=.d) $(ADAPTOR_OBJS:.o=.d))
