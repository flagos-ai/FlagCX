include ../../make.inc

ifneq ($(USE_METAX),1)
$(error rdma_atomic_probe is a MetaX-only diagnostic)
endif

.DEFAULT_GOAL := all
TARGET := $(BINDIR)/rdma_atomic_probe
SOURCES := main.cpp native.cpp adaptor.cpp
OBJECTS := $(SOURCES:%.cpp=$(OBJDIR)/%.o)
CONFIG := $(BUILDDIR)/sdk/allocation_modes.h

.PHONY: all clean FORCE
all: $(TARGET)

FORCE:

$(CONFIG): configure.py FORCE
	python3 $< --cxx '$(CXX)' --sdk '$(DEVICE_HOME)' --output '$(BUILDDIR)/sdk'

$(OBJDIR)/%.o: %.cpp probe.h $(CONFIG)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(HOST_ADAPTOR_DEFINE) $(FLAGCX_INCLUDES) \
		$(MPI_INCLUDE) $(DEVICE_INCLUDE) -I$(BUILDDIR)/sdk -MMD -MP -c $< -o $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) $^ -o $@ $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK) \
		-L$(DEVICE_HOME)/lib -Wl,-rpath,$(DEVICE_HOME)/lib \
		-Wl,-rpath,$(DEVICE_HOME)/lib64 -libverbs -lpthread -ldl

clean:
	rm -rf $(BUILDDIR)

-include $(OBJECTS:.o=.d)
