# Copyright (c) 2026 BAAI. All rights reserved.
# Modified by 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved
# Modified by 2025 DU. All Rights Reserved.
BUILDDIR ?= $(abspath ./build)

# set to 0 if not provided
USE_NVIDIA ?= 0
USE_ASCEND ?= 0
USE_ILUVATAR_COREX ?= 0
USE_CAMBRICON ?= 0
USE_GLOO ?= 0
USE_BOOTSTRAP ?= 0
USE_METAX ?= 0
USE_MUSA ?= 0
USE_KUNLUNXIN ?=0
USE_AMD ?= 0
USE_DU ?= 0
USE_TSM ?= 0
USE_MPI ?= 0
USE_UCX ?= 0
USE_IBUC ?= 0
USE_ACCL_BAREX ?= 0
USE_ENFLAME ?= 0
USE_SUNRISE ?= 0
USE_PPU ?= 0
COMPILE_KERNEL ?= 0

# Device API backend selection
USE_SHMEM ?= 0
SHMEM_HOME ?= /usr/local/nvshmem

# set to empty if not provided
DEVICE_HOME ?=
CCL_HOME ?=
HOST_CCL_HOME ?=
MPI_HOME ?=
UCX_HOME ?=

ifeq ($(strip $(DEVICE_HOME)),)
	ifeq ($(USE_NVIDIA), 1)
		DEVICE_HOME = /usr/local/cuda
	else ifeq ($(USE_ASCEND), 1)
		DEVICE_HOME = /usr/local/Ascend/ascend-toolkit/latest
	else ifeq ($(USE_ILUVATAR_COREX), 1)
		DEVICE_HOME = /usr/local/corex
	else ifeq ($(USE_CAMBRICON), 1)
		DEVICE_HOME = $(NEUWARE_HOME)
	else ifeq ($(USE_METAX), 1)
		DEVICE_HOME = /opt/maca
	else ifeq ($(USE_MUSA), 1)
		DEVICE_HOME = /usr/local/musa
	else ifeq ($(USE_KUNLUNXIN), 1)
		DEVICE_HOME = /usr/local/xpu
	else ifeq ($(USE_DU), 1)
		DEVICE_HOME = ${CUDA_PATH}
	else ifeq ($(USE_AMD), 1)
		DEVICE_HOME = /opt/rocm
	else ifeq ($(USE_TSM), 1)
		DEVICE_HOME = /usr/local/kuiper
	else ifeq ($(USE_ENFLAME), 1)
		DEVICE_HOME = /opt/tops/
	else ifeq ($(USE_SUNRISE), 1)
		DEVICE_HOME = /usr/local/tangrt
	else ifeq ($(USE_PPU), 1)
		DEVICE_HOME = /usr/local/cuda
	else
		DEVICE_HOME = /usr/local/cuda
	endif
endif

ifeq ($(strip $(CCL_HOME)),)
	ifeq ($(USE_NVIDIA), 1)
		CCL_HOME = /usr/local/nccl/build
	else ifeq ($(USE_ASCEND), 1)
		CCL_HOME = /usr/local/Ascend/ascend-toolkit/latest
	else ifeq ($(USE_ILUVATAR_COREX), 1)
		CCL_HOME = /usr/local/corex
	else ifeq ($(USE_CAMBRICON), 1)
		CCL_HOME = $(NEUWARE_HOME)
	else ifeq ($(USE_METAX), 1)
		CCL_HOME = /opt/maca
	else ifeq ($(USE_MUSA), 1)
		CCL_HOME = /usr/local/musa
	else ifeq ($(USE_KUNLUNXIN), 1)
		CCL_HOME = /usr/local/xccl
	else ifeq ($(USE_DU), 1)
		CCL_HOME = ${CUDA_PATH}
	else ifeq ($(USE_AMD), 1)
		CCL_HOME = /opt/rocm
	else ifeq ($(USE_TSM), 1)
		CCL_HOME = /usr/local/kuiper
	else ifeq ($(USE_ENFLAME), 1)
		CCL_HOME = /usr
	else ifeq ($(USE_SUNRISE), 1)
		CCL_HOME = /usr/local/pccl
	else ifeq ($(USE_PPU), 1)
		CCL_HOME = /usr/local/cuda
	else
		CCL_HOME = /usr/local/nccl/build
	endif
endif

ifeq ($(strip $(HOST_CCL_HOME)),)
	ifeq ($(USE_GLOO), 1)
		HOST_CCL_HOME = /usr/local
	else ifeq ($(USE_MPI), 1)
		HOST_CCL_HOME = $(MPI_HOME)
	else
		HOST_CCL_HOME = 
	endif
endif

ifeq ($(strip $(MPI_HOME)),)
	ifeq ($(USE_MPI), 1)
		MPI_HOME = /usr/local
	endif
endif

ifeq ($(strip $(UCX_HOME)),)
	ifeq ($(USE_UCX), 1)
		UCX_HOME = /usr/local/ucx
	endif
endif

DEVICE_LIB =
DEVICE_INCLUDE =
DEVICE_LINK =
DEVICE_PLATFORM =
DEVICE_COMPILER =
DEVICE_COMPILE_FLAG =
DEVICE_LINK_FLAG =
DEVICE_FILE_EXTENSION =
CCL_LIB =
CCL_INCLUDE =
CCL_LINK =
HOST_CCL_LIB = 
HOST_CCL_INCLUDE =
HOST_CCL_LINK =
ADAPTOR_FLAG =
HOST_CCL_ADAPTOR_FLAG =
UCX_LIB =
UCX_INCLUDE =
UCX_LINK =
NET_ADAPTOR_FLAG =
COMPILE_KERNEL_HOST_FLAG=
COMPILE_KERNEL_FLAG =
HOST_COMPILER ?= g++
ifeq ($(USE_NVIDIA), 1)
  include makefiles/nvidia.mk
else ifeq ($(USE_ASCEND), 1)
  include makefiles/ascend.mk
else ifeq ($(USE_ILUVATAR_COREX), 1)
  include makefiles/iluvatar_corex.mk
else ifeq ($(USE_CAMBRICON), 1)
  include makefiles/cambricon.mk
else ifeq ($(USE_METAX), 1)
  include makefiles/metax.mk
else ifeq ($(USE_MUSA), 1)
  include makefiles/musa.mk
else ifeq ($(USE_KUNLUNXIN), 1)
  include makefiles/kunlunxin.mk
else ifeq ($(USE_DU), 1)
  include makefiles/du.mk
else ifeq ($(USE_AMD), 1)
  include makefiles/amd.mk
else ifeq ($(USE_TSM), 1)
  include makefiles/tsm.mk
else ifeq ($(USE_ENFLAME), 1)
  include makefiles/enflame.mk
else ifeq ($(USE_SUNRISE), 1)
  include makefiles/sunrise.mk
else ifeq ($(USE_PPU), 1)
  include makefiles/ppu.mk
else
  include makefiles/nvidia.mk
  USE_NVIDIA := 1
endif

ifeq ($(USE_GLOO), 1)
	HOST_CCL_LIB = $(HOST_CCL_HOME)/lib
	HOST_CCL_INCLUDE = $(HOST_CCL_HOME)/include
	HOST_CCL_LINK = -lgloo -libverbs
	HOST_CCL_ADAPTOR_FLAG = -DUSE_GLOO_ADAPTOR
else ifeq ($(USE_MPI), 1)
	HOST_CCL_LIB = $(MPI_HOME)/lib
	HOST_CCL_INCLUDE = $(MPI_HOME)/include
	HOST_CCL_LINK = -lmpi
	HOST_CCL_ADAPTOR_FLAG = -DUSE_MPI_ADAPTOR
else ifeq ($(USE_BOOTSTRAP), 1)
	HOST_CCL_LIB = /usr/local/lib
	HOST_CCL_INCLUDE = /usr/local/include
	HOST_CCL_LINK = 
	HOST_CCL_ADAPTOR_FLAG = -DUSE_BOOTSTRAP_ADAPTOR
else
	HOST_CCL_LIB = /usr/local/lib
	HOST_CCL_INCLUDE = /usr/local/include
	HOST_CCL_LINK = 
	HOST_CCL_ADAPTOR_FLAG = -DUSE_BOOTSTRAP_ADAPTOR
endif

# UCX network adaptor configuration
ifeq ($(USE_UCX), 1)
	UCX_LIB = $(UCX_HOME)/lib
	UCX_INCLUDE = $(UCX_HOME)/include
	UCX_LINK = -lucp -lucs -luct
	NET_ADAPTOR_FLAG = -DUSE_UCX
else
	UCX_LIB = $(UCX_HOME)/lib
	UCX_INCLUDE = $(UCX_HOME)/include
	UCX_LINK = 
	NET_ADAPTOR_FLAG = 
endif

# IBUC network adaptor configuration
ifeq ($(USE_IBUC), 1)
	NET_ADAPTOR_FLAG += -DUSE_IBUC
endif

# ACCL (accl::barex) transport for PPU + vsolar hosts lacking peer-mem/DMA-BUF.
# Devel package installs under /usr. This build option selects BAREX through
# the common net-adaptor path; the P2P engine itself remains transport-neutral.
ifeq ($(USE_ACCL_BAREX), 1)
	ACCL_BAREX_HOME ?= /usr
	ACCL_BAREX_INCLUDE = $(ACCL_BAREX_HOME)/include
	ACCL_BAREX_LIB = $(ACCL_BAREX_HOME)/lib
	ACCL_BAREX_LINK = -laccl_barex
	NET_ADAPTOR_FLAG += -DUSE_ACCL_BAREX -DCMAKE_INCLUDE=1
else
	ACCL_BAREX_INCLUDE = .
	ACCL_BAREX_LIB = .
	ACCL_BAREX_LINK =
endif

ifeq ($(COMPILE_KERNEL), 1)
	COMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL
	COMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST
endif

LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
BUILD_INCDIR := $(BUILDDIR)/include
PREFIX ?= /usr/local
DESTDIR  ?= $(PREFIX)/lib
INC_DESTDIR ?= $(PREFIX)/include
# Source builds use the submodule; distro builds override this with /usr/include.
JSON_INCLUDE_DIR ?= $(abspath third-party/json/single_include)

# Public headers exported alongside libflagcx.so
PUBLIC_HEADERS := \
	flagcx/include/flagcx.h \
	flagcx/include/flagcx_kernel.h \
	flagcx/include/flagcx_p2p.h
BUILD_PUBLIC_HEADERS := $(PUBLIC_HEADERS:flagcx/include/%=$(BUILD_INCDIR)/%)

INCLUDEDIR := \
	$(abspath flagcx/include) \
	$(abspath flagcx/adaptor/include) \
	$(abspath flagcx/adaptor/device_api) \
	$(abspath flagcx/adaptor/shmem) \
	$(abspath flagcx/runner/include) \
	$(abspath flagcx/core/include) \
	$(abspath flagcx/service/include) \
	$(JSON_INCLUDE_DIR)

# Append NVSHMEM include path (must come after INCLUDEDIR := assignment)
ifeq ($(USE_SHMEM), 1)
INCLUDEDIR += $(SHMEM_HOME)/include
endif

LIBSRCFILES:= \
	$(wildcard flagcx/*.cc) \
	$(wildcard flagcx/adaptor/*.cc) \
	$(wildcard flagcx/adaptor/device/*.cc) \
	$(wildcard flagcx/adaptor/ccl/*.cc) \
	$(wildcard flagcx/adaptor/net/*.cc) \
	$(wildcard flagcx/adaptor/tuner/*.cc) \
	$(wildcard flagcx/runner/*.cc) \
	$(wildcard flagcx/core/*.cc) \
	$(wildcard flagcx/service/*.cc)

# Platform .mk provides extra sources (device_api backend, shmem adaptor)
LIBSRCFILES += $(PLATFORM_EXTRA_SRCS)

ifeq ($(COMPILE_KERNEL), 1)
DEVSRCFILES := $(PLATFORM_KERNEL_SRCS)
DEVOBJ := $(DEVSRCFILES:%.$(DEVICE_FILE_EXTENSION)=$(OBJDIR)/%.o)
endif
LIBOBJ := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)

TARGET = libflagcx.so
all: $(LIBDIR)/$(TARGET) $(BUILD_PUBLIC_HEADERS)

print_var:
	@echo "USE_KUNLUNXIN : $(USE_KUNLUNXIN)"
	@echo "DEVICE_HOME: $(DEVICE_HOME)"
	@echo "CCL_HOME: $(CCL_HOME)"
	@echo "HOST_CCL_HOME: $(HOST_CCL_HOME)"
	@echo "MPI_HOME: $(MPI_HOME)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
	@echo "USE_ASCEND: $(USE_ASCEND)"
	@echo "USE_ILUVATAR_COREX: $(USE_ILUVATAR_COREX)"
	@echo "USE_CAMBRICON: $(USE_CAMBRICON)"
	@echo "USE_KUNLUNXIN: $(USE_KUNLUNXIN)"
	@echo "USE_GLOO: $(USE_GLOO)"
	@echo "USE_MPI: $(USE_MPI)"
	@echo "USE_MUSA: $(USE_MUSA)"
	@echo "USE_DU: $(USE_DU)"
	@echo "USE_AMD: $(USE_AMD)"
	@echo "USE_TSM: $(USE_TSM)"
	@echo "USE_ENFLAME: $(USE_ENFLAME)"
	@echo "COMPILE_KERNEL: $(COMPILE_KERNEL)"
	@echo "DEVICE_LIB: $(DEVICE_LIB)"
	@echo "DEVICE_INCLUDE: $(DEVICE_INCLUDE)"
	@echo "CCL_LIB: $(CCL_LIB)"
	@echo "CCL_INCLUDE: $(CCL_INCLUDE)"
	@echo "HOST_CCL_LIB: $(HOST_CCL_LIB)"
	@echo "HOST_CCL_INCLUDE: $(HOST_CCL_INCLUDE)"
	@echo "ADAPTOR_FLAG: $(ADAPTOR_FLAG)"
	@echo "HOST_CCL_ADAPTOR_FLAG: $(HOST_CCL_ADAPTOR_FLAG)"
	@echo "USE_UCX: $(USE_UCX)"
	@echo "UCX_HOME: $(UCX_HOME)"
	@echo "UCX_LIB: $(UCX_LIB)"
	@echo "UCX_INCLUDE: $(UCX_INCLUDE)"
	@echo "USE_IBUC: $(USE_IBUC)"
	@echo "NET_ADAPTOR_FLAG: $(NET_ADAPTOR_FLAG)"
	@echo "DEVSRCFILES: $(DEVSRCFILES)"

ifeq ($(COMPILE_KERNEL), 1)
DEVOBJS = $(DEVOBJ) $(OBJDIR)/kernel_dlink.o
else
DEVOBJS =
endif

HOST_LINKER   ?= $(HOST_COMPILER)
DEVICE_LINKER := $(DEVICE_COMPILER)
LINKER        := $(HOST_LINKER)
ifeq ($(COMPILE_KERNEL)$(USE_DU),11)
  LINKER := $(DEVICE_LINKER)
endif

$(LIBDIR)/$(TARGET): $(LIBOBJ) $(DEVOBJS)
	@mkdir -p `dirname $@`
	@echo "Linking   $@"
	@$(LINKER) $^ -o $@ -L$(CCL_LIB) -L$(DEVICE_LIB) -L$(HOST_CCL_LIB) -L$(UCX_LIB) -L$(ACCL_BAREX_LIB) -shared -fvisibility=default -Wl,--no-as-needed -Wl,-rpath,$(LIBDIR) -Wl,-rpath,$(CCL_LIB) -Wl,-rpath,$(HOST_CCL_LIB) -Wl,-rpath,$(UCX_LIB) -Wl,-rpath,$(ACCL_BAREX_LIB) -lpthread -lrt -ldl $(CCL_LINK) $(DEVICE_LINK) $(HOST_CCL_LINK) $(UCX_LINK) $(ACCL_BAREX_LINK) -g

# Copy public headers from flagcx/include/ into the build output tree so they
# sit next to the shared libraries (build/include + build/lib).
$(BUILD_INCDIR)/%.h: flagcx/include/%.h
	@mkdir -p `dirname $@`
	@echo "Copying   $@"
	@cp $< $@

$(OBJDIR)/%.o: %.cc
	@mkdir -p `dirname $@`
	@echo "Compiling $@"
	@$(HOST_COMPILER) $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) -I$(ACCL_BAREX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(COMPILE_KERNEL_HOST_FLAG) -c -fPIC -fvisibility=default -Wvla -Wno-unused-function -Wno-sign-compare -Wall -MMD -MP -g

ifeq ($(COMPILE_KERNEL), 1)
$(OBJDIR)/kernel_dlink.o: $(DEVOBJ)
	@$(DEVICE_LINKER) -dlink $^ -o $@ $(DEVICE_LINK) $(DEVICE_LINK_FLAG)

$(OBJDIR)/%.o: %.$(DEVICE_FILE_EXTENSION)
	@mkdir -p `dirname $@`
	@echo "Compiling $@ ($(DEVICE_PLATFORM))"
	@$(DEVICE_COMPILER) $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(DEVICE_COMPILE_FLAG) $(COMPILE_KERNEL_FLAG) -g
endif

ifeq ($(COMPILE_KERNEL), 1)
-include $(LIBOBJ:.o=.d) $(DEVOBJ:.o=.d)
else
-include $(LIBOBJ:.o=.d)
endif

INSTALLDIR := /usr/local/lib
install:
	@mkdir -p $(DESTDIR)
	@cp $(LIBDIR)/$(TARGET) $(DESTDIR)/$(TARGET)
	@mkdir -p $(INC_DESTDIR)
	@cp $(PUBLIC_HEADERS) $(INC_DESTDIR)/

clean:
	@rm -rf $(LIBDIR)/$(TARGET) $(DESTDIR)/$(TARGET) $(BUILD_INCDIR) $(OBJDIR)
