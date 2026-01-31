# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# 2025 - Modified by DU. All Rights Reserved.
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
USE_TOPS ?= 0
COMPILE_KERNEL ?= 0

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
	else ifeq ($(USE_TOPS), 1)
		DEVICE_HOME = /opt/tops/
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
	else ifeq ($(USE_TOPS), 1)
		CCL_HOME = /usr
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
DEVICE_RUNTIME =
DEVICE_COMPILER =
DEVICE_LINKER =
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
ifeq ($(USE_NVIDIA), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_RUNTIME = CUDA
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
	DEVICE_LINKER = $(DEVICE_HOME)/bin/nvcc -dlink
	DEVICE_COMPILE_FLAG = -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g
	DEVICE_LINK_FLAG = --cudart=shared -Xcompiler -fPIC
	DEVICE_FILE_EXTENSION = cu
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_NVIDIA_ADAPTOR
else ifeq ($(USE_ASCEND), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lascendcl
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lhccl
	ADAPTOR_FLAG = -DUSE_ASCEND_ADAPTOR
else ifeq ($(USE_ILUVATAR_COREX), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_ILUVATAR_COREX_ADAPTOR
else ifeq ($(USE_CAMBRICON), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcnrt
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lcncl
	ADAPTOR_FLAG = -DUSE_CAMBRICON_ADAPTOR
else ifeq ($(USE_METAX), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lmccl
	ADAPTOR_FLAG = -DUSE_METAX_ADAPTOR
else ifeq ($(USE_MUSA), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lmccl -lmusa
	ADAPTOR_FLAG = -DUSE_MUSA_ADAPTOR
else ifeq ($(USE_KUNLUNXIN), 1)
	DEVICE_LIB = $(DEVICE_HOME)/so
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lxpurt -lcudart
	CCL_LIB = $(CCL_HOME)/so
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lbkcl
	ADAPTOR_FLAG = -DUSE_KUNLUNXIN_ADAPTOR
else ifeq ($(USE_DU), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_DU_ADAPTOR
else ifeq ($(USE_AMD), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lhiprtc
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include/rccl
	CCL_LINK = -lrccl
	ADAPTOR_FLAG = -DUSE_AMD_ADAPTOR -D__HIP_PLATFORM_AMD__
else ifeq ($(USE_TSM), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lhpgr
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -ltccl
	ADAPTOR_FLAG = -DUSE_TSM_ADAPTOR
else ifeq ($(USE_TOPS), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -ltopsrt
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -leccl
	ADAPTOR_FLAG = -DUSE_TOPS_ADAPTOR
else
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_NVIDIA_ADAPTOR
endif

ifeq ($(USE_GLOO), 1)
	HOST_CCL_LIB = $(HOST_CCL_HOME)/lib
	HOST_CCL_INCLUDE = $(HOST_CCL_HOME)/include
	HOST_CCL_LINK = -lgloo
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

ifeq ($(COMPILE_KERNEL), 1)
	COMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL
	COMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST
endif

LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PREFIX ?= /usr/local
DESTDIR  ?= $(PREFIX)/lib

INCLUDEDIR := \
	$(abspath flagcx/include) \
	$(abspath flagcx/adaptor/include) \
	$(abspath flagcx/runner/include) \
	$(abspath flagcx/core/include) \
	$(abspath flagcx/service/include) \
	$(abspath third-party/json/single_include)

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

ifeq ($(COMPILE_KERNEL), 1)
DEVSRCFILES:= \
	$(wildcard flagcx/kernels/*.$(DEVICE_FILE_EXTENSION))
DEVOBJ:= $(DEVSRCFILES:%.$(DEVICE_FILE_EXTENSION)=$(OBJDIR)/%.o)
endif
LIBOBJ:= $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)

TARGET = libflagcx.so
all: $(LIBDIR)/$(TARGET)

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
	@echo "USE_TOPS: $(USE_TOPS)"
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

ifeq ($(COMPILE_KERNEL), 1)
DEVOBJS = $(DEVOBJ) $(OBJDIR)/kernel_dlink.o
else
DEVOBJS =
endif

$(LIBDIR)/$(TARGET): $(LIBOBJ) $(DEVOBJS)
	@mkdir -p `dirname $@`
	@echo "Linking   $@"
	@g++ $^ -o $@ -L$(CCL_LIB) -L$(DEVICE_LIB) -L$(HOST_CCL_LIB) -L$(UCX_LIB) -shared -fvisibility=default -Wl,--no-as-needed -Wl,-rpath,$(LIBDIR) -Wl,-rpath,$(CCL_LIB) -Wl,-rpath,$(HOST_CCL_LIB) -Wl,-rpath,$(UCX_LIB) -lpthread -lrt -ldl $(CCL_LINK) $(DEVICE_LINK) $(HOST_CCL_LINK) $(UCX_LINK) -g

$(OBJDIR)/%.o: %.cc
	@mkdir -p `dirname $@`
	@echo "Compiling $@"
	@g++ $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) -I$(DEVICE_INCLUDE) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(COMPILE_KERNEL_HOST_FLAG) -c -fPIC -fvisibility=default -Wvla -Wno-unused-function -Wno-sign-compare -Wall -MMD -MP -g

ifeq ($(COMPILE_KERNEL), 1)
$(OBJDIR)/kernel_dlink.o: $(DEVOBJ)
	@$(DEVICE_LINKER) $^ -o $@ $(DEVICE_LINK) $(DEVICE_LINK_FLAG)

$(OBJDIR)/%.o: %.$(DEVICE_FILE_EXTENSION)
	@mkdir -p `dirname $@`
	@echo "Compiling $@ ($(DEVICE_RUNTIME))"
	@$(DEVICE_COMPILER) $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) -I$(DEVICE_INCLUDE) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(DEVICE_COMPILE_FLAG) $(COMPILE_KERNEL_FLAG) -g
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

clean:
	@rm -rf $(LIBDIR)/$(TARGET) $(DESTDIR)/$(TARGET) $(OBJDIR)
