# test/kernel/klx/Makefile
# Kunlunxin (P800) test kernel compilation.
#
# device_api.xpu exercises FlagCX's xSHMEM CommTraits through the public Device
# API. It is compiled by the XTDK clang in a single relocatable object that
# device-links libxshmem_device.a inline (-xpu-l xshmem_device), so the
# resulting device_api.o is host-linkable by g++ into test binaries.

include ../../make.inc

# Toolchain / SDK roots (symlinks under /usr/local point at the SDKs).
XTDK_HOME ?= /workspace/my_flagcx/xtdk-llvm15-ubuntu2004_x86_64
XCCL_HOME ?= /usr/local/xccl
XRE_HOME  ?= /usr/local/xpu
XPU_CLANG := $(XTDK_HOME)/bin/clang++

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    -I$(XCCL_HOME)/include \
    -I$(XRE_HOME)/include

# xpu3 = P800. c++17 for the FlagCX headers. Device-link the xshmem static lib
# so the __global__ kernel's xshmem device symbols resolve at compile time.
XPU_CFLAGS := --xpu-arch=xpu3 -std=c++17 -O2 -fno-builtin \
    --target=x86_64-linux-gnu -DUSE_KUNLUNXIN_ADAPTOR -DCOMPILE_KERNEL
XPU_DEVLINK := -xpu-L $(XCCL_HOME)/lib -xpu-l xshmem_device

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o

ifneq ($(USE_SHMEM),1)
device_api.o:
	@echo "KLX: device_api.xpu is the XSHMEM Device API implementation and requires USE_SHMEM=1." >&2
	@false
else
XPU_CFLAGS += -DFLAGCX_COMM_TRAITS_SHMEM
all: $(OBJDIR)/device_api.o

device_api.o: $(OBJDIR)/device_api.o

$(OBJDIR)/device_api.o: device_api.xpu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.xpu (KLX / FlagCX xshmem Device API)"
	@$(XPU_CLANG) $(XPU_CFLAGS) $(EXTRA_XPU_FLAGS) $(ALL_INCLUDES) $(XPU_DEVLINK) -c -o $@ $<
endif

# The xshmem Device API path device-links inline (above), so there is no separate
# device-link object here. Fail LOUDLY instead of pretending to succeed: a
# silent no-op used to leave the caller linking a file that was never created.
# Consumers must filter these targets out on KLX (see test/unittest/device_api).
device_ir.o dlink.o dlink_api.o dlink_ir.o:
	@echo "KLX: '$@' is not available (device_ir has no .xpu counterpart; xshmem is device-linked into device_api.o). Filter this target out for KLX." >&2
	@false

clean:
	@rm -rf $(BUILDDIR)
