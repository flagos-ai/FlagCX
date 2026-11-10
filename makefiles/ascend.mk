# makefiles/platforms/ascend.mk
# Huawei Ascend platform configuration.

DEVICE_HOME  ?= /usr/local/Ascend/ascend-toolkit/latest
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_LINK  := -lascendcl
DEVICE_PLATFORM := ASCEND

# bisheng ships under the host-arch directory of the toolkit, and the arch name
# differs between x86_64 and aarch64 installs (the 910C machine used for
# verification is aarch64). Resolve whichever exists, keeping the historical
# compiler/bin spelling as the last candidate.
ASCEND_BISHENG ?= $(firstword $(wildcard \
    $(DEVICE_HOME)/aarch64-linux/bin/bisheng \
    $(DEVICE_HOME)/x86_64-linux/bin/bisheng \
    $(DEVICE_HOME)/compiler/bin/bisheng))
DEVICE_COMPILER := $(ASCEND_BISHENG)

# Device architecture selector. Verified on Ascend 910C (CANN 9.1.0-beta.3):
#   bisheng -x asc --npu-soc=Ascend910_9382 --cce-auto-infer-kernel-type=false \
#       -c probe.asc -o probe.o            -> rc=0
# --npu-soc takes the name reported by the runtime itself (aclrtGetSocName());
# --npu-arch and --cce-soc-version both reject every spelling tried on this
# toolchain, so the soc form is the only supported entry point here.
ASCEND_SOC ?= Ascend910_9382

# ld.lld cannot derive a kernel's AI-Cube/AI-Vector type for kernels that do not
# move data, which fails the link step of an otherwise valid translation unit.
# Disabling the auto-inference is what makes a device translation unit compile
# (rc=0) instead of erroring out in the linker.
# -x asc is deliberately absent. The rule expands these flags AFTER the source
# file (Makefile:546), where the driver reports that an input following the
# last input file has no effect, so the language already comes from the .asc
# extension (DEVICE_FILE_EXTENSION below). Verified rather than assumed: the
# object this rule produces carries an .aicore_binary section, i.e. the device
# image really is embedded even without the flag.
DEVICE_COMPILE_FLAG := -c --npu-soc=$(ASCEND_SOC) \
    --cce-auto-infer-kernel-type=false -std=c++17 -O2 -fPIC -MMD -MP \
    -DFLAGCX_ASCEND_DEVICE_COMPILE=1
DEVICE_LINK_FLAG :=
# Each CCE object carries its own device image (as on DU/Iluvatar/KLX, unlike
# nvcc's separate -dlink step), so no device-link object is produced.
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := asc

# Ascend C headers were verified at
#   $(DEVICE_HOME)/aarch64-linux/include/ascendc/basic_api/kernel_operator.h
# on the 910C machine; some installs expose the same tree under include/ascendc.
# Add whichever exists so a toolkit without Ascend C still configures for the
# host-only build.
DEVICE_INCLUDE := $(DEVICE_HOME)/include \
    $(firstword $(wildcard $(DEVICE_HOME)/aarch64-linux/include/ascendc) \
                $(wildcard $(DEVICE_HOME)/include/ascendc) \
                $(wildcard $(DEVICE_HOME)/compiler/include/ascendc))

CCL_HOME    ?= /usr/local/Ascend/ascend-toolkit/latest
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lhccl
ADAPTOR_FLAG := -DUSE_ASCEND_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/ascend
PLATFORM_KERNEL_SRCS := \
    $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
