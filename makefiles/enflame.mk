# makefiles/platforms/enflame.mk
# Enflame platform configuration.

# No trailing slash: DEVICE_INCLUDE is expanded with -I... and a trailing
# slash would yield /opt/tops//include.
DEVICE_HOME  ?= /opt/tops
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -ltopsrt
# DEVICE_PLATFORM is not the accelerator's marketing name — it is the selector
# `test/make.inc` lowercases to locate per-platform test artefacts:
#   KERNEL_PLATFORM   := $(lowercase DEVICE_PLATFORM) -> test/kernel/enflame/
#   PLATFORM_IR_DIR   := $(PROJECT_ROOT)/bindings/ir/$(lowercase ...)
# (KLX does the same thing on purpose: DEVICE_PLATFORM := KLX lands its test
# kernels in test/kernel/klx/.) So this must read ENFLAME, matching
# test/kernel/enflame/ and bindings/ir/enflame/, not TOPS.
DEVICE_PLATFORM := ENFLAME
DEVICE_COMPILER := /usr/bin/topscc

TOPSCC_ARCH ?= gcu400 gcu410
TOPSCC_ARCH_FLAG := $(addprefix -arch ,$(TOPSCC_ARCH))

DEVICE_COMPILE_FLAG := -c -std=c++17 -O3 -fPIC -MMD -MP $(TOPSCC_ARCH_FLAG)
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION := tops

CCL_HOME    ?= /usr
CCL_LIB     := $(CCL_HOME)/lib
# No literal -I here: test/make.inc does `CCL_INCLUDE := -I$(CCL_INCLUDE)`, one
# prefix for the whole value. Writing -I in this file yields `-I-I/usr/include`.
#
# ECCL's deb installs everything the device pass needs into /usr/include:
#   eccl.h, eccl_ext.h, eccl_device.h, eccl_device/{coop,core,gin,...}.h,
#   eccl_device/impl/*__funcs.h, eccl_device/gin/**  and  tops/atomic
# The last one is ECCL's own compatibility shim for tops::atomic_ref — the TOPS
# SDK ships no tops/atomic, and eccl_device/utility.h does `#include <tops/atomic>`
# unconditionally. /usr/include is already a topscc default search root, so no
# extra -I is required. Keep this value a bare path: a second directory would be
# silently dropped, because only one -I is prefixed onto the whole value.
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -leccl
ADAPTOR_FLAG := -DUSE_ENFLAME_ADAPTOR

# --- Device API backend selection ---
# ECCL is the only backend Enflame supports for now
ifeq ($(USE_SHMEM), 1)
  $(warning USE_SHMEM=1 is not supported on Enflame and fallback to ECCL backend)
endif
ifeq ($(FORCE_DEFAULT_PATH), 1)
  $(warning FORCE_DEFAULT_PATH=1 is not supported on Enflame and fallback to ECCL backend)
endif

ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_CCL
PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/eccl_dev_api_backend.cc

DEVICE_NEEDS_DLINK := 0

# --- Kernel sources ---
PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/enflame
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
