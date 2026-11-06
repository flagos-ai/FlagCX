# makefiles/generic.mk
# Vendor-neutral platform configuration.
#
# Builds a libflagcx.so that links against no vendor SDK at all: the device
# and CCL adaptors are supplied at run time by adaptor plugins (see
# adaptor_plugin/). One such library is installable on every supported
# accelerator; the vendor-specific code ships as a separate plugin .so.
#
# Because no device compiler is available on this path, device kernels cannot
# be built here. COMPILE_KERNEL=1 therefore has to stay with the per-vendor
# platform makefiles.

ifeq ($(COMPILE_KERNEL), 1)
  $(error COMPILE_KERNEL=1 is unsupported with USE_GENERIC=1: device kernels \
          need a vendor device compiler. Build a per-vendor target instead.)
endif

DEVICE_PLATFORM := GENERIC

# No vendor SDK: leave every device/CCL discovery variable empty so that
# neither an include path nor a -l/-L/-rpath entry for a vendor library can
# reach the compile or link line.
DEVICE_LIB     :=
DEVICE_INCLUDE :=
DEVICE_LINK    :=
CCL_LIB        :=
CCL_INCLUDE    :=
CCL_LINK       :=

# Defining no USE_<VENDOR>_ADAPTOR selects the vendor-neutral fallbacks that
# platform_traits.h and comm_traits.h already provide in their #else branch.
ADAPTOR_FLAG := -DUSE_GENERIC_ADAPTOR

PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc

# No device sources on this path.
PLATFORM_KERNEL_SRCS :=
DEVICE_NEEDS_DLINK := 0
