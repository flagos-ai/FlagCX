# test/kernel/Makefile
# Dispatches kernel compilation to the active platform subdirectory.

include ../make.inc

KERNEL_PLATFORM_DIR := $(CURDIR)/$(KERNEL_PLATFORM)

# Known targets that consumers request
FORWARD_TARGETS := device_api.o device_ir.o nvshmem_adaptor_kernel.o \
    dlink.o dlink_api.o dlink_ir.o dlink_shmem_test.o

.PHONY: all clean $(FORWARD_TARGETS)

all:
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) all \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME) \
		FORCE_DEFAULT_PATH=$(FORCE_DEFAULT_PATH)

ifeq ($(strip $(KERNEL_PLATFORM)),)
clean:
	@$(MAKE) -C $(CURDIR)/nvidia clean
	@$(MAKE) -C $(CURDIR)/du clean
else
clean:
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) clean
endif

$(FORWARD_TARGETS):
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) $@ \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME) \
		FORCE_DEFAULT_PATH=$(FORCE_DEFAULT_PATH)
