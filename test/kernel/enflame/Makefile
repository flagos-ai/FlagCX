# test/kernel/enflame/Makefile
# Enflame GCU/TOPS kernel compilation for test binaries.
#
# Mirrors test/kernel/nvidia/Makefile. The differences are all consequences of
# topscc replacing nvcc:
#   * sources are .tops (makefiles/enflame.mk: DEVICE_FILE_EXTENSION := tops)
#   * there is no DEVICE_COMPILER_GENCODE — the arch flags live in
#     DEVICE_COMPILE_FLAG (`-arch gcu400 -arch gcu410`)
#   * there is no device-link step, so the dlink* targets stay unimplemented

include ../../make.inc

# FORCE_DEFAULT_PATH is deliberately NOT propagated here. makefiles/enflame.mk
# warns that the request is unsupported and keeps the ECCL backend
# (-DFLAGCX_COMM_TRAITS_CCL in ADAPTOR_FLAG), so adding
# -DFLAGCX_FORCE_DEFAULT_PATH would contradict the traits actually compiled.

PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    $(PLATFORM_INCLUDES) \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE) \
    $(SHMEM_INCLUDE)

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o \
        dlink_shmem_test.o nvshmem_adaptor_kernel.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

# A plain (non-rdc) topscc object already carries a self-contained
# .tops_fatbin, so no separate device-link pass is required — that is exactly
# what makefiles/enflame.mk records with DEVICE_NEEDS_DLINK := 0. The flags
# therefore come straight from DEVICE_COMPILE_FLAG
# (-c -std=c++17 -O3 -fPIC -MMD -MP -arch gcu400 -arch gcu410) rather than
# being spelled out per rule as on CUDA.
$(OBJDIR)/device_api.o: device_api.tops
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.tops (Enflame/TOPS)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -Wno-deprecated-declarations -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.tops
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.tops (Enflame/TOPS)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -Wno-deprecated-declarations -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# Not available on Enflame:
#   * dlink*      — device code is self-contained in device_api.o / device_ir.o
#                   (see above). Running topscc -dlink would also be wrong: it
#                   is an alias for --tops-device-lib and consumes the next
#                   argument, which is why enflame.mk sets DEVICE_NEEDS_DLINK
#                   to 0. KLX refuses the same targets for the same reason.
#   * nvshmem_*   — USE_SHMEM is unsupported; enflame.mk falls back to ECCL.
dlink.o dlink_api.o dlink_ir.o dlink_shmem_test.o nvshmem_adaptor_kernel.o:
	@echo "Enflame: '$@' is not available (device code is self-contained in device_api.o / device_ir.o; USE_SHMEM is unsupported)." >&2
	@false

# Consume the -MMD output. DEVICE_COMPILE_FLAG already asks topscc for -MMD -MP,
# but nothing was reading the files, so editing any header left stale objects
# behind and the next `make` reported success without compiling anything.
# nvidia/Makefile has the same gap; this is a deliberate improvement, not a
# deviation in spelling. The leading `-` keeps the first build quiet, when no
# .d exists yet.
-include $(OBJDIR)/device_api.d
-include $(OBJDIR)/device_ir.d

clean:
	@rm -rf $(BUILDDIR)
