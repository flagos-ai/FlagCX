# Headers needed by a standalone Device API bitcode consumer, plus the six
# public headers the wheel already exports. Keep the source tree out of the
# consumer compile path so missing exports fail during consumer builds.
.DEFAULT_GOAL := all

IR_COMMON_HEADERS := \
	flagcx_device_wrapper.h \
	flagcx_device_scalar_ir.h \
	flagcx_device_unified_ir.h \
	flagcx_device_api.h \
	flagcx_device_core.h \
	flagcx_device_enums.h \
	comm_traits.h \
	platform_traits.h \
	default_comm_traits.h \
	default_completion.h \
	fifo_producer_gate.h \
	device_utils.h \
	flagcx.h \
	flagcx_kernel.h \
	flagcx_kernel_core.h \
	flagcx_p2p.h

IR_HEADER_NAMES := $(IR_COMMON_HEADERS) $(IR_PLATFORM_HEADERS)
IR_HEADER_TARGETS := $(addprefix $(INCLUDEDIR)/,$(IR_HEADER_NAMES))
IR_HEADER_MANIFEST := $(INCLUDEDIR)/.flagcx_device_headers

vpath %.h \
	$(FLAGCX_ROOT)/bindings/ir \
	$(FLAGCX_ROOT)/flagcx/include \
	$(FLAGCX_ROOT)/flagcx/adaptor/include \
	$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api

$(INCLUDEDIR)/%.h: %.h
	@mkdir -p $(INCLUDEDIR)
	cp -f $< $@

headers: $(IR_HEADER_TARGETS)
	@printf '%s\n' $(IR_HEADER_NAMES) > $(IR_HEADER_MANIFEST)

.PHONY: headers
