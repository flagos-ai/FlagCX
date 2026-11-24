# Optional SDK capability, established by linking the actual typed callback.
# Probe failure disables introspection only; it never disables ordinary work.
FLAGCX_NATIVE_CAPTURE_QUERY := $(shell python3 $(CAPTURE_PROBE_ROOT)/tools/probe_capture.py \
  --root=$(CAPTURE_PROBE_ROOT) --flags="$(ADAPTOR_FLAG)" \
  --cxx="$(CAPTURE_PROBE_CXX)" --cppflags="$(CAPTURE_PROBE_CPPFLAGS)" \
  --ldflags="-L$(DEVICE_LIB) $(DEVICE_LINK)" --report=$(BUILDDIR)/capture-sdk.json)
CAPTURE_PROBE_DEFINE := -DFLAGCX_NATIVE_CAPTURE_QUERY=$(FLAGCX_NATIVE_CAPTURE_QUERY)
