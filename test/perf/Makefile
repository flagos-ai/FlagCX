# Perf test orchestration — delegates to host_api/ and device_api/.
#
# Usage:
#   make              Build all perf benchmarks
#   make host_api     Build host API benchmarks only
#   make device_api   Build device API benchmarks only (requires COMPILE_KERNEL=1)
#   make clean        Clean all build artifacts

SUBDIRS := host_api

ifeq ($(COMPILE_KERNEL), 1)
SUBDIRS += device_api
endif

.PHONY: all host_api device_api clean

all: $(SUBDIRS)

host_api:
	@$(MAKE) -C host_api

# Device API tests require kernel functions compiled into libflagcx (.cu files).
device_api:
ifeq ($(COMPILE_KERNEL), 1)
	@$(MAKE) -C device_api
else
	$(error device_api benchmarks require COMPILE_KERNEL=1)
endif

clean:
	@$(MAKE) -C host_api clean
	@$(MAKE) -C device_api clean
