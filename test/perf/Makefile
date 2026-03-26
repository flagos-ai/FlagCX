# Perf test orchestration — delegates to host_api/ and device_api/.
#
# Usage:
#   make              Build all perf benchmarks
#   make host_api     Build host API benchmarks only
#   make device_api   Build device API benchmarks only
#   make clean        Clean all build artifacts

.PHONY: all host_api device_api clean

all: host_api device_api

host_api:
	@$(MAKE) -C host_api

# Device API tests require kernel functions compiled into libflagcx (.cu files).
# Use -k to continue past failures and || true to not fail the overall build.
device_api:
	@$(MAKE) -C device_api || echo "Warning: device_api build failed (kernel functions may not be compiled into libflagcx)"

clean:
	@$(MAKE) -C host_api clean
	@$(MAKE) -C device_api clean
