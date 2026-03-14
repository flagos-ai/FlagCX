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

device_api:
	@$(MAKE) -C device_api

clean:
	@$(MAKE) -C host_api clean
	@$(MAKE) -C device_api clean
