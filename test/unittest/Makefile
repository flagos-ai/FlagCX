# Unittest orchestration — delegates to component sub-directories.
# Each component produces up to two binaries:
#   <component>_unit_tests — from test_*.cpp (no MPI, runs locally)
#   <component>_mpi_tests  — from coll_*.cpp (requires MPI + GPUs)
#
# Usage:
#   make              Build all components
#   make run-unit     Run all unit tests (no MPI/GPU)
#   make run-mpi      Run all MPI tests
#   make run          Run everything
#   make clean        Clean all build artifacts

COMPONENTS := adaptor core kernel service runner p2p topo

.PHONY: all $(COMPONENTS) clean run-unit run-mpi run

all: $(COMPONENTS)

$(COMPONENTS):
	@$(MAKE) -C $@

clean:
	@for comp in $(COMPONENTS); do $(MAKE) -C $$comp clean; done

run-unit:
	@for comp in $(COMPONENTS); do $(MAKE) -C $$comp run-unit; done

run-mpi:
	@for comp in $(COMPONENTS); do $(MAKE) -C $$comp run-mpi; done

run: run-unit run-mpi
