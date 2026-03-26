# Top-level test orchestration — delegates to unittest/ and perf/.
#
# Usage:
#   make              Build all unittest binaries
#   make unittest     Build unittest only
#   make perf         Build perf benchmarks only
#   make run-unit     Run unit tests (no MPI/GPU needed)
#   make run-mpi      Run MPI tests (requires MPI + GPUs)
#   make run-all      Run everything
#   make clean        Clean all build artifacts

.PHONY: all unittest perf clean run-unit run-mpi run-all

all: unittest

unittest:
	@$(MAKE) -C unittest

perf:
	@$(MAKE) -C perf

clean:
	@$(MAKE) -C unittest clean
	@$(MAKE) -C perf clean

# Run only unit tests (no MPI/GPU needed)
run-unit:
	@$(MAKE) -C unittest run-unit

# Run MPI tests (requires MPI + GPUs)
run-mpi:
	@$(MAKE) -C unittest run-mpi

# Run everything
run-all: run-unit run-mpi
