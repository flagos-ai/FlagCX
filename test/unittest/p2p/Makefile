include ../../make.inc

# [unit] tests: test_*.cpp — no MPI, requires IB hardware for some tests
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
GTEST_SRC_DIR := $(PROJECT_ROOT)/third-party/googletest/googletest/src
GTEST_OBJS := $(OBJDIR)/gtest/gtest-all.o $(OBJDIR)/gtest/gtest_main.o
UNIT_TARGET := $(BINDIR)/p2p_unit_tests

.PHONY: all clean run-unit run-mpi run

ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif

all: $(ALL_TARGETS)

$(UNIT_TARGET): $(UNIT_OBJS) $(GTEST_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) -lpthread

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE)

$(OBJDIR)/gtest/%.o: $(GTEST_SRC_DIR)/%.cc
	@mkdir -p $(dir $@)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(GTEST_INCLUDE) -I$(PROJECT_ROOT)/third-party/googletest/googletest

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi:
	@echo "No MPI tests for p2p (yet)."

run: run-unit run-mpi
