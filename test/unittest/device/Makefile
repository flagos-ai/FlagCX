# Device adaptor test - single GPU, no MPI required
BUILDDIR ?= $(abspath ./build)
BINDIR := $(BUILDDIR)/bin
OBJDIR := $(BUILDDIR)/obj

# Project root
PROJECT_ROOT := $(abspath ../../..)

# Include directories
INCLUDEDIR := \
	$(PROJECT_ROOT)/flagcx/include \
	$(PROJECT_ROOT)/flagcx/adaptor/include \
	$(PROJECT_ROOT)/flagcx/core/include \
	$(PROJECT_ROOT)/flagcx/service/include \
	$(PROJECT_ROOT)/flagcx/runner/include \
	$(PROJECT_ROOT)/third-party/googletest/googletest/include \
	$(PROJECT_ROOT)/third-party/json/single_include

# Source files
SRCFILES := device_adaptor_test.cpp
BINOBJ := $(SRCFILES:%.cpp=$(OBJDIR)/%.o)

# GoogleTest
GTEST_DIR := $(PROJECT_ROOT)/third-party/googletest
GTEST_LIB := $(GTEST_DIR)/build/lib

# FlagCX library
FLAGCX_LIB := $(PROJECT_ROOT)/build/lib

# Target
TARGET := device_test

.PHONY: all clean run

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(BINOBJ)
	@mkdir -p `dirname $@`
	@echo "Linking   $@"
	@g++ $(BINOBJ) -o $@ \
		-L$(FLAGCX_LIB) \
		-L$(GTEST_LIB) \
		-Wl,--no-as-needed \
		-Wl,-rpath,$(FLAGCX_LIB) \
		-lflagcx \
		-lgtest \
		-lgtest_main \
		-lpthread \
		-g

$(OBJDIR)/%.o: %.cpp
	@mkdir -p `dirname $@`
	@echo "Compiling $@"
	@g++ $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -c \
		-std=c++17 \
		-Wvla \
		-Wno-unused-function \
		-Wno-sign-compare \
		-Wall \
		-g

-include $(BINOBJ:.o=.d)

clean:
	@rm -rf $(BUILDDIR)

run: $(BINDIR)/$(TARGET)
	@echo "Running device adaptor tests..."
	@$(BINDIR)/$(TARGET)

# Run with verbose output
run-verbose: $(BINDIR)/$(TARGET)
	@echo "Running device adaptor tests (verbose)..."
	@$(BINDIR)/$(TARGET) --gtest_print_time=1

# Run specific test
run-test: $(BINDIR)/$(TARGET)
	@echo "Running test: $(TEST)"
	@$(BINDIR)/$(TARGET) --gtest_filter=$(TEST)

help:
	@echo "Device Adaptor Test Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build the test"
	@echo "  make run          - Build and run all tests"
	@echo "  make run-verbose  - Run with verbose output"
	@echo "  make run-test TEST=<test_name> - Run specific test"
	@echo "  make clean        - Clean build files"
	@echo ""
	@echo "Examples:"
	@echo "  make run-test TEST=DeviceAdaptorTest.MemoryCopy"
	@echo "  make run-test TEST=DeviceAdaptorTest.*Memory*"

