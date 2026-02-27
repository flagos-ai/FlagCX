# NCCL Wrapper for FlagCX
# Produces libnccl.so that implements NCCL 2.27.7 API via FlagCX

FLAGCX_HOME ?= $(abspath ../..)
CUDA_HOME   ?= /usr/local/cuda
NCCL_HOME     ?= /usr/local/nccl
REAL_NCCL_LIB ?= $(NCCL_HOME)/lib/libnccl.so.2
BUILDDIR      ?= $(abspath build)

LIBDIR     := $(BUILDDIR)/lib
OBJDIR     := $(BUILDDIR)/obj
INCLUDEDIR := $(BUILDDIR)/include

# Library versioning
NCCL_MAJOR   := 2
NCCL_MINOR   := 27
NCCL_PATCH   := 7
NCCL_VERSION := $(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH)
SONAME       := libnccl.so.$(NCCL_MAJOR)
TARGET       := libnccl.so.$(NCCL_VERSION)

# Source files
SRCS := $(wildcard src/*.cc)
OBJS := $(SRCS:src/%.cc=$(OBJDIR)/%.o)

# Compiler flags
CXX      := g++
CXXFLAGS := -fPIC -fvisibility=default -std=c++11 -Wall -Wno-sign-compare -O2 -g \
            -DREAL_NCCL_PATH=\"$(REAL_NCCL_LIB)\"
INCLUDES := -I./include \
            -I$(FLAGCX_HOME)/flagcx/include \
            -I$(CUDA_HOME)/include
LDFLAGS  := -shared -Wl,-soname,$(SONAME) \
            -L$(FLAGCX_HOME)/build/lib -lflagcx \
            -L$(CUDA_HOME)/lib64 -lcudart \
            -ldl \
            -Wl,-rpath,$(FLAGCX_HOME)/build/lib \
            -Wl,-rpath,$(CUDA_HOME)/lib64

.PHONY: all clean headers

all: $(LIBDIR)/$(TARGET) symlinks headers

$(LIBDIR)/$(TARGET): $(OBJS)
	@mkdir -p $(LIBDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: src/%.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

symlinks: $(LIBDIR)/$(TARGET)
	@ln -sf $(TARGET) $(LIBDIR)/$(SONAME)
	@ln -sf $(SONAME) $(LIBDIR)/libnccl.so

headers:
	@mkdir -p $(INCLUDEDIR)
	@cp -f ./include/nccl.h $(INCLUDEDIR)/nccl.h

-include $(OBJS:.o=.d)

clean:
	@rm -rf $(BUILDDIR)
