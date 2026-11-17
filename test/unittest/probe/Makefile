include ../../make.inc

MPIRUN ?= mpirun
IPC_XML_DIR ?=
IPC_FILTER ?= RmaTest.IpcGetSameStreamProbe:RmaTest.IpcGetCounterOrderProbe
RMA_DIR := $(abspath ../rma)
IPC_TEST_CMD = $(MPI_TARGET)
ifneq ($(strip $(IPC_XML_DIR)),)
IPC_TEST_CMD = bash $(PROJECT_ROOT)/.github/scripts/ci/run_gtest_rank_xml.sh $(IPC_XML_DIR) $(MPI_TARGET)
endif
LOCAL_INCLUDE := -I$(RMA_DIR)/include \
	-I$(PROJECT_ROOT)/flagcx/adaptor/device_api $(KERNEL_INCLUDE)

MPI_TARGET := $(BINDIR)/probe_mpi_tests
MPI_OBJS := $(OBJDIR)/coll_probe.o $(OBJDIR)/rma_test.o $(OBJDIR)/main_mpi.o

.PHONY: all run-mpi-ipc clean

all: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) -L$(GTEST_LIB) -lgtest -lpthread $(MPI_LINK) $(DEVICE_LINK)

$(OBJDIR)/coll_probe.o: coll_probe.cpp
	@mkdir -p $(OBJDIR)
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/rma_test.o: $(RMA_DIR)/rma_test.cpp
	@mkdir -p $(OBJDIR)
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/main_mpi.o: $(RMA_DIR)/main_mpi.cpp
	@mkdir -p $(OBJDIR)
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

run-mpi-ipc: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np 2 \
		-x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 \
		-x FLAGCX_VMM_ENABLE=0 -x FLAGCX_IB_DISABLE=1 \
		-x FLAGCX_P2P_DISABLE=0 \
		$(IPC_TEST_CMD) \
		--gtest_filter=$(IPC_FILTER)

clean:
	@rm -rf $(BUILDDIR)
