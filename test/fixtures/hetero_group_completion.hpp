#ifndef FLAGCX_TEST_HETERO_GROUP_COMPLETION_HPP_
#define FLAGCX_TEST_HETERO_GROUP_COMPLETION_HPP_
#include "group.h"
#include "launch_kernel.h"
#include "proxy.h"
#include "work_completion.hpp"
#include <climits>
#include <memory>
#include <unistd.h>

namespace {
class HeteroGroupCompletionTest : public WorkCompletionTest {
protected:
  struct Participant {
    flagcxHeteroComm comm = {};
    flagcxProxyState proxy = {};
    flagcxTasks::Peer task = {};
    flagcxChannelPeer peer = {};
    flagcxChannelPeer *peerPointer = &peer;
    WorkTestStream stream;
    flagcxWorkRecord *record = nullptr;
    Participant() {
      comm.nRanks = comm.localRanks = 1;
      comm.proxyState = &proxy;
      comm.tasks.peers = &task;
      comm.channels[0].peers = &peerPointer;
      comm.groupNext = (flagcxHeteroComm *)0x1;
      pthread_mutex_init(&proxy.mutex, nullptr);
      pthread_cond_init(&proxy.cond, nullptr);
    }
    ~Participant() {
      pthread_cond_destroy(&proxy.cond);
      pthread_mutex_destroy(&proxy.mutex);
    }
  } a, b;
  flagcxLaunchFunc_t savedKernel;
  flagcxResult_t completionResult = flagcxSuccess;
  int completionCalls = 0;
  bool failNextMarker = false;
  void SetUp() override {
    WorkCompletionTest::SetUp();
    savedKernel = deviceAsyncKernel;
    deviceAsyncKernel = nullptr;
    device.hostGetDevicePointer = nullptr;
    device.launchHostFunc = [](flagcxStream_t input, void (*)(void *), void *) {
      auto *self = static_cast<HeteroGroupCompletionTest *>(current);
      self->completionCalls++;
      if (self->completionResult != flagcxSuccess)
        return self->completionResult;
      ((WorkTestStream *)input)->submitted++;
      if (self->failNextMarker)
        self->recordFailures = 1;
      return flagcxSuccess;
    };
    device.streamWaitEvent = [](flagcxStream_t, flagcxEvent_t) {
      return flagcxSuccess;
    };
    device.streamSynchronize = [](flagcxStream_t input) {
      auto *stream = (WorkTestStream *)input;
      stream->completed = stream->submitted;
      return flagcxSuccess;
    };
    device.deviceMalloc = [](void **out, size_t size, flagcxMemType_t,
                             flagcxStream_t) {
      *out = calloc(1, size);
      return *out != nullptr ? flagcxSuccess : flagcxSystemError;
    };
    device.deviceFree = [](void *p, flagcxMemType_t, flagcxStream_t) {
      free(p);
      return flagcxSuccess;
    };
  }
  void selectMode(int mode) {
    if (mode == 1) {
      device.hostGetDevicePointer = [](void **out, void *p) {
        *out = p;
        return flagcxSuccess;
      };
      deviceAsyncKernel = [](flagcxStream_t, void *) {};
      device.launchDeviceFunc = [](flagcxStream_t input, flagcxLaunchFunc_t,
                                   void *) {
        auto *self = static_cast<HeteroGroupCompletionTest *>(current);
        self->completionCalls++;
        if (self->completionResult != flagcxSuccess)
          return self->completionResult;
        ((WorkTestStream *)input)->submitted++;
        return flagcxSuccess;
      };
    } else if (mode == 2) {
      flagcxClearStreamValueProbeCache();
      device.hostGetDevicePointer = [](void **out, void *p) {
        *out = p;
        return flagcxSuccess;
      };
      device.getDevice = [](int *out) {
        *out = 75;
        return flagcxSuccess;
      };
      device.streamCreate = [](flagcxStream_t *out) {
        *out = (flagcxStream_t) new WorkTestStream;
        return flagcxSuccess;
      };
      device.streamDestroy = [](flagcxStream_t input) {
        delete (WorkTestStream *)input;
        return flagcxSuccess;
      };
      device.streamWriteValue64 = [](flagcxStream_t, void *p, uint64_t value,
                                     int) {
        *(uint64_t *)p = value;
        return flagcxSuccess;
      };
      device.streamWaitValue64 = [](flagcxStream_t input, void *, uint64_t,
                                    int) {
        auto *self = static_cast<HeteroGroupCompletionTest *>(current);
        self->completionCalls++;
        if (self->completionResult != flagcxSuccess)
          return self->completionResult;
        ((WorkTestStream *)input)->submitted++;
        return flagcxSuccess;
      };
      bool supported = false;
      ASSERT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
      ASSERT_TRUE(supported);
      completionCalls = 0;
    }
  }
  void submit(Participant *p) {
    ASSERT_EQ(flagcxWorkPrepare(&p->comm.submittedWork,
                                (flagcxStream_t)&p->stream, 0, &p->record),
              flagcxSuccess);
    for (int receive = 0; receive < 2; receive++) {
      auto *task = (flagcxTaskP2p *)calloc(1, sizeof(flagcxTaskP2p));
      task->buff = (void *)(uintptr_t)(receive ? 0x2000 : 0x1000);
      task->bytes = 4;
      task->dtype = flagcxFloat;
      task->opId = INT_MAX;
      task->step = -1;
      task->stream = (flagcxStream_t)&p->stream;
      if (receive)
        flagcxIntruQueueEnqueue(&p->task.recvQueue, task);
      else
        flagcxIntruQueueEnqueue(&p->task.sendQueue, task);
    }
    ASSERT_EQ(flagcxWorkFinish(p->record, (flagcxStream_t)&p->stream,
                               flagcxSuccess, flagcxWorkHeteroGroup, 1),
              flagcxSuccess);
    flagcxGroupCommJoin(&p->comm);
  }
  void checkMode(int mode) {
    ASSERT_NO_FATAL_FAILURE(selectMode(mode));
    ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
    ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
    ASSERT_NO_FATAL_FAILURE(submit(&a));
    ASSERT_NO_FATAL_FAILURE(submit(&b)); // Last joined is the launch stream.
    ASSERT_EQ(flagcxGroupEndInternal(mode == 2), flagcxSuccess);
    EXPECT_EQ(completionCalls, 0);
    ASSERT_EQ(flagcxGroupEndInternal(mode == 2), flagcxSuccess);
    EXPECT_EQ(completionCalls, 1);
    EXPECT_EQ(((WorkTestEvent *)a.record->event)->stream, &b.stream);
    EXPECT_EQ(((WorkTestEvent *)b.record->event)->stream, &b.stream);
    a.stream.completed = a.stream.submitted;
    EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxInProgress);
    EXPECT_EQ(flagcxWorkRelease(&a.comm.submittedWork), flagcxInvalidUsage);
    unsigned boundary = b.stream.submitted;
    b.stream.submitted++; // Independent later work is outside this group.
    b.stream.completed = boundary;
    EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxSuccess);
    EXPECT_EQ(flagcxWorkWait(&b.comm.submittedWork), flagcxSuccess);
    EXPECT_EQ(deviceSyncs, 0);
  }
  void retire(Participant *p) {
    auto *queue = &p->proxy.proxyOps[0].prodPeers.sendQueue;
    while (!flagcxIntruQueueEmpty(queue)) {
      auto *op = flagcxIntruQueueDequeue(queue);
      delete op;
    }
    for (auto *queue : {&p->task.sendQueue, &p->task.recvQueue})
      while (!flagcxIntruQueueEmpty(queue))
        free(flagcxIntruQueueDequeue(queue));
    releaseFake(&p->comm.submittedWork);
  }
  void TearDown() override {
    while (flagcxGetGroupDepth() > 0)
      flagcxGroupEndInternal();
    retire(&a);
    retire(&b);
    EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    flagcxClearStreamValueProbeCache();
    deviceAsyncKernel = savedKernel;
    WorkCompletionTest::TearDown();
  }
};
} // namespace
#endif
