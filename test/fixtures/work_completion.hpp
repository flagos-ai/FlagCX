#ifndef FLAGCX_TEST_WORK_COMPLETION_HPP_
#define FLAGCX_TEST_WORK_COMPLETION_HPP_
#include "adaptor.h"
#include "device_event.h"
#include "work_completion.h"
#include <gtest/gtest.h>
#include <vector>

namespace {
struct WorkTestStream {
  unsigned submitted = 0, completed = 0, captured = 0;
  flagcxStreamCaptureStatus_t capture = flagcxStreamCaptureNone;
};
struct WorkTestEvent {
  WorkTestStream *stream = nullptr;
  unsigned fence = 0;
};
class WorkCompletionTest : public ::testing::Test {
protected:
  inline static WorkCompletionTest *current;
  flagcxDeviceAdaptor *savedDevice = nullptr;
  flagcxDeviceAdaptor device = {};
  flagcxWorkTracker tracker = {};
  WorkTestStream stream, otherStream;
  int created = 0, recorded = 0, waited = 0, destroyed = 0, deviceSyncs = 0;
  int createFailures = 0, recordFailures = 0, waitFailures = 0,
      destroyFailures = 0;
  int nativeDepth = 0, queried = 0;
  int partialCreateFailures = 0, wrappersAllocated = 0, wrappersFreed = 0;
  flagcxResult_t captureResult = flagcxSuccess;
  std::vector<WorkTestStream *> nativePending;
  void SetUp() override {
    current = this;
    savedDevice = deviceAdaptor;
    device = *savedDevice;
    deviceAdaptor = &device;
    device.setDevice = [](int) { return flagcxSuccess; };
    device.streamGetCaptureStatus = [](flagcxStream_t input,
                                       flagcxStreamCaptureStatus_t *output) {
      *output =
          input ? ((WorkTestStream *)input)->capture : flagcxStreamCaptureNone;
      return current->captureResult;
    };
    device.eventCreate = [](flagcxEvent_t *output, flagcxEventType_t) {
      static const flagcxEventCreateOps ops = {
          [](size_t) -> void * {
            if (current->createFailures > 0) {
              current->createFailures--;
              return nullptr;
            }
            current->wrappersAllocated++;
            return new WorkTestEvent;
          },
          [](void *wrapper) {
            current->wrappersFreed++;
            delete (WorkTestEvent *)wrapper;
          },
          [](void *, unsigned int) {
            if (current->partialCreateFailures > 0) {
              current->partialCreateFailures--;
              return flagcxSystemError;
            }
            current->created++;
            return flagcxSuccess;
          }};
      void *wrapper = nullptr;
      flagcxResult_t result = flagcxEventCreateTransactional(
          &wrapper, sizeof(WorkTestEvent), 0, &ops);
      *output = (flagcxEvent_t)wrapper;
      return result;
    };
    device.eventRecord = [](flagcxEvent_t input, flagcxStream_t workStream) {
      current->recorded++;
      EXPECT_TRUE(workStream == nullptr ||
                  ((WorkTestStream *)workStream)->capture ==
                      flagcxStreamCaptureNone)
          << "completion event inserted into active capture";
      EXPECT_EQ(current->nativeDepth, 0) << "recorded before native submission";
      if (current->recordFailures > 0) {
        current->recordFailures--;
        return flagcxRemoteError;
      }
      auto *event = (WorkTestEvent *)input;
      event->stream = (WorkTestStream *)workStream;
      event->fence = event->stream ? event->stream->submitted : 0;
      return flagcxSuccess;
    };
    device.eventQuery = [](flagcxEvent_t input) {
      current->queried++;
      auto *event = (WorkTestEvent *)input;
      return event->stream == nullptr ||
                     event->stream->completed >= event->fence
                 ? flagcxSuccess
                 : flagcxInProgress;
    };
    device.eventSynchronize = [](flagcxEvent_t input) {
      current->waited++;
      if (current->waitFailures > 0) {
        current->waitFailures--;
        return flagcxUnhandledDeviceError;
      }
      return current->device.eventQuery(input);
    };
    device.eventDestroy = [](flagcxEvent_t input) {
      if (current->destroyFailures > 0) {
        current->destroyFailures--;
        return flagcxSystemError;
      }
      current->destroyed++;
      current->wrappersFreed++;
      delete (WorkTestEvent *)input;
      return flagcxSuccess;
    };
    device.deviceSynchronize = [] {
      current->deviceSyncs++;
      return flagcxRemoteError;
    };
  }
  void installNative(flagcxCCLAdaptor *adaptor) {
    adaptor->groupStart = [] {
      current->nativeDepth++;
      return flagcxSuccess;
    };
    adaptor->groupEnd = [] {
      if (--current->nativeDepth == 0) {
        for (auto *stream : current->nativePending)
          if (stream->capture == flagcxStreamCaptureActive)
            stream->captured++;
          else
            stream->submitted++;
        current->nativePending.clear();
      }
      return flagcxSuccess;
    };
    adaptor->allReduce = [](const void *, void *, size_t, flagcxDataType_t,
                            flagcxRedOp_t, flagcxInnerComm_t,
                            flagcxStream_t input) {
      auto *stream = (WorkTestStream *)input;
      if (current->nativeDepth)
        current->nativePending.push_back(stream);
      else if (stream->capture == flagcxStreamCaptureActive)
        stream->captured++;
      else
        stream->submitted++;
      return flagcxSuccess;
    };
  }
  flagcxWorkRecord *prepare(flagcxWorkTracker *owner = nullptr) {
    flagcxWorkRecord *record = nullptr;
    EXPECT_EQ(flagcxWorkPrepare(owner ? owner : &tracker,
                                (flagcxStream_t)&stream, 1, &record),
              flagcxSuccess);
    return record;
  }
  flagcxResult_t finish(flagcxWorkRecord *record,
                        WorkTestStream *workStream = nullptr,
                        flagcxResult_t result = flagcxSuccess,
                        int deferred = 0) {
    return flagcxWorkFinish(record,
                            (flagcxStream_t)(workStream ? workStream : &stream),
                            result, flagcxWorkNativeGroup, deferred);
  }
  void releaseFake(flagcxWorkTracker *owner) {
    // No real GPU work exists in this fixture. Resolve retained uncertainty
    // only for teardown; production must never fabricate this confirmation.
    for (auto *node = owner->head; node; node = node->next)
      node->state = flagcxWorkComplete;
    EXPECT_EQ(flagcxWorkWait(owner), flagcxSuccess);
    EXPECT_EQ(flagcxWorkRelease(owner), flagcxSuccess);
  }
  void TearDown() override {
    createFailures = recordFailures = waitFailures = destroyFailures = 0;
    partialCreateFailures = 0;
    captureResult = flagcxSuccess;
    stream.capture = otherStream.capture = flagcxStreamCaptureNone;
    EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxSuccess),
              flagcxSuccess);
    releaseFake(&tracker);
    EXPECT_EQ(created, destroyed);
    EXPECT_EQ(wrappersAllocated, wrappersFreed);
    EXPECT_EQ(deviceSyncs, 0);
    deviceAdaptor = savedDevice;
  }
};
} // namespace
#endif
