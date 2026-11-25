#include "adaptor.h"
#include "global_comm.h"
#include "runner.h"
#include "work.h"
#include <gtest/gtest.h>

namespace {
flagcxResult_t nativeResult;
int nativeCalls;
flagcxResult_t called() {
  ++nativeCalls;
  return nativeResult;
}
class HomoRunnerStatus : public ::testing::Test {
protected:
  flagcxCCLAdaptor *saved;
  flagcxCCLAdaptor mock = {};
  flagcxComm comm = {};
  void SetUp() override {
    saved = cclAdaptors[flagcxCCLAdaptorDevice];
    cclAdaptors[flagcxCCLAdaptorDevice] = &mock;
    mock.reduce = [](const void *, void *, size_t, flagcxDataType_t,
                     flagcxRedOp_t, int, flagcxInnerComm_t,
                     flagcxStream_t) { return called(); };
    mock.gather = mock.scatter = mock.broadcast =
        [](const void *, void *, size_t, flagcxDataType_t, int,
           flagcxInnerComm_t, flagcxStream_t) { return called(); };
    mock.allReduce = mock.reduceScatter =
        [](const void *, void *, size_t, flagcxDataType_t, flagcxRedOp_t,
           flagcxInnerComm_t, flagcxStream_t) { return called(); };
    mock.allGather = mock.alltoAll = [](const void *, void *, size_t,
                                        flagcxDataType_t, flagcxInnerComm_t,
                                        flagcxStream_t) { return called(); };
    mock.alltoAllv = [](const void *, size_t *, size_t *, void *, size_t *,
                        size_t *, flagcxDataType_t, flagcxInnerComm_t,
                        flagcxStream_t) { return called(); };
    mock.send = [](const void *, size_t, flagcxDataType_t, int,
                   flagcxInnerComm_t, flagcxStream_t) { return called(); };
    mock.recv = [](void *, size_t, flagcxDataType_t, int, flagcxInnerComm_t,
                   flagcxStream_t) { return called(); };
    mock.groupStart = mock.groupEnd = called;
  }
  void TearDown() override {
    flagcxWorkNativeGroupResult(flagcxSuccess);
    cclAdaptors[flagcxCCLAdaptorDevice] = saved;
  }
};
TEST_F(HomoRunnerStatus, EveryNativeCallPreservesStatusAndExecutesOnce) {
  using Call = flagcxResult_t (*)(flagcxComm_t);
  const Call calls[] = {
      [](flagcxComm_t c) {
        return homoRunner.reduce(NULL, NULL, 0, flagcxInt, flagcxSum, 0, c,
                                 NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.gather(NULL, NULL, 0, flagcxInt, 0, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.scatter(NULL, NULL, 0, flagcxInt, 0, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.broadcast(NULL, NULL, 0, flagcxInt, 0, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.allReduce(NULL, NULL, 0, flagcxInt, flagcxSum, c,
                                    NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.reduceScatter(NULL, NULL, 0, flagcxInt, flagcxSum, c,
                                        NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.allGather(NULL, NULL, 0, flagcxInt, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.alltoAll(NULL, NULL, 0, flagcxInt, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.alltoAllv(NULL, NULL, NULL, NULL, NULL, NULL,
                                    flagcxInt, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.send(NULL, 0, flagcxInt, 0, c, NULL);
      },
      [](flagcxComm_t c) {
        return homoRunner.recv(NULL, 0, flagcxInt, 0, c, NULL);
      },
      [](flagcxComm_t) { return homoRunner.groupStart(); },
      [](flagcxComm_t) { return homoRunner.groupEnd(); }};
  for (auto result : {flagcxSuccess, flagcxInProgress, flagcxRemoteError,
                      flagcxNotSupported}) {
    for (auto call : calls) {
      nativeResult = result;
      nativeCalls = 0;
      EXPECT_EQ(call(&comm), result);
      EXPECT_EQ(nativeCalls, 1);
    }
  }
}
} // namespace
