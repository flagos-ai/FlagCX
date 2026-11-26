#ifndef FLAGCX_TEST_DEVICE_API_REPORT_H_
#define FLAGCX_TEST_DEVICE_API_REPORT_H_

#include <cstddef>
#include <cstdio>

// Report once after a required case's repetitions/subcases have been graded.
// Size is the requested sweep size, before payload rounding. All string fields
// are fixed test identifiers/status literals, so no JSON escaping is needed.
class DeviceApiReporter {
public:
  DeviceApiReporter(const char *binary, int rank)
      : binary_(binary), rank_(rank) {}

  void report(const char *caseId, size_t sizeBytes, const char *status) const {
    // One complete, flushed record; do not depend on other ranks' size banners.
    std::printf(
        "FLAGCX_DEVICE_API_RESULT {\"version\":1,\"binary\":\"%s\","
        "\"rank\":%d,\"case\":\"%s\",\"size_bytes\":%zu,\"status\":\"%s\"}\n",
        binary_, rank_, caseId, sizeBytes, status);
    std::fflush(stdout);
  }

private:
  const char *binary_;
  int rank_;
};

#endif // FLAGCX_TEST_DEVICE_API_REPORT_H_
