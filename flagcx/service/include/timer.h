/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_TIMER_H_
#define FLAGCX_TIMER_H_
#if ENABLE_TIMER
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>
static double freq = -1;
static void calibrate() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = -tv.tv_sec * 1E6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i = 0; i < 10000; i++)
    total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec * 1E6 + tv.tv_usec;
  freq = timeCycles / time;
}
static inline double gettime() {
  if (freq == -1)
    calibrate();
  return __rdtsc() / freq;
}
static uint64_t counts[8];
static double times[8];
static double startTimes[8];
#define TIME_START(index)                                                      \
  do {                                                                         \
    counts[index]++;                                                           \
    startTimes[index] = gettime();                                             \
  } while (0);

#define TIME_STOP(index)                                                       \
  do {                                                                         \
    times[index] += gettime() - startTimes[index];                             \
  } while (0);

#define TIME_CANCEL(index)                                                     \
  do {                                                                         \
    counts[index]--;                                                           \
  } while (0);

#define TIME_PRINT(name)                                                       \
  do {                                                                         \
    printf("%s stats", name);                                                  \
    for (int i = 0; i < 8; i++) {                                              \
      if (counts[i])                                                           \
        printf(" [%d] %g/%ld = %g", i, times[i], counts[i],                    \
               times[i] / counts[i]);                                          \
      counts[i] = 0;                                                           \
    }                                                                          \
    printf("\n");                                                              \
  } while (0);
#else
#define TIME_START(index)                                                      \
  while (0)                                                                    \
    ;
#define TIME_STOP(index)                                                       \
  while (0)                                                                    \
    ;
#define TIME_CANCEL(index)                                                     \
  while (0)                                                                    \
    ;
#define TIME_PRINT(name)
#endif

#include <cassert>
#include <pthread.h>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "adaptor.h"
#include "debug.h"
#include "flagcx.h"

constexpr int RECORD_NUM = 2048;

template <typename T>
struct flagcxRecordKey {
  T value;

  flagcxRecordKey() = default;
  flagcxRecordKey(const T &v) : value(v) {}

  bool operator<(const flagcxRecordKey<T> &other) const {
    return value < other.value;
  }
  bool operator==(const flagcxRecordKey<T> &other) const {
    return value == other.value;
  }
};

template <typename T>
struct flagcxRecord {
  flagcxEvent_t beginEvent;
  flagcxEvent_t endEvent;
  flagcxRecordKey<T> recordKey;
  float duration; // ms
  flagcxStream_t stream;

  flagcxRecord();
  flagcxRecord(const flagcxRecord &) = delete;
  flagcxRecord &operator=(const flagcxRecord &) = delete;
  flagcxRecord(flagcxRecord &&) = delete;
  flagcxRecord &operator=(flagcxRecord &&) = delete;

  ~flagcxRecord();
};

template <typename T>
flagcxRecord<T>::flagcxRecord() : duration(0.0f) {
  deviceAdaptor->eventCreate(&beginEvent, flagcxEventDefault);
  deviceAdaptor->eventCreate(&endEvent, flagcxEventDefault);
}

template <typename T>
flagcxRecord<T>::~flagcxRecord<T>() {
  deviceAdaptor->eventDestroy(beginEvent);
  deviceAdaptor->eventDestroy(endEvent);
}

template <typename T>
class flagcxTimer {

public:
  flagcxRecord<T> flagcxRecords[RECORD_NUM];
  pthread_t queryThread;
  bool stopQuery = false;
  std::queue<flagcxRecord<T> *> availableRecords; // NOLINT
  std::queue<flagcxRecord<T> *> usingRecords;     // NOLINT
  std::queue<flagcxRecord<T> *> profilingRecords; // NOLINT
  std::queue<flagcxRecord<T> *> profiledRecords;  // NOLINT
  pthread_mutex_t mutexAvailable{};
  pthread_cond_t condAvailable{};
  pthread_mutex_t mutexProfiling{};
  pthread_cond_t condProfiling{};
  pthread_mutex_t mutexProfiled{};

  void initSyncPrimitives();
  void destroySyncPrimitives();

public:
  flagcxTimer();
  ~flagcxTimer();

  void start();
  void stop();

  flagcxResult_t begin(const flagcxRecordKey<T> &recordKey,
                       flagcxStream_t stream, bool blocking = false);
  flagcxResult_t end(const flagcxRecordKey<T> &recordKey,
                     bool blocking = false);

  float getRecord(const flagcxRecordKey<T> &recordKey, bool blocking = false);
};

template <typename T>
void flagcxTimer<T>::initSyncPrimitives() {
  pthread_mutex_init(&mutexAvailable, nullptr);
  pthread_mutex_init(&mutexProfiling, nullptr);
  pthread_mutex_init(&mutexProfiled, nullptr);

  pthread_cond_init(&condAvailable, nullptr);
  pthread_cond_init(&condProfiling, nullptr);
}

template <typename T>
void flagcxTimer<T>::destroySyncPrimitives() {
  pthread_cond_destroy(&condAvailable);
  pthread_cond_destroy(&condProfiling);

  pthread_mutex_destroy(&mutexAvailable);
  pthread_mutex_destroy(&mutexProfiling);
  pthread_mutex_destroy(&mutexProfiled);
}

template <typename T>
void *flagcxQuery(void *flagcxTimer_) {
  auto *timer = static_cast<flagcxTimer<T> *>(flagcxTimer_);
  flagcxRecord<T> *currRecord = nullptr;

  while (true) {
    currRecord = nullptr;
    pthread_mutex_lock(&timer->mutexProfiling);
    // wait for profilingRecords not empty or stop signal
    while (!timer->stopQuery && timer->profilingRecords.empty()) {
      pthread_cond_wait(&timer->condProfiling, &timer->mutexProfiling);
    }
    // elegant exit
    if (timer->stopQuery && timer->profilingRecords.empty() && !currRecord) {
      pthread_mutex_unlock(&timer->mutexProfiling);
      break;
    }

    if (timer->profilingRecords.empty()) {
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }

    // limitation: process record one by one linearly.
    currRecord = timer->profilingRecords.front();
    if (!currRecord) {
      WARN("profilingRecords front is null, drop this record");
      timer->profilingRecords.pop();
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }
    // INFO(FLAGCX_TUNING, "Start to process record %s",
    // currRecord->recordKey.value.toString().c_str());
    flagcxResult_t res = flagcxSuccess;
    res = deviceAdaptor->eventQuery(currRecord->endEvent);
    if (res != flagcxSuccess) {
      if (res != flagcxInProgress) {
        WARN("Cannot query event, drop this record %s",
             currRecord->recordKey.value.toString().c_str());
        timer->profilingRecords.pop();
        pthread_mutex_unlock(&timer->mutexProfiling);
        continue;
      }
      // INFO(FLAGCX_TUNING, "Record %s endEvent not ready.",
      // currRecord->recordKey.value.toString().c_str());
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue; // still in progress, try again later
    }
    // when here, both beginEvent and endEvent are recorded
    res = deviceAdaptor->eventElapsedTime(&currRecord->duration,
                                          currRecord->beginEvent,
                                          currRecord->endEvent); // ms
    if (res != flagcxSuccess) {
      WARN("Cannot get elapsed time, drop this record %s",
           currRecord->recordKey.value.toString().c_str());
      timer->profilingRecords.pop();
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }

    // move currRecord from profilingRecords to profiledRecords
    timer->profilingRecords.pop();
    pthread_mutex_unlock(&timer->mutexProfiling);
    pthread_mutex_lock(&timer->mutexProfiled);
    timer->profiledRecords.push(currRecord);
    pthread_mutex_unlock(&timer->mutexProfiled);
    // INFO(FLAGCX_TUNING, "Moving record %s to profiled queue.",
    // currRecord->recordKey.value.toString().c_str());
  }
  return nullptr;
}

template <typename T>
flagcxTimer<T>::flagcxTimer() {
  initSyncPrimitives();
  for (auto &rec : flagcxRecords) {
    this->availableRecords.push(&rec);
  }
}
template <typename T>
flagcxTimer<T>::~flagcxTimer() {
  if (!stopQuery) {
    stop();
  }
  destroySyncPrimitives();
}

template <typename T>
void flagcxTimer<T>::start() {
  pthread_create(&queryThread, NULL, &flagcxQuery<T>, this);
  INFO(FLAGCX_TUNING, "flagcx timer start profiling thread");
}

template <typename T>
void flagcxTimer<T>::stop() {
  INFO(FLAGCX_TUNING, "stopping timer");
  pthread_mutex_lock(&this->mutexProfiling);
  stopQuery = true;
  pthread_cond_signal(&this->condProfiling);
  pthread_mutex_unlock(&this->mutexProfiling);
  pthread_join(queryThread, NULL);
}

template <typename T>
float flagcxTimer<T>::getRecord(const flagcxRecordKey<T> &recordKey,
                                bool blocking) {
  flagcxRecord<T> *found_record = nullptr;
  int iter = 0;
  do {
    std::queue<flagcxRecord<T> *> remaining_records;
    pthread_mutex_lock(&this->mutexProfiled);
    while (!this->profiledRecords.empty()) {
      flagcxRecord<T> *record = this->profiledRecords.front();
      this->profiledRecords.pop();
      if (found_record == nullptr && record->recordKey == recordKey) {
        found_record = record;
      } else {
        remaining_records.push(record);
      }
    }
    this->profiledRecords.swap(remaining_records);
    pthread_mutex_unlock(&this->mutexProfiled);
    // TODO: add a timeout to avoid infinite loop
    // INFO(FLAGCX_TUNING, "Searched %d times for getRecord %s.", iter,
    // recordKey.value.toString().c_str());
    iter++;
  } while (blocking && !found_record);

  if (found_record) {
    float duration = found_record->duration;
    pthread_mutex_lock(&this->mutexAvailable);
    this->availableRecords.push(found_record);
    pthread_cond_signal(&this->condAvailable);
    pthread_mutex_unlock(&this->mutexAvailable);
    return duration;
  }

  return -1.0f; // Indicate that no matching record was found
}

template <typename T>
flagcxResult_t flagcxTimer<T>::begin(const flagcxRecordKey<T> &recordKey,
                                     flagcxStream_t stream_, bool blocking) {
  flagcxRecord<T> *record = nullptr;

  pthread_mutex_lock(&this->mutexAvailable);
  while (availableRecords.empty() && blocking) {
    WARN("flagcx event is empty!");
    pthread_cond_wait(&this->condAvailable, &this->mutexAvailable);
  }
  if (!availableRecords.empty()) {
    record = availableRecords.front();
    availableRecords.pop();
  }
  pthread_mutex_unlock(&this->mutexAvailable);

  if (record) {
    record->recordKey = recordKey;
    record->stream = stream_;
    FLAGCXCHECK(deviceAdaptor->eventRecord(record->beginEvent, record->stream));
    usingRecords.push(record);
  } else {
    WARN("no available records");
    return flagcxInternalError;
  }

  return flagcxSuccess;
}

template <typename T>
flagcxResult_t flagcxTimer<T>::end(const flagcxRecordKey<T> &recordKey,
                                   bool blocking) {
  if (usingRecords.empty()) {
    return flagcxInvalidUsage;
  }

  // Find the record with recordKey
  flagcxRecord<T> *record = nullptr;
  std::queue<flagcxRecord<T> *> usingRecordsCopy;

  while (!usingRecords.empty()) {
    record = usingRecords.front();
    usingRecords.pop();
    if (record->recordKey == recordKey) {
      // Record found, update the endEvent and add it back to usingRecords
      FLAGCXCHECK(deviceAdaptor->eventRecord(record->endEvent, record->stream));
      break;
    } else {
      // Record not found, keep it in usingRecords
      usingRecordsCopy.push(record);
      record = nullptr;
    }
  }

  // Add the records from usingRecordsCopy to usingRecords
  while (!usingRecords.empty()) {
    usingRecordsCopy.push(usingRecords.front());
    usingRecords.pop();
  }
  usingRecords = usingRecordsCopy;

  if (record == nullptr) {
    WARN("no matching begin for end call");
    return flagcxInvalidUsage;
  }

  if (blocking) {
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(record->stream));
  }

  pthread_mutex_lock(&this->mutexProfiling);
  this->profilingRecords.push(record);
  pthread_cond_signal(&this->condProfiling);
  pthread_mutex_unlock(&this->mutexProfiling);

  return flagcxSuccess;
}

#endif
