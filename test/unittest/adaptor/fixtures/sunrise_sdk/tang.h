#pragma once
#include <stddef.h>

typedef void *tangStream_t;
typedef void *tangEvent_t;
typedef struct {
  char bytes[64];
} tangIpcMemHandle_t;
typedef enum { tangSuccess, tangErrorNotReady } tangError_t;
typedef enum {
  tangMemcpyHostToDevice,
  tangMemcpyDeviceToHost,
  tangMemcpyDeviceToDevice
} tangMemcpyKind;
typedef enum {
  tangStreamCaptureModeGlobal,
  tangStreamCaptureModeRelaxed
} tangStreamCaptureMode;
enum {
  tangStreamNonBlocking = 1,
  tangEventWaitDefault = 0,
  tangEventDefault = 0,
  tangEventDisableTiming = 2,
  tangEventRecordDefault = 0,
  tangIpcMemLazyEnablePeerAccess = 1
};
struct tangDeviceProp {
  char name[256];
  int pciBusID, pciDeviceID, pciDomainID;
};

tangError_t tangDeviceSynchronize();
tangError_t tangMemcpy(void *, const void *, size_t, tangMemcpyKind);
tangError_t tangMemcpyAsync(void *, const void *, size_t, tangMemcpyKind,
                            tangStream_t);
tangError_t tangMemset(void *, int, size_t);
tangError_t tangMemsetAsync(void *, int, size_t, tangStream_t);
tangError_t tangMallocHost(void **, size_t);
tangError_t tangMalloc(void **, size_t);
tangError_t tangMallocAsync(void **, size_t, tangStream_t);
tangError_t tangFreeHost(void *);
tangError_t tangFree(void *);
tangError_t tangSetDevice(int);
tangError_t tangGetDevice(int *);
tangError_t tangGetDeviceCount(int *);
tangError_t tangHostGetDevicePointer(void **, void *, unsigned);
tangError_t tangThreadExchangeStreamCaptureMode(tangStreamCaptureMode *);
tangError_t tangStreamCreateWithFlags(tangStream_t *, unsigned);
tangError_t tangStreamSynchronize(tangStream_t);
tangError_t tangStreamDestroy(tangStream_t);
tangError_t tangStreamQuery(tangStream_t);
tangError_t tangStreamWaitEvent(tangStream_t, tangEvent_t, unsigned);
tangError_t tangEventCreateWithFlags(tangEvent_t *, unsigned);
tangError_t tangEventDestroy(tangEvent_t);
tangError_t tangEventRecordWithFlags(tangEvent_t, tangStream_t = nullptr,
                                     unsigned = 0);
tangError_t tangEventSynchronize(tangEvent_t);
tangError_t tangEventQuery(tangEvent_t);
tangError_t tangEventElapsedTime(float *, tangEvent_t, tangEvent_t);
tangError_t tangIpcGetMemHandle(tangIpcMemHandle_t *, void *);
tangError_t tangIpcOpenMemHandle(void **, tangIpcMemHandle_t, unsigned);
tangError_t tangIpcCloseMemHandle(void *);
tangError_t tangGetDeviceProperties(tangDeviceProp *, int);
tangError_t tangDeviceGetPCIBusId(char *, int, int);
tangError_t tangDeviceGetByPCIBusId(int *, const char *);
tangError_t tangLaunchHostFunc(tangStream_t, void (*)(void *), void *);
