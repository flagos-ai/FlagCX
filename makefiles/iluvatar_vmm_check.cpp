// Compile-only SDK check. This program is never linked or executed.
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#ifndef FLAGCX_VMM_CHECK_STAGE
#define FLAGCX_VMM_CHECK_STAGE 0
#endif

int main() {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUmemGenericAllocationHandle handle = {};
  CUdeviceptr address = 0;
  size_t bytes = 0;

#if FLAGCX_VMM_CHECK_STAGE == 0 || FLAGCX_VMM_CHECK_STAGE == 2
  (void)cuMemGetAllocationGranularity(&bytes, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  (void)cuMemCreate(&handle, bytes, &prop, 0);
  (void)cuMemAddressReserve(&address, bytes, bytes, 0, 0);
  (void)cuMemMap(address, bytes, 0, handle, 0);
  (void)cuMemSetAccess(address, bytes, &access, 1);
  unsigned sync = 1;
  (void)cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address);
  (void)cuMemGetAddressRange(&address, &bytes, address);
  (void)cuMemUnmap(address, bytes);
  (void)cuMemAddressFree(address, bytes);
  (void)cuMemRelease(handle);
  (void)CUDA_SUCCESS;
  (void)CUDA_ERROR_NOT_SUPPORTED;
#endif

#if FLAGCX_VMM_CHECK_STAGE == 0 || FLAGCX_VMM_CHECK_STAGE == 3
  int fd = -1;
  (void)cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  (void)cuMemImportFromShareableHandle(
      &handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
#endif
  return 0;
}
