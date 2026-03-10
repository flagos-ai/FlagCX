#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      return 0;
  }
}

FLAGCX_DEVICE_DECORATOR void
flagcxDeviceTrigger::setValue(uint64_t addr, uint64_t count, uint64_t peerRank,
                              uint64_t datatype, uint64_t type) {
  fst = addr;
  snd = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
            << flagcxDeviceTriggerOffCount |
        (peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
            << flagcxDeviceTriggerOffPeerRank |
        (datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
            << flagcxDeviceTriggerOffDatatype |
        (type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
            << flagcxDeviceTriggerOffPrim;
}