// Unit tests for PCI Bus ID conversion utilities.
// Source: flagcx/service/include/utils.h (declared), flagcx/service/utils.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "adaptor.h"
#include "comm.h"
#include "flagcx_common.h"
#include "topo.h"
#include "utils.h"
#include "xml.h"

namespace {

class BusIdLookup : public ::testing::Test {
protected:
  void SetUp() override {
    savedAdaptor = deviceAdaptor;
    deviceAdaptor = &adaptor;
  }
  void TearDown() override { deviceAdaptor = savedAdaptor; }

  flagcxDeviceAdaptor adaptor = {};
  flagcxDeviceAdaptor *savedAdaptor = nullptr;
};

TEST_F(BusIdLookup, MissingAdaptorReturnsUnsupported) {
  deviceAdaptor = nullptr;
  int64_t id = -1;
  EXPECT_EQ(getBusId(0, &id), flagcxNotSupported);
  EXPECT_EQ(id, -1);
}

TEST_F(BusIdLookup, MissingCallbackYieldsStableNonPciIdentity) {
  // Backends without a PCI bus ID still need an identity that distinguishes
  // devices on one host, because that is all comm->busId is compared for.
  int64_t first = -1;
  int64_t second = -1;
  EXPECT_EQ(getBusId(2, &first), flagcxSuccess);
  EXPECT_EQ(getBusId(2, &second), flagcxSuccess);
  EXPECT_EQ(first, second) << "identity must be stable for one device";

  int64_t other = -1;
  EXPECT_EQ(getBusId(3, &other), flagcxSuccess);
  EXPECT_NE(first, other) << "distinct devices must not alias";
}

TEST_F(BusIdLookup, NonPciIdentityCannotResolveAsSysfsPath) {
  // The synthetic domain must stay outside the range a real BDF can take, so
  // that a consumer treating it as a /sys/class/pci_bus location fails
  // instead of matching an unrelated device.
  int64_t id = -1;
  ASSERT_EQ(getBusId(1, &id), flagcxSuccess);

  char text[64] = {0};
  ASSERT_EQ(int64ToBusId(id, text, sizeof(text)), flagcxSuccess);
  EXPECT_STREQ(text, "ffff:00:01.0");

  int64_t roundTrip = 0;
  ASSERT_EQ(busIdToInt64(text, &roundTrip), flagcxSuccess);
  EXPECT_EQ(roundTrip, id);
}

TEST_F(BusIdLookup, NonPciIdentityRejectsUnaddressableDeviceIndex) {
  // Truncating the index would hand two devices the same identity and trip
  // the duplicate-device check, so it is refused instead.
  int64_t id = -1;
  EXPECT_EQ(getBusId(0x100, &id), flagcxNotSupported);
  EXPECT_EQ(id, -1);
}

TEST_F(BusIdLookup, CallbackErrorIsPreserved) {
  adaptor.getDevicePciBusId = [](char *, int, int) {
    return flagcxUnhandledDeviceError;
  };
  int64_t id = -1;
  EXPECT_EQ(getBusId(0, &id), flagcxUnhandledDeviceError);
  EXPECT_EQ(id, -1);
}

TEST_F(BusIdLookup, ValidCallbackConvertsBusId) {
  adaptor.getDevicePciBusId = [](char *busId, int len, int dev) {
    if (dev != 3 || len < static_cast<int>(sizeof("0000:41:00.1")))
      return flagcxInvalidArgument;
    std::strcpy(busId, "0000:41:00.1");
    return flagcxSuccess;
  };
  int64_t id = -1;
  EXPECT_EQ(getBusId(3, &id), flagcxSuccess);
  EXPECT_EQ(id, 0x41001);
}

TEST_F(BusIdLookup, InvalidArgumentsDoNotCallBackend) {
  // A missing callback must not hide invalid arguments.
  int64_t id = -1;
  EXPECT_EQ(getBusId(-1, &id), flagcxInvalidArgument);
  EXPECT_EQ(getBusId(0, nullptr), flagcxInvalidArgument);
  EXPECT_EQ(id, -1);
}

TEST_F(BusIdLookup, MissingReverseCallbackRejectsTopologyDiscovery) {
  flagcxHeteroComm comm = {};
  flagcxXml xml = {};
  EXPECT_EQ(flagcxTopoGetXmlTopo(&comm, &xml), flagcxNotSupported);
  EXPECT_EQ(xml.maxIndex, 0);
}

} // namespace

TEST(BusId, Int64ToBusIdAndBack) {
  // Known ID value — convert to string and back, verify round-trip
  int64_t origId = 0x00041001; // domain=0, bus=0x41, dev=0x00, func=1
  char busId[64] = {0};

  flagcxResult_t result = int64ToBusId(origId, busId, sizeof(busId));
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_GT(strlen(busId), 0u);

  // Parse back to int64
  int64_t parsedId = 0;
  result = busIdToInt64(busId, &parsedId);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(parsedId, origId);
}

TEST(BusId, KnownFormat) {
  // int64ToBusId extracts: domain=id>>20, bus=(id&0xff000)>>12,
  //                         dev=(id&0xff0)>>4, func=id&0xf
  // For id=0x00041001: domain=0, bus=0x41, dev=0x00, func=1
  int64_t id = 0x00041001;
  char busId[64] = {0};

  flagcxResult_t result = int64ToBusId(id, busId, sizeof(busId));
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_STREQ(busId, "0000:41:00.1");
}

TEST(BusId, EightDigitDomainRoundTrip) {
  int64_t id = -1;
  ASSERT_EQ(busIdToInt64("12345678:3d:00.0", &id), flagcxSuccess);
  char busId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE] = {};
  EXPECT_EQ(int64ToBusId(id, busId, sizeof(busId)), flagcxSuccess);
  EXPECT_STREQ(busId, "12345678:3d:00.0");
  EXPECT_EQ(int64ToBusId(id, busId, 16), flagcxInvalidArgument);
}

TEST(BusId, InvalidBusId) {
  // busIdToInt64 handles non-hex characters by stopping early.
  // "invalid" has no valid hex digits to parse, result should be 0.
  int64_t id = -1;
  flagcxResult_t result = busIdToInt64("invalid", &id);
  EXPECT_EQ(result, flagcxSuccess); // function always returns success
  // strtol("", NULL, 16) returns 0
  EXPECT_EQ(id, 0);
}
