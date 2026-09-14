#include "register.h"
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

TEST(IpcHandleStorageTest, StoresSupportedRuntimeSizesAndClearsTail) {
  constexpr size_t sizes[] = {64, 80, FLAGCX_IPC_HANDLE_SIZE};

  for (size_t handleSize : sizes) {
    std::array<unsigned char, FLAGCX_IPC_HANDLE_SIZE> source = {};
    for (size_t i = 0; i < handleSize; ++i)
      source[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);

    flagcxIpcHandleData storage;
    memset(&storage, 0xa5, sizeof(storage));
    ASSERT_EQ(flagcxStoreIpcHandle(&storage, source.data(), handleSize),
              flagcxSuccess);
    EXPECT_EQ(memcmp(storage.reserved, source.data(), handleSize), 0);
    for (size_t i = handleSize; i < sizeof(storage); ++i)
      EXPECT_EQ(static_cast<unsigned char>(storage.reserved[i]), 0);
  }
}

TEST(IpcHandleStorageTest, RejectsInvalidRuntimeSizes) {
  std::array<unsigned char, FLAGCX_IPC_HANDLE_SIZE + 1> source = {};
  flagcxIpcHandleData storage = {};

  EXPECT_EQ(flagcxStoreIpcHandle(&storage, source.data(), 0),
            flagcxNotSupported);
  EXPECT_EQ(flagcxStoreIpcHandle(&storage, source.data(), source.size()),
            flagcxNotSupported);
  EXPECT_EQ(flagcxStoreIpcHandle(nullptr, source.data(), 64),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxStoreIpcHandle(&storage, nullptr, 64), flagcxInvalidArgument);
}

} // namespace
