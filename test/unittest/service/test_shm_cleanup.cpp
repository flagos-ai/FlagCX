#include "shmutils.h"
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

TEST(ShmCleanupTest, FailedUnlinkRetainsMappingAndDescriptorForRetry) {
  char path[SHM_PATH_MAX] = {};
  void *mapping = nullptr;
  flagcxShmIpcDesc_t desc = {};
  ASSERT_EQ(
      flagcxShmOpen(path, sizeof(path), 64, &mapping, nullptr, 1, &desc.handle),
      flagcxSuccess);
  ASSERT_EQ(unlink(path), 0);
  ASSERT_EQ(mkdir(path, 0700), 0);
  EXPECT_EQ(flagcxShmIpcClose(&desc), flagcxSystemError);
  EXPECT_NE(desc.handle, nullptr);
  ((char *)mapping)[0] = 17;
  ASSERT_EQ(rmdir(path), 0);
  EXPECT_EQ(flagcxShmIpcClose(&desc), flagcxSuccess);
  EXPECT_EQ(desc.handle, nullptr);
  EXPECT_EQ(flagcxShmIpcClose(&desc), flagcxSuccess);
}

TEST(ShmCleanupTest, ExportAndImportMappingsCloseIndependently) {
  flagcxShmIpcDesc_t exported = {}, imported = {};
  void *a = nullptr, *b = nullptr;
  ASSERT_EQ(flagcxShmAllocateShareableBuffer(64, &exported, &a, nullptr),
            flagcxSuccess);
  ASSERT_EQ(flagcxShmImportShareableBuffer(&exported, &b, nullptr, &imported),
            flagcxSuccess);
  ((char *)a)[0] = 29;
  EXPECT_EQ(flagcxShmIpcClose(&exported), flagcxSuccess);
  EXPECT_EQ(((char *)b)[0], 29);
  EXPECT_EQ(flagcxShmIpcClose(&imported), flagcxSuccess);
}
