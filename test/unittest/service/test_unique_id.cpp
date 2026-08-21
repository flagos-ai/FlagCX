#include <gtest/gtest.h>

#include "flagcx.h"
#include "flagcx_ccl_adaptor.h"

namespace {

constexpr size_t kCann90HcclRootInfoBytes = 4108;

static_assert(sizeof(flagcxUniqueId) == FLAGCX_UNIQUE_ID_BYTES,
              "flagcxUniqueId size must match FLAGCX_UNIQUE_ID_BYTES");
static_assert(sizeof(flagcxInnerUniqueId) == FLAGCX_INNER_UNIQUE_ID_BYTES,
              "flagcxInnerUniqueId size must match its capacity macro");
static_assert(sizeof(flagcxInnerUniqueId) >= kCann90HcclRootInfoBytes,
              "flagcxInnerUniqueId must accommodate CANN 9.0 HcclRootInfo");
static_assert(alignof(flagcxInnerUniqueId) >= alignof(uint64_t),
              "flagcxInnerUniqueId must support aligned native ID casts");

TEST(UniqueId, KeepsPublicAbiAndAccommodatesCannInternally) {
  EXPECT_EQ(sizeof(flagcxUniqueId), 256u);
  EXPECT_EQ(sizeof(flagcxInnerUniqueId), 4120u);
  EXPECT_GE(sizeof(flagcxInnerUniqueId), kCann90HcclRootInfoBytes);
  EXPECT_GE(alignof(flagcxInnerUniqueId), alignof(uint64_t));
}

} // namespace
