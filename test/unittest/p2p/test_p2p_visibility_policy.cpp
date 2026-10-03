/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "p2p_visibility.h"

TEST(P2pVisibilityPolicyTest, HostAndZeroByteReadsNeedNoFlush) {
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_READ_REQUIRES_FLUSH,
                                            FLAGCX_NET_GDR_FLUSH_NONE,
                                            FLAGCX_PTR_HOST, 4096, 0),
            flagcxSuccess);
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_READ_REQUIRES_FLUSH,
                                            FLAGCX_NET_GDR_FLUSH_NONE,
                                            FLAGCX_PTR_CUDA, 0, 0),
            flagcxSuccess);
}

TEST(P2pVisibilityPolicyTest, GpuReadWithoutRequirementNeedsNoFlush) {
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_FLUSH_NONE,
                                            FLAGCX_NET_GDR_FLUSH_NONE,
                                            FLAGCX_PTR_CUDA, 4096, 0),
            flagcxSuccess);
}

TEST(P2pVisibilityPolicyTest, RequiredGpuReadFailsClosed) {
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_READ_REQUIRES_FLUSH,
                                            FLAGCX_NET_GDR_FLUSH_NONE,
                                            FLAGCX_PTR_CUDA, 4096, 0),
            flagcxNotSupported);
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_READ_REQUIRES_FLUSH,
                                            FLAGCX_NET_GDR_FLUSH_READ,
                                            FLAGCX_PTR_CUDA, 4096, 0),
            flagcxNotSupported);
  EXPECT_EQ(flagcxP2pValidateReadVisibility(FLAGCX_GDR_READ_REQUIRES_FLUSH,
                                            FLAGCX_NET_GDR_FLUSH_READ,
                                            FLAGCX_PTR_CUDA, 4096, 1),
            flagcxSuccess);
}
