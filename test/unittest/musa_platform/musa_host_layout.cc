/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "device_api/platform_traits.h"
#include <cstdio>

using Traits = PlatformTraits<MusaPlatform>;

static_assert(FLAGCX_CHECK_DEVICE_CC == 0,
              "the host probe must use the ordinary host compiler");
static_assert(FLAGCX_SIMT_WIDTH == 32, "MUSA host SIMT layout changed");
static_assert(sizeof(flagcxLaneMask_t) == 8, "lane mask ABI changed");
static_assert(sizeof(Traits::CoopAny) == 24, "opaque CoopAny layout changed");
static_assert(alignof(Traits::CoopAny) == 8, "CoopAny alignment changed");
static_assert(Traits::Intrin::fullMask() == 0xffffffffull,
              "MUSA full mask changed");
static_assert(Traits::CoopTileSpan::supportsSync(0, 2, 64),
              "a full-CTA span must support block synchronization");
static_assert(Traits::CoopTileSpan::supportsSync(1, 1, 128),
              "a single-warp span must support warp synchronization");
static_assert(!Traits::CoopTileSpan::supportsSync(0, 2, 128),
              "a partial-CTA multi-warp span requires a subgroup barrier");
static_assert(!Traits::CoopTileSpan::supportsSync(1, 2, 128),
              "an interior multi-warp span requires a subgroup barrier");
static_assert(!Traits::CoopTileSpan::supportsSync(-1, 1, 64),
              "a span cannot start before the first warp");
static_assert(!Traits::CoopTileSpan::supportsSync(0, 0, 64),
              "an empty span cannot synchronize");
static_assert(!Traits::CoopTileSpan::supportsSync(1, 2, 64),
              "a span cannot extend beyond the block");
static_assert(!Traits::CoopTileSpan::supportsSync(0, 2, 80),
              "a partial final warp cannot turn a span into a full CTA");

int main() {
  std::printf("MUSA host layout PASS: SIMT=%d mask=%zu CoopAny=%zu align=%zu\n",
              FLAGCX_SIMT_WIDTH, sizeof(flagcxLaneMask_t),
              sizeof(Traits::CoopAny), alignof(Traits::CoopAny));
  return 0;
}
