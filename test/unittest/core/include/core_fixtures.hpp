#pragma once

#include "flagcx.h"
#include "flagcx_test.hpp"

class FlagCXTopoTest : public FlagCXTest {
protected:
  void SetUp() override;
  void TearDown() override;

  flagcxHandlerGroup_t handler;
};
