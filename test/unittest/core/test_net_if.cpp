// Unit tests for network interface list parsing.
// Source: flagcx/service/include/utils.h (declared), flagcx/service/utils.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "utils.h"

TEST(NetIf, ParseSingleInterface) {
  struct netIf ifList[16];
  int count = parseStringList("eth0", ifList, 16);
  EXPECT_EQ(count, 1);
  EXPECT_STREQ(ifList[0].prefix, "eth0");
  EXPECT_EQ(ifList[0].port, -1); // no port specified
}

TEST(NetIf, ParseMultipleInterfaces) {
  struct netIf ifList[16];
  int count = parseStringList("eth0,ib0,ib1", ifList, 16);
  EXPECT_EQ(count, 3);
  EXPECT_STREQ(ifList[0].prefix, "eth0");
  EXPECT_STREQ(ifList[1].prefix, "ib0");
  EXPECT_STREQ(ifList[2].prefix, "ib1");
}

TEST(NetIf, ParseWithPort) {
  struct netIf ifList[16];
  int count = parseStringList("eth0:1234", ifList, 16);
  EXPECT_EQ(count, 1);
  EXPECT_STREQ(ifList[0].prefix, "eth0");
  EXPECT_EQ(ifList[0].port, 1234);
}

TEST(NetIf, MatchIfListExact) {
  struct netIf ifList[16];
  int count = parseStringList("eth0,ib0", ifList, 16);
  ASSERT_EQ(count, 2);

  // Exact match should succeed
  EXPECT_TRUE(matchIfList("eth0", -1, ifList, count, true));
  EXPECT_TRUE(matchIfList("ib0", -1, ifList, count, true));

  // Non-matching should fail
  EXPECT_FALSE(matchIfList("eth1", -1, ifList, count, true));

  // Empty list always matches
  EXPECT_TRUE(matchIfList("anything", -1, ifList, 0, true));
}
