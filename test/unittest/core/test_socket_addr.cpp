// Unit tests for socket address parsing.
// Source: flagcx/core/include/socket.h (declared), flagcx/core/socket.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "socket.h"

TEST(SocketAddr, ParseIPv4WithPort) {
  union flagcxSocketAddress addr;
  memset(&addr, 0, sizeof(addr));

  flagcxResult_t result =
      flagcxSocketGetAddrFromString(&addr, "127.0.0.1:12345");
  EXPECT_EQ(result, flagcxSuccess);

  // Verify it's an IPv4 address
  EXPECT_EQ(addr.sa.sa_family, AF_INET);
  // Verify port (stored in network byte order)
  EXPECT_EQ(ntohs(addr.sin.sin_port), 12345);
}

TEST(SocketAddr, ParseIPv4WithoutPort) {
  union flagcxSocketAddress addr;
  memset(&addr, 0, sizeof(addr));

  flagcxResult_t result = flagcxSocketGetAddrFromString(&addr, "192.168.1.1");
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(addr.sa.sa_family, AF_INET);
}

TEST(SocketAddr, ToString) {
  union flagcxSocketAddress addr;
  memset(&addr, 0, sizeof(addr));

  flagcxResult_t result = flagcxSocketGetAddrFromString(&addr, "10.0.0.1:8080");
  ASSERT_EQ(result, flagcxSuccess);

  char buf[SOCKET_NAME_MAXLEN] = {0};
  const char *str = flagcxSocketToString(&addr, buf);
  ASSERT_NE(str, nullptr);
  EXPECT_GT(strlen(buf), 0u);
  // The string should contain the IP address
  EXPECT_NE(strstr(buf, "10.0.0.1"), nullptr);
}

TEST(SocketAddr, ParseNullInput) {
  union flagcxSocketAddress addr;
  memset(&addr, 0, sizeof(addr));

  flagcxResult_t result = flagcxSocketGetAddrFromString(&addr, nullptr);
  // Should handle null gracefully (return error or no-op)
  EXPECT_NE(result, flagcxSuccess);
}
