#include <gtest/gtest.h>

#include "p2p.h"

TEST(P2pTransferMode, WaitsUntilBothEndpointsPublishRegistrationState) {
  int mode = flagcxP2pTransferFifo;
  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationUnknown,
                                        flagcxP2pRegistrationDisabled, &mode),
            flagcxSuccess);
  EXPECT_EQ(mode, flagcxP2pTransferUnknown);
}

TEST(P2pTransferMode, UsesFifoWhenNeitherEndpointIsRegistered) {
  int mode = flagcxP2pTransferUnknown;
  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationDisabled,
                                        flagcxP2pRegistrationDisabled, &mode),
            flagcxSuccess);
  EXPECT_EQ(mode, flagcxP2pTransferFifo);
}

TEST(P2pTransferMode, UsesReadWhenOnlySenderIsRegistered) {
  int mode = flagcxP2pTransferUnknown;
  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationEnabled,
                                        flagcxP2pRegistrationDisabled, &mode),
            flagcxSuccess);
  EXPECT_EQ(mode, flagcxP2pTransferRead);
}

TEST(P2pTransferMode, UsesWriteWhenReceiverIsRegistered) {
  int mode = flagcxP2pTransferUnknown;
  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationDisabled,
                                        flagcxP2pRegistrationEnabled, &mode),
            flagcxSuccess);
  EXPECT_EQ(mode, flagcxP2pTransferWrite);

  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationEnabled,
                                        flagcxP2pRegistrationEnabled, &mode),
            flagcxSuccess);
  EXPECT_EQ(mode, flagcxP2pTransferWrite);
}

TEST(P2pTransferMode, RejectsInvalidArguments) {
  int mode = flagcxP2pTransferUnknown;
  EXPECT_EQ(
      flagcxP2pSelectTransferMode(99, flagcxP2pRegistrationEnabled, &mode),
      flagcxInvalidArgument);
  EXPECT_EQ(flagcxP2pSelectTransferMode(flagcxP2pRegistrationEnabled,
                                        flagcxP2pRegistrationEnabled, nullptr),
            flagcxInvalidArgument);
}
