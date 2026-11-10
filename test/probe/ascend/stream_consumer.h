#pragma once

#include <stdint.h>

// Enqueue one device read of the payload and store {mismatches, checked_bytes}.
// The caller must enqueue this immediately after aclrtValueWait on one stream.
void launchStreamConsumer(void *payload, uint32_t *status, uint8_t expected,
                          void *stream);

// Enqueue a device producer writing the receiver-owned payload.
void launchStreamProducer(void *payload, uint8_t value, void *stream);

// Enqueue a separate, ordered device task that writes the 64-bit signal.
void launchStreamSignal(void *signal, uint64_t value, void *stream);

// Read the signal on device into status[2:4], after the payload consumer.
void launchStreamSignalReader(void *signal, uint32_t *status, void *stream);
