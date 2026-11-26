#ifndef FLAGCX_BUFFER_OWNER_H_
#define FLAGCX_BUFFER_OWNER_H_

// Allocation ownership is independent of allocator, mapping mode and views.
enum flagcxBufferOwnership { flagcxBufferExternal = 0, flagcxBufferInternal };

#endif
