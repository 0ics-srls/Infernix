#pragma once

// The mapped-memory request through which a layer call on the main GPU (the RTX 4090) hands the
// experts a second GPU holds (the V100) to that GPU's expert service (cuda/remote_experts.h). Shared
// by the device kernels (moe_layer.cu) and the service; plain data, no host-only types.
//
// Unlike the CPU request (miss_request.h), a job may have any number of columns: the service runs
// the same exact kernels on its device (the narrow route, passes of eight columns), so decode,
// verification and prefill calls all use it. The call's x is published whole beside the request.

#include <cstdint>

namespace infernix::ops::offloaded_moe {

inline constexpr int kMaxRemoteJobs    = 512;           // one per routed expert of a layer
inline constexpr int kMaxRemoteColumns = 4096;          // columns of a call (a prefill chunk)
inline constexpr int kMaxRemoteEntries = 10 * 4096;     // routed (column, slot) entries: top_k * columns

// Written by the device: every field, a system-scope fence, then `sequence` (never 0). The service
// answers by writing the same value to the channel's done word after every output is written.
// Job j's entries are entry[first[j] .. first[j] + count[j]); an entry is t * top_k + slot, its
// column t = entry / top_k. y column i holds the output of entry[i].
struct alignas(64) RemoteRequest {
    std::uint32_t sequence;
    std::int32_t layer;
    std::int32_t columns;
    std::int32_t top_k;
    std::int32_t jobs;
    std::int32_t entries;
    std::int32_t expert[kMaxRemoteJobs];
    std::int32_t first[kMaxRemoteJobs];
    std::int32_t count[kMaxRemoteJobs];
    std::int32_t entry[kMaxRemoteEntries];
};

// The main GPU's bookkeeping of the outstanding request (device memory the service allocates beside
// the channel): the published sequence (0: nothing published) and a device copy of the entry list,
// which the wait kernel reads to place y's columns.
struct RemoteCall {
    std::int32_t pending;
    std::int32_t entries;
    std::int32_t jobs;
    std::int32_t pad;
    std::int32_t entry[kMaxRemoteEntries];
};

} // namespace infernix::ops::offloaded_moe
