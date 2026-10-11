#pragma once

// The expert service of a second GPU (the V100 of the V100 + RTX 4090 pair). It holds a set of routed
// experts in its own memory and computes them for the main GPU's layer calls: the main GPU publishes
// a call's x and the held experts' entries (MoeRemoteChannel, cpu/remote_request.h), computes its
// own experts and the shared expert meanwhile, and places this service's outputs once it answers.
//
// The service runs the same exact kernels as the main GPU (ops::moe_experts, the narrow route, built
// for both architectures in this archive), so an expert's output has the same bits on either GPU and
// on the CPU (design §16.2): the split between the GPUs is placement, never arithmetic.
//
// Which experts it holds is the caller's choice (the Program ranks them by routing frequency); the
// main GPU's held tables (I32 [layers][experts]: slot or -1) are uploaded by load().

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/remote_request.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace infernix::ops::offloaded_moe {

class RemoteExpertService {
public:
    struct Options {
        int device      = -1; // the GPU that holds and computes the experts
        int main_device = 0;  // the GPU that runs the Program (where the held tables live)
        std::uint32_t layers  = 0;
        std::uint32_t experts = 0;
        std::uint64_t record_stride = 0;
        ExpertActivation activation = ExpertActivation::kA4;
        std::vector<const ExpertScales*> scales; // host [experts] per layer
        // Device memory left free on the remote GPU beside the records: the service's buffers
        // (x, outputs, workspace: ~0.5 GiB), the CUDA context and a margin.
        std::uint64_t reserve_bytes = 1536ULL << 20;
        // At most this many records (0: as many as fit).
        std::uint32_t max_records = 0;
    };

    explicit RemoteExpertService(Options options);
    ~RemoteExpertService();
    RemoteExpertService(const RemoteExpertService&)            = delete;
    RemoteExpertService& operator=(const RemoteExpertService&) = delete;

    // Records the remote GPU has room for.
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    // Before the first call: loads `keys` (layer * experts + expert, at most capacity()), each record
    // read by `read` into a pinned buffer of record_stride bytes, uploads the held tables to the main
    // GPU and starts serving.
    using Read = std::function<void(std::uint32_t key, std::uint8_t* destination)>;
    void load(std::span<const std::uint32_t> keys, const Read& read);

    [[nodiscard]] bool holds(std::uint32_t key) const noexcept { return key < slot_of_.size() && slot_of_[key] >= 0; }
    [[nodiscard]] std::uint32_t held() const noexcept { return held_; }
    [[nodiscard]] MoeRemoteChannel channel(int layer) const;

    struct Stats {
        std::uint64_t requests = 0, jobs = 0, entries = 0;
        std::uint64_t busy_ns  = 0; // request seen to answer written
    };
    [[nodiscard]] Stats stats() const;
    // Why the service stopped serving; empty while it serves (the main GPU's wait then times out).
    [[nodiscard]] std::string failure() const;

private:
    void serve();
    void record_failure(std::string what) noexcept;

    Options options_;
    std::uint32_t capacity_ = 0, held_ = 0;
    std::vector<std::int32_t> slot_of_; // [layers * experts]: slot on the remote GPU or -1

    // Remote GPU memory.
    std::uint8_t* records_       = nullptr; // [capacity][record_stride]
    std::int32_t* frames_        = nullptr; // [layers][experts]: slot or -1
    ExpertScales* scales_        = nullptr; // [layers][experts]
    std::uint16_t* x_device_     = nullptr; // BF16 [H, kMaxRemoteColumns]
    std::uint16_t* outputs_      = nullptr; // BF16 [H, kMaxRemoteEntries]
    std::int32_t* dispatch_      = nullptr; // counts, offsets, cursor, entries, jobs, job_count, entry list
    void* workspace_             = nullptr;
    std::size_t dispatch_ints_   = 0;
    // Main GPU memory.
    std::int32_t* held_tables_   = nullptr; // [layers][experts]
    RemoteCall* call_            = nullptr;
    std::uint32_t* sequence_     = nullptr;
    // Mapped host memory (portable: both GPUs reach it).
    RemoteRequest* request_      = nullptr;
    std::uint16_t* x_            = nullptr;
    std::uint16_t* y_            = nullptr;
    std::uint32_t* done_         = nullptr; // done, status, heartbeat
    std::int32_t* staging_       = nullptr; // pinned host: the dispatch block before its upload

    std::atomic<bool> stop_{false};
    mutable std::mutex mutex_;
    std::string failure_;
    Stats stats_;
    std::thread thread_;
};

} // namespace infernix::ops::offloaded_moe
