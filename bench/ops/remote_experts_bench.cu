// The second GPU's expert service (ops/offloaded_sparse_moe/cuda/remote_experts.h) on one GPU,
// driven the way the main GPU drives it: synthetic requests written to the mapped request, the
// answer awaited on the done word. Prints the service's per-request times; run it under ncu to
// profile the remote GPU's kernels (a process with both GPUs cannot be profiled on the V100).
// With "check" it first compares the record layout and the row-pair layout bit for bit.
//
//   infernix_remote_experts_bench [requests=2000] [columns=5] [jobs=7] [max-entries-per-job=2] [check]

#include "ops/offloaded_sparse_moe/cuda/remote_experts.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace infernix::ops;
using namespace infernix::ops::offloaded_moe;

namespace {

constexpr std::uint32_t kLayers = 2, kExperts = 512;
constexpr int kTopK = 10;
constexpr std::uint32_t kHeld = 64;

std::unique_ptr<RemoteExpertService> make_service(bool row_pairs, const std::vector<ExpertScales>& scales) {
    RemoteExpertService::Options o;
    o.device = 0;
    o.main_device = 0;
    o.layers = kLayers;
    o.experts = kExperts;
    o.record_stride = kRecordBytes;
    o.scales = {scales.data(), scales.data()};
    o.max_records = kHeld;
    o.row_pair_units = row_pairs;
    o.fast_call      = row_pairs; // the reference: record layout and moe_experts' record path
    auto service = std::make_unique<RemoteExpertService>(o);
    std::vector<std::uint32_t> keys;
    for (std::uint32_t k = 0; k < kHeld; ++k) { keys.push_back(k); }
    service->load(keys, [](std::uint32_t key, std::uint8_t* d) {
        std::uint64_t z = 0x9E3779B97F4A7C15ULL * (key + 1);
        for (std::uint64_t i = 0; i + 8 <= kRecordBytes; i += 8) {
            z = z * 6364136223846793005ULL + 1442695040888963407ULL;
            std::memcpy(d + i, &z, 8);
        }
        // E4M3 row scales between 0.5 and 2 (exponent field 6..8) in every unit's scale bytes.
        for (std::uint64_t u = 0; u + kUnitBytes <= kRecordBytes; u += kUnitBytes) {
            for (int b = 128; b < 144; ++b) { d[u + b] = static_cast<std::uint8_t>(0x30 + ((u / 144 + b) % 24)); }
        }
    });
    return service;
}

// Writes request q into the channel's mapped memory and waits for the answer.
void run(const MoeRemoteChannel& ch, int q, int columns, int jobs, int per_job, std::uint32_t& sequence) {
    auto* r = ch.request;
    int at  = 0;
    for (int j = 0; j < jobs; ++j) {
        r->expert[j] = (j * 9 + q) % static_cast<int>(kHeld);
        const int n  = 1 + (j % per_job);
        r->first[j]  = at;
        r->count[j]  = n;
        for (int c = 0; c < n; ++c) { r->entry[at++] = ((c + j) % columns) * kTopK + (j % kTopK); }
    }
    r->layer   = 0;
    r->columns = columns;
    r->top_k   = kTopK;
    r->jobs    = jobs;
    r->entries = at;
    sequence   = sequence + 1 == 0 ? 1 : sequence + 1;
    std::atomic_thread_fence(std::memory_order_release);
    reinterpret_cast<volatile std::uint32_t*>(&r->sequence)[0] = sequence;
    const auto* done = reinterpret_cast<const volatile std::uint32_t*>(ch.done);
    while (*done != sequence) {}
    std::atomic_thread_fence(std::memory_order_acquire);
}

void fill_x(const MoeRemoteChannel& ch, int columns) {
    std::mt19937 rng(11);
    for (int i = 0; i < 2560 * columns; ++i) {
        const float v = std::uniform_real_distribution<float>(-2.0F, 2.0F)(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        ch.x[i] = static_cast<std::uint16_t>(bits >> 16);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int requests = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int columns  = argc > 2 ? std::atoi(argv[2]) : 5;
    const int jobs     = argc > 3 ? std::atoi(argv[3]) : 7;
    const int per_job  = argc > 4 ? std::atoi(argv[4]) : 2;
    const bool check   = argc > 5 && std::string(argv[5]) == "check";
    std::vector<ExpertScales> scales(kExperts);
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        scales[e] = {0.02F + 0.0001F * e, 0.02F + 0.0001F * e, 0.05F, 0.001F, 0.0011F, 0.0009F};
    }
    if (check) {
        auto plain = make_service(false, scales);
        auto pairs = make_service(true, scales);
        const MoeRemoteChannel a = plain->channel(0), b = pairs->channel(0);
        fill_x(a, columns);
        fill_x(b, columns);
        std::uint32_t sa = 0, sb = 0;
        long long differ = 0, total = 0;
        for (int q = 0; q < 64; ++q) {
            run(a, q, columns, jobs, per_job, sa);
            run(b, q, columns, jobs, per_job, sb);
            const int entries = a.request->entries;
            for (int i = 0; i < 2560 * entries; ++i) {
                ++total;
                if (a.y[i] != b.y[i]) { ++differ; }
            }
        }
        std::printf("check: %lld of %lld outputs differ between the record path and the fast row-pair path; failures: '%s' '%s'\n",
                    differ, total, plain->failure().c_str(), pairs->failure().c_str());
    }
    auto service = make_service(true, scales);
    const MoeRemoteChannel ch = service->channel(0);
    fill_x(ch, columns);
    std::uint32_t sequence = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int q = 0; q < requests; ++q) { run(ch, q, columns, jobs, per_job, sequence); }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto st = service->stats();
    std::printf("%d requests of %d jobs, %d columns: %.1f us per request round trip; service %.1f us (host %.1f, "
                "GPU %.1f); stream: uploads %.1f us, experts %.1f us, gather %.1f us; failure: '%s'\n",
                requests, jobs, columns, 1e6 * s / requests, 1e-3 * st.busy_ns / st.requests,
                1e-3 * st.enqueue_ns / st.requests, 1e-3 * st.wait_ns / st.requests, 1e3 * st.upload_ms / st.requests,
                1e3 * st.experts_ms / st.requests, 1e3 * st.gather_ms / st.requests, service->failure().c_str());
    return 0;
}
