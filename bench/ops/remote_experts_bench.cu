// The second GPU's expert service (ops/offloaded_sparse_moe/cuda/remote_experts.h) on one GPU,
// driven the way the main GPU drives it: synthetic requests written to the mapped request, the
// answer awaited on the done word. Prints the service's per-request times; run it under ncu to
// profile the remote GPU's kernels (a process with both GPUs cannot be profiled on the V100).
//
//   infernix_remote_experts_bench [requests=2000] [columns=5] [jobs=7] [max-entries-per-job=2]

#include "ops/offloaded_sparse_moe/cuda/remote_experts.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace infernix::ops;
using namespace infernix::ops::offloaded_moe;

int main(int argc, char** argv) {
    const int requests = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int columns  = argc > 2 ? std::atoi(argv[2]) : 5;
    const int jobs     = argc > 3 ? std::atoi(argv[3]) : 7;
    const int per_job  = argc > 4 ? std::atoi(argv[4]) : 2;
    constexpr std::uint32_t kLayers = 2, kExperts = 512;
    constexpr int kTopK = 10;
    const std::uint64_t stride = kRecordBytes;

    std::vector<ExpertScales> scales(kExperts);
    for (auto& s : scales) { s = {0.02F, 0.02F, 0.05F, 0.001F, 0.001F, 0.001F}; }
    RemoteExpertService::Options o;
    o.device = 0;
    o.main_device = 0;
    o.layers = kLayers;
    o.experts = kExperts;
    o.record_stride = stride;
    o.scales = {scales.data(), scales.data()};
    o.max_records = 64;
    RemoteExpertService service(o);
    std::fprintf(stderr, "service: capacity %u\n", service.capacity());

    std::mt19937 rng(7);
    std::vector<std::uint32_t> keys;
    for (std::uint32_t k = 0; k < std::min<std::uint32_t>(service.capacity(), kLayers * kExperts); ++k) { keys.push_back(k); }
    service.load(keys, [&](std::uint32_t, std::uint8_t* d) {
        std::uint64_t z = rng();
        for (std::uint64_t i = 0; i + 8 <= stride; i += 8) {
            z = z * 6364136223846793005ULL + 1442695040888963407ULL;
            std::memcpy(d + i, &z, 8);
        }
        // E4M3 block scales of 1.0 (0x38) in every unit's scale bytes: bytes 128..143 of each 144-byte unit.
        for (std::uint64_t u = 0; u + 144 <= stride; u += 144) {
            for (int b = 128; b < 144; ++b) { d[u + b] = 0x38; }
        }
    });
    std::fprintf(stderr, "loaded %u\n", service.held());
    const MoeRemoteChannel ch = service.channel(0);
    auto* x = ch.x;
    for (int i = 0; i < 2560 * columns; ++i) {
        const float v = std::uniform_real_distribution<float>(-2.0F, 2.0F)(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        x[i] = static_cast<std::uint16_t>(bits >> 16);
    }
    auto* r = ch.request;
    auto* done = reinterpret_cast<volatile std::uint32_t*>(const_cast<std::uint32_t*>(ch.done));
    std::uint32_t sequence = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int q = 0; q < requests; ++q) {
        int at = 0;
        for (int j = 0; j < jobs; ++j) {
            r->expert[j] = (j * 9 + q) % 64;
            const int n  = 1 + (j % per_job);
            r->first[j]  = at;
            r->count[j]  = n;
            for (int c = 0; c < n; ++c) { r->entry[at++] = ((c + j) % columns) * kTopK + (j % kTopK); }
        }
        // distinct experts per request
        
        r->layer = 0;
        r->columns = columns;
        r->top_k = kTopK;
        r->jobs = jobs;
        r->entries = at;
        sequence = sequence + 1 == 0 ? 1 : sequence + 1;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile std::uint32_t*>(&r->sequence)[0] = sequence;
        while (*done != sequence) {}
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto st = service.stats();
    std::printf("%d requests of %d jobs, %d columns: %.1f us per request round trip; service %.1f us (host %.1f, "
                "GPU %.1f); stream: uploads %.1f us, experts %.1f us, gather %.1f us; failure: %s\n",
                requests, jobs, columns, 1e6 * s / requests, 1e-3 * st.busy_ns / st.requests,
                1e-3 * st.enqueue_ns / st.requests, 1e-3 * st.wait_ns / st.requests, 1e3 * st.upload_ms / st.requests,
                1e3 * st.experts_ms / st.requests, 1e3 * st.gather_ms / st.requests, service.failure().c_str());
    return 0;
}
