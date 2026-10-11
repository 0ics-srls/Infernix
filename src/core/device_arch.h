#pragma once

// Host-side capabilities of the current CUDA device, for the V100 + RTX 4090 fork: upstream assumes
// sm_120a everywhere, so launchers that pick a route needing TMA (sm_90+) or FP4 / block-scaled MMA
// (sm_120) ask here first and take a portable route on the RTX 4090 (sm_89) and the V100 (sm_70).
// The device-side twin is ops/common/arch.cuh.

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>

namespace infernix {

// Compute capability major of the current device (0 when it cannot be read). Cached per device.
[[nodiscard]] inline int current_device_compute_major() noexcept {
    static std::array<std::atomic<int>, 16> cache{}; // major + 1, 0 = unknown
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) {
        (void)cudaGetLastError();
        return 0;
    }
    int known = cache[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (known == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            (void)cudaGetLastError();
            major = 0;
        }
        known = major + 1;
        cache[static_cast<std::size_t>(device)].store(known, std::memory_order_relaxed);
    }
    return known - 1;
}

// TMA, mbarrier transactions, programmatic dependent launch: sm_90 and later.
[[nodiscard]] inline bool device_has_tma() noexcept { return current_device_compute_major() >= 9; }

// FP4 and block-scaled tensor-core MMA: sm_120 (the RTX 5090).
[[nodiscard]] inline bool device_has_fp4_mma() noexcept { return current_device_compute_major() >= 12; }

} // namespace infernix
