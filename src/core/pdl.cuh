#pragma once

#include "ops/common/arch.cuh"

#include <cuda_runtime.h>

#include <array>
#include <atomic>

#include <cstddef>
#include <utility>

namespace infernix::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Whether the current device runs programmatic dependent launch (compute capability 9.0+). The
// V100 + RTX 4090 fork runs on sm_70 and sm_89: there every launch is an ordinary stream-ordered
// launch and the device-side calls below are no-ops. Cached per device.
[[nodiscard]] inline bool device_supports_pdl() noexcept {
    static std::array<std::atomic<int>, 16> cache{}; // 0 unknown, 1 no, 2 yes
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) {
        (void)cudaGetLastError();
        return false;
    }
    int known = cache[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (known == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            (void)cudaGetLastError();
            major = 0;
        }
        known = major >= 9 ? 2 : 1;
        cache[static_cast<std::size_t>(device)].store(known, std::memory_order_relaxed);
    }
    return known == 2;
}

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = device_supports_pdl() ? 1 : 0;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Whether a launch on this stream becomes a programmatic dependent of the preceding kernel: only
// while the stream captures a CUDA Graph, so eager launches keep full stream serialization.
[[nodiscard]] inline bool graph_dependent(cudaStream_t stream) noexcept {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return status == cudaStreamCaptureStatusActive;
}

// Launches a kernel that implements the consumer protocol below (enter() before touching anything
// but its immutable weights). Captured launches become programmatic dependents; eager launches are
// ordinary stream-ordered launches.
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_consumer(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    if (graph_dependent(launch.stream)) {
        return launch_dependent(launch, kernel, std::forward<CallArgs>(args)...);
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// How a launch orders against the kernel before it on the stream, for launchers shared by call
// sites that differ: Programmatic is launch_consumer(), Serialized an ordinary launch. Whether a
// consumer pays depends on its neighbours - a dependent launched early packs onto the SMs its
// producer leaves free - so each call site picks the one that measured faster.
enum class Dependency : unsigned char { Serialized, Programmatic };

template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t launch_with(Dependency dependency, const LaunchConfig& launch,
                                             void (*kernel)(KernelArgs...), CallArgs&&... args) {
    if (dependency == Dependency::Programmatic) {
        return launch_consumer(launch, kernel, std::forward<CallArgs>(args)...);
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() {
#if INFERNIX_DEVICE_SM90
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if INFERNIX_DEVICE_SM90
    cudaGridDependencySynchronize();
#endif
}

// Consumer protocol for launch_consumer(). Call one of these in every thread, before any access
// other than reading immutable weights; a CTA that returns early must still have called it, so
// completion of this grid implies completion of everything it depends on.
//
// Number of SMs on this device.
__device__ __forceinline__ unsigned sm_count() {
    unsigned count;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(count));
    return count;
}

// enter(): for a short kernel. It waits until the preceding grid has completed and its writes are
// visible. A grid covering at most half the SMs first lets the next captured kernel begin
// launching once every CTA of this grid is resident. A wider grid does not: a dependent launched
// beside it gets only the SMs it leaves free, and a streaming consumer smaller than one wave (the
// NVFP4 down projection behind its 17408-column quantize, for example) packs onto those few SMs
// and runs up to half as fast. Such a grid lets dependents launch as its CTAs exit.
__device__ __forceinline__ void enter() {
    if (2U * gridDim.x * gridDim.y * gridDim.z <= sm_count()) { trigger_dependents(); }
    wait_for_dependencies();
}

// enter_streaming(): for a kernel that streams most of its bytes in a main loop. It only waits;
// the kernel calls trigger_dependents() once that loop is done, so dependents neither take its
// SMs nor compete for its bandwidth while it streams. Without that call, dependents launch when
// its CTAs exit.
__device__ __forceinline__ void enter_streaming() { wait_for_dependencies(); }

} // namespace infernix::pdl
