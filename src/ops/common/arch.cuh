#pragma once

// Device architecture gates of the V100 + RTX 4090 fork. Upstream compiles only for sm_120a; the
// RTX 4090 (sm_89) has no TMA, no mbarrier transactions, no programmatic dependent launch and no
// block-scaled or FP4 tensor-core MMA. Routes built on those keep compiling for sm_89 with a trap
// in place of the instruction, and their dispatchers never select them there (core/arch.h).
//
// In the host pass (__CUDA_ARCH__ undefined) the gates are open, so declarations and templates see
// the upstream code unchanged.

#if defined(__CUDA_ARCH__)
#define INFERNIX_DEVICE_SM80 (__CUDA_ARCH__ >= 800)
#define INFERNIX_DEVICE_SM90 (__CUDA_ARCH__ >= 900)
#define INFERNIX_DEVICE_SM120 (__CUDA_ARCH__ >= 1200)
#else
#define INFERNIX_DEVICE_SM80 1
#define INFERNIX_DEVICE_SM90 1
#define INFERNIX_DEVICE_SM120 1
#endif

// Warp reductions of sm_80 (__reduce_*_sync) with a shuffle tree below it (the V100's expert service
// runs the expert kernels on sm_70). Integer max and sum are exact in either order.
__device__ __forceinline__ unsigned infernix_warp_max_u32(unsigned mask, unsigned v) {
#if INFERNIX_DEVICE_SM80
    return __reduce_max_sync(mask, v);
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned o = __shfl_xor_sync(mask, v, offset);
        v                = o > v ? o : v;
    }
    return v;
#endif
}

__device__ __forceinline__ int infernix_warp_max_i32(unsigned mask, int v) {
#if INFERNIX_DEVICE_SM80
    return __reduce_max_sync(mask, v);
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int o = __shfl_xor_sync(mask, v, offset);
        v           = o > v ? o : v;
    }
    return v;
#endif
}

__device__ __forceinline__ int infernix_warp_add_i32(unsigned mask, int v) {
#if INFERNIX_DEVICE_SM80
    return __reduce_add_sync(mask, v);
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(mask, v, offset); }
    return v;
#endif
}
