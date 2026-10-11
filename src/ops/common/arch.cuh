#pragma once

// Device architecture gates of the V100 + RTX 4090 fork. Upstream compiles only for sm_120a; the
// RTX 4090 (sm_89) has no TMA, no mbarrier transactions, no programmatic dependent launch and no
// block-scaled or FP4 tensor-core MMA. Routes built on those keep compiling for sm_89 with a trap
// in place of the instruction, and their dispatchers never select them there (core/arch.h).
//
// In the host pass (__CUDA_ARCH__ undefined) the gates are open, so declarations and templates see
// the upstream code unchanged.

#if defined(__CUDA_ARCH__)
#define INFERNIX_DEVICE_SM90 (__CUDA_ARCH__ >= 900)
#define INFERNIX_DEVICE_SM120 (__CUDA_ARCH__ >= 1200)
#else
#define INFERNIX_DEVICE_SM90 1
#define INFERNIX_DEVICE_SM120 1
#endif
