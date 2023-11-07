/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace faiss {
namespace hip {

// We validate this against the actual architecture in device initialization
#ifdef HIP_WF32
constexpr int kWarpSize = 32; // either = 32 or = 64 (Defined in hip_runtime.h)
#else
constexpr int kWarpSize = 64;
#endif

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {
    // For the time being, assume synchronicity.
    __threadfence_block();
    //__syncthreads();
}

#if CUDA_VERSION > 9000
#warning("CUDA > 9000, somehow")
// Based on the CUDA version (we assume what version of nvcc/ptxas we were
// compiled with), the register allocation algorithm is much better, so only
// enable the 2048 selection code if we are above 9.0 (9.2 seems to be ok)
#define GPU_MAX_SELECTION_K 2048
#else
#warning("CUDA small")
#define GPU_MAX_SELECTION_K 1024
#endif

} // namespace hip
} // namespace faiss
