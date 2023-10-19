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

__device__ __forceinline__ unsigned int
BFE32(unsigned int val,
    unsigned int pos,
    unsigned int len)
{
    //static_assert(std::is_unsigned<UnsignedBits>::value, "UnsignedBits must be unsigned");
    return __bitextract_u32(val, pos, len);
    //return (static_cast<unsigned int>(source) << (32 - bit_start - num_bits)) >> (32 - num_bits);
}

__device__ __forceinline__ uint64_t
BFE64(uint64_t val,
    unsigned int pos,
    unsigned int len)
{
    //static_assert(std::is_unsigned<UnsignedBits>::value, "UnsignedBits must be unsigned");
    return __bitextract_u64(val, pos, len);
    //return (source << (64 - bit_start - num_bits)) >> (64 - num_bits);
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos,int len) {
    return BFE32(val, pos, len);
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
    return BFE64(val, pos, len);
}

__device__ __forceinline__ int getLaneId() {
    return threadIdx.x & 63;
}

} // namespace hip
} // namespace faiss
