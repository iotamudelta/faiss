/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/hip/utils/Float16.h>

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss {
namespace hip {

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
        Half4* t = reinterpret_cast<Half4*>(p);
        out = *t;
        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        Half4* t = reinterpret_cast<Half4*>(p);
        *t = v;
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
	Half8* t = reinterpret_cast<Half8*>(p);
        out = *t;
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
        Half8* t = reinterpret_cast<Half8*>(p);
        *t = v;
    }
};

} // namespace hip
} // namespace faiss
