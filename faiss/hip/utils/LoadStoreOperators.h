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
        half2* t = reinterpret_cast<half2*>(p);
        out.a = t[0];
	out.b = t[1];
    //DONE        asm("ld.global.v2.u32 {%0, %1}, [%2];"
    //DONE            : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
    //DONE            : "l"(p));

    //DONE        asm("ld.global.v2.u32 {%0, %1}, [%2];"
    //DONE            : "=r"(out.a.x), "=r"(out.b.x)
    //DONE             : "l"(p));

        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        half2* t = reinterpret_cast<half2*>(p);
        v.a = t[0];
        v.b = t[1];
//#if CUDA_VERSION >= 9000
//DONE        asm("st.v2.u32 [%0], {%1, %2};"
//DONE            :
//DONE            : "l"(p), "r"(__HALF2_TO_UI(v.a)), "r"(__HALF2_TO_UI(v.b)));
//#else
//DONE        asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(v.a.x), "r"(v.b.x));
//#endif
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
	half2* t = reinterpret_cast<half2*>(p);
        out.a.a = t[0];
        out.a.b = t[1];
        out.b.a = t[2];
	out.b.b = t[3];
//#if CUDA_VERSION >= 9000
//DONE        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(__HALF2_TO_UI(out.a.a)),
//DONE              "=r"(__HALF2_TO_UI(out.a.b)),
//DONE              "=r"(__HALF2_TO_UI(out.b.a)),
//DONE              "=r"(__HALF2_TO_UI(out.b.b))
//DONE            : "l"(p));
//#else
//DONE        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(out.a.a.x), "=r"(out.a.b.x), "=r"(out.b.a.x), "=r"(out.b.b.x)
//DONE            : "l"(p));
//#endif
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
        half2* t = reinterpret_cast<half2*>(p);
        v.a.a = t[0];
        v.a.b = t[1];
        v.b.a = t[2];
        v.b.b = t[3];
//#if CUDA_VERSION >= 9000
//DONE        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
//DONE            :
//DONE            : "l"(p),
//DONE              "r"(__HALF2_TO_UI(v.a.a)),
//DONE              "r"(__HALF2_TO_UI(v.a.b)),
//DONE              "r"(__HALF2_TO_UI(v.b.a)),
//DONE              "r"(__HALF2_TO_UI(v.b.b)));
//#else
//DONE        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
//DONE            :
//DONE            : "l"(p), "r"(v.a.a.x), "r"(v.a.b.x), "r"(v.b.a.x), "r"(v.b.b.x));
//#endif
    }
};

} // namespace hip
} // namespace faiss
