/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/hip/utils/PtxUtils.h>

namespace faiss {
namespace hip {

///
/// This file contains loader functions for PQ codes of various byte
/// length.
///

// Type-specific wrappers around the PTX bfe.* instruction, for
// quantization code extraction
inline __device__ unsigned int getByte(unsigned char v, int pos, int width) {
    return v;
}

inline __device__ unsigned int getByte(unsigned short v, int pos, int width) {
    return getBitfield((unsigned int)v, pos, width);
}

inline __device__ unsigned int getByte(unsigned int v, int pos, int width) {
    return getBitfield(v, pos, width);
}

inline __device__ unsigned int getByte(uint64_t v, int pos, int width) {
    return getBitfield(v, pos, width);
}

template <int NumSubQuantizers>
struct LoadCode32 {};

template <>
struct LoadCode32<1> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 1;
        //using T = uint8_t __attribute__((ext_vector_type(1)));
        //T* t = reinterpret_cast<T*>(p);
        uint8_t* u = reinterpret_cast<uint8_t*>(code32);
        u[0] = __builtin_nontemporal_load(p);
    }
};

template <>
struct LoadCode32<2> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 2;
        using T = uint8_t __attribute__((ext_vector_type(2)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<3> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 3;
	using T = uint8_t __attribute__((ext_vector_type(3)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[1] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<4> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 4;
        using T = uint32_t __attribute__((ext_vector_type(1)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<8> {
    static inline __device__ void load(
            unsigned int code32[2],
            uint8_t* p,
            int offset) {
        p += offset * 8;
        using T = uint32_t __attribute__((ext_vector_type(2)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<12> {
    static inline __device__ void load(
            unsigned int code32[3],
            uint8_t* p,
            int offset) {
        p += offset * 12;
	using T = uint32_t __attribute__((ext_vector_type(3)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<16> {
    static inline __device__ void load(
            unsigned int code32[4],
            uint8_t* p,
            int offset) {
        p += offset * 16;
	using T = uint32_t __attribute__((ext_vector_type(4)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<20> {
    static inline __device__ void load(
            unsigned int code32[5],
            uint8_t* p,
            int offset) {
        p += offset * 20;
	using T = uint32_t __attribute__((ext_vector_type(5)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<24> {
    static inline __device__ void load(
            unsigned int code32[6],
            uint8_t* p,
            int offset) {
        p += offset * 24;
	using T = uint32_t __attribute__((ext_vector_type(6)));
	T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
	u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<28> {
    static inline __device__ void load(
            unsigned int code32[7],
            uint8_t* p,
            int offset) {
        p += offset * 28;
	using T = uint32_t __attribute__((ext_vector_type(7)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<32> {
    static inline __device__ void load(
            unsigned int code32[8],
            uint8_t* p,
            int offset) {
        p += offset * 32;
	using T = uint32_t __attribute__((ext_vector_type(8)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<40> {
    static inline __device__ void load(
            unsigned int code32[10],
            uint8_t* p,
            int offset) {
        p += offset * 40;
	using T = uint32_t __attribute__((ext_vector_type(10)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<48> {
    static inline __device__ void load(
            unsigned int code32[12],
            uint8_t* p,
            int offset) {
        p += offset * 48;
	using T = uint32_t __attribute__((ext_vector_type(12)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<56> {
    static inline __device__ void load(
            unsigned int code32[14],
            uint8_t* p,
            int offset) {
        p += offset * 56;
	using T = uint32_t __attribute__((ext_vector_type(14)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<64> {
    static inline __device__ void load(
            unsigned int code32[16],
            uint8_t* p,
            int offset) {
        p += offset * 64;
	using T = uint32_t __attribute__((ext_vector_type(16)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

template <>
struct LoadCode32<96> {
    static inline __device__ void load(
            unsigned int code32[24],
            uint8_t* p,
            int offset) {
        p += offset * 96;
	using T = uint32_t __attribute__((ext_vector_type(24)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
    }
};

} // namespace hip
} // namespace faiss
