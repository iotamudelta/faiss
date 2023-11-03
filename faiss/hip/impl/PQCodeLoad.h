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
        using T = uint8_t __attribute__((ext_vector_type(1)));
        T* t = reinterpret_cast<T*>(p);
        T* u = reinterpret_cast<T*>(code32);
        u[0] = __builtin_nontemporal_load(t);
//DONE        asm("ld.global.cs.u8 {%0}, [%1];" : "=r"(code32[0]) : "l"(p));
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
//DONE        asm("ld.global.cs.u16 {%0}, [%1];" : "=r"(code32[0]) : "l"(p));
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
        u[0] = __builtin_nontemporal_load(t);

        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 0];" : "=r"(a) : "l"(p));
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 1];" : "=r"(b) : "l"(p));
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 2];" : "=r"(c) : "l"(p));
        // FIXME: this is also slow, since we have to recover the
        // individual bytes loaded
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
	//DONE        asm("ld.global.cs.u32 {%0}, [%1];" : "=r"(code32[0]) : "l"(p));
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
	//DONE        asm("ld.global.cs.v2.u32 {%0, %1}, [%2];"
//DONE            : "=r"(code32[0]), "=r"(code32[1])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
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
//DONE        asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
    }
};

template <>
struct LoadCode32<20> {
    static inline __device__ void load(
            unsigned int code32[5],
            uint8_t* p,
            int offset) {
        p += offset * 20;
        //using T = uint32_t __attribute__((ext_vector_type(1)));
        code32[0] = __builtin_nontemporal_load(p);
	code32[1] = __builtin_nontemporal_load(p + 4);
        code32[2] = __builtin_nontemporal_load(p + 8);
	code32[3] = __builtin_nontemporal_load(p + 12);
	code32[4] = __builtin_nontemporal_load(p + 16);

	// FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 12];" : "=r"(code32[3]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 16];" : "=r"(code32[4]) : "l"(p));
        //code32[0] = p[0];
        //code32[1] = p[1];
        //code32[2] = p[2];
        //code32[3] = p[3];
        //code32[4] = p[4];
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
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 0];"
//DONE            : "=r"(code32[0]), "=r"(code32[1])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 8];"
//DONE            : "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 12];" : "=r"(code32[3]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 16];" : "=r"(code32[4]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 20];" : "=r"(code32[5]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 24];" : "=r"(code32[6]) : "l"(p));
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
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5]), "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 0];"
//DONE            : "=r"(code32[0]), "=r"(code32[1])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 8];"
//DONE            : "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 24];"
//DONE            : "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 32];"
//DONE            : "=r"(code32[8]), "=r"(code32[9])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5]), "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];"
//DONE            : "=r"(code32[8]),
//DONE              "=r"(code32[9]),
//DONE              "=r"(code32[10]),
//DONE              "=r"(code32[11])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 0];"
//DONE            : "=r"(code32[0]), "=r"(code32[1])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 8];"
//DONE            : "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 24];"
//DONE            : "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 32];"
//DONE            : "=r"(code32[8]), "=r"(code32[9])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 40];"
//DONE            : "=r"(code32[10]), "=r"(code32[11])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V2 " {%0, %1}, [%2 + 48];"
//DONE            : "=r"(code32[12]), "=r"(code32[13])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5]), "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];"
//DONE            : "=r"(code32[8]),
//DONE              "=r"(code32[9]),
//DONE              "=r"(code32[10]),
//DONE              "=r"(code32[11])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 48];"
//DONE            : "=r"(code32[12]),
//DONE              "=r"(code32[13]),
//DONE              "=r"(code32[14]),
//DONE              "=r"(code32[15])
//DONE            : "l"(p));
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
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5]), "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 32];"
//DONE            : "=r"(code32[8]),
//DONE              "=r"(code32[9]),
//DONE              "=r"(code32[10]),
//DONE              "=r"(code32[11])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 48];"
//DONE            : "=r"(code32[12]),
//DONE              "=r"(code32[13]),
//DONE              "=r"(code32[14]),
//DONE              "=r"(code32[15])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 64];"
//DONE            : "=r"(code32[16]),
//DONE              "=r"(code32[17]),
//DONE              "=r"(code32[18]),
//DONE              "=r"(code32[19])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 80];"
//DONE            : "=r"(code32[20]),
//DONE              "=r"(code32[21]),
//DONE              "=r"(code32[22]),
//DONE              "=r"(code32[23])
//DONE            : "l"(p));
    }
};

} // namespace hip
} // namespace faiss
