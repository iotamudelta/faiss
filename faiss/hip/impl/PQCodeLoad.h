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
            code32[0] = p[0]; //TODO no idea if this is right
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
                    code32[0] = p[0]; //TODO no idea if this is right
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
        unsigned int a;
        unsigned int b;
        unsigned int c;

        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 0];" : "=r"(a) : "l"(p));
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 1];" : "=r"(b) : "l"(p));
//DONE        asm("ld.global.cs.u8 {%0}, [%1 + 2];" : "=r"(c) : "l"(p));
        a = p[0];
        b = p[1];
        c = p[2];
        // FIXME: this is also slow, since we have to recover the
        // individual bytes loaded
        code32[0] = (c << 16) | (b << 8) | a;
    }
};

template <>
struct LoadCode32<4> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 4;
            code32[0] = p[0];
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
//DONE        asm("ld.global.cs.v2.u32 {%0, %1}, [%2];"
//DONE            : "=r"(code32[0]), "=r"(code32[1])
//DONE            : "l"(p));
        code32[0] = p[0];
        code32[1] = p[0];
    }
};

template <>
struct LoadCode32<12> {
    static inline __device__ void load(
            unsigned int code32[3],
            uint8_t* p,
            int offset) {
        p += offset * 12;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
    }
};

template <>
struct LoadCode32<16> {
    static inline __device__ void load(
            unsigned int code32[4],
            uint8_t* p,
            int offset) {
        p += offset * 16;
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
        code32[3] = p[0];
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
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 12];" : "=r"(code32[3]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 16];" : "=r"(code32[4]) : "l"(p));
        code32[0] = p[0];
        code32[1] = p[1];
        code32[2] = p[2];
        code32[3] = p[3];
        code32[4] = p[4];
    }
};

template <>
struct LoadCode32<24> {
    static inline __device__ void load(
            unsigned int code32[6],
            uint8_t* p,
            int offset) {
        p += offset * 24;
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
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[1];
        code32[3] = p[1];
        code32[4] = p[2];
        code32[5] = p[2];
    }
};

template <>
struct LoadCode32<28> {
    static inline __device__ void load(
            unsigned int code32[7],
            uint8_t* p,
            int offset) {
        p += offset * 28;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 0];" : "=r"(code32[0]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 4];" : "=r"(code32[1]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 8];" : "=r"(code32[2]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 12];" : "=r"(code32[3]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 16];" : "=r"(code32[4]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 20];" : "=r"(code32[5]) : "l"(p));
//DONE        asm(LD_NC_V1 " {%0}, [%1 + 24];" : "=r"(code32[6]) : "l"(p));
        code32[0] = p[0];
        code32[1] = p[1];
        code32[2] = p[2];
        code32[3] = p[3];
        code32[4] = p[4];
        code32[5] = p[5];
        code32[6] = p[6];
    }
};

template <>
struct LoadCode32<32> {
    static inline __device__ void load(
            unsigned int code32[8],
            uint8_t* p,
            int offset) {
        p += offset * 32;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4];"
//DONE            : "=r"(code32[0]), "=r"(code32[1]), "=r"(code32[2]), "=r"(code32[3])
//DONE            : "l"(p));
//DONE        asm(LD_NC_V4 " {%0, %1, %2, %3}, [%4 + 16];"
//DONE            : "=r"(code32[4]), "=r"(code32[5]), "=r"(code32[6]), "=r"(code32[7])
//DONE            : "l"(p));
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
        code32[3] = p[0];

        code32[4] = p[1];
        code32[5] = p[1];
        code32[6] = p[1];
        code32[7] = p[1];
    }
};

template <>
struct LoadCode32<40> {
    static inline __device__ void load(
            unsigned int code32[10],
            uint8_t* p,
            int offset) {
        p += offset * 40;
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
        code32[0] = p[0];
        code32[1] = p[0];

        code32[2] = p[1];
        code32[3] = p[1];

        code32[4] = p[2];
        code32[5] = p[2];

        code32[6] = p[3];
        code32[7] = p[3];

        code32[8] = p[4];
        code32[9] = p[4];
    }
};

template <>
struct LoadCode32<48> {
    static inline __device__ void load(
            unsigned int code32[12],
            uint8_t* p,
            int offset) {
        p += offset * 48;
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
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
        code32[3] = p[0];

        code32[4] = p[1];
        code32[5] = p[1];
        code32[6] = p[1];
        code32[7] = p[1];

        code32[8] = p[2];
        code32[9] = p[2];
        code32[10] = p[2];
        code32[11] = p[2];
    }
};

template <>
struct LoadCode32<56> {
    static inline __device__ void load(
            unsigned int code32[14],
            uint8_t* p,
            int offset) {
        p += offset * 56;
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
        code32[0] = p[0];
        code32[1] = p[0];

        code32[2] = p[1];
        code32[3] = p[1];

        code32[4] = p[2];
        code32[5] = p[2];

        code32[6] = p[3];
        code32[7] = p[3];

        code32[8] = p[4];
        code32[9] = p[4];

        code32[10] = p[5];
        code32[11] = p[5];

        code32[12] = p[6];
        code32[13] = p[6];
    }
};

template <>
struct LoadCode32<64> {
    static inline __device__ void load(
            unsigned int code32[16],
            uint8_t* p,
            int offset) {
        p += offset * 64;
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
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
        code32[3] = p[0];

        code32[4] = p[1];
        code32[5] = p[1];
        code32[6] = p[1];
        code32[7] = p[1];

        code32[8] = p[2];
        code32[9] = p[2];
        code32[10] = p[2];
        code32[11] = p[2];

        code32[12] = p[3];
        code32[13] = p[3];
        code32[14] = p[3];
        code32[15] = p[3];
    }
};

template <>
struct LoadCode32<96> {
    static inline __device__ void load(
            unsigned int code32[24],
            uint8_t* p,
            int offset) {
        p += offset * 96;
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
        code32[0] = p[0];
        code32[1] = p[0];
        code32[2] = p[0];
        code32[3] = p[0];

        code32[4] = p[1];
        code32[5] = p[1];
        code32[6] = p[1];
        code32[7] = p[1];

        code32[8] = p[2];
        code32[9] = p[2];
        code32[10] = p[2];
        code32[11] = p[2];

        code32[12] = p[3];
        code32[13] = p[3];
        code32[14] = p[3];
        code32[15] = p[3];

        code32[16] = p[4];
        code32[17] = p[4];
        code32[18] = p[4];
        code32[19] = p[4];

        code32[20] = p[5];
        code32[21] = p[5];
        code32[22] = p[5];
        code32[23] = p[5];
    }
};

} // namespace hip
} // namespace faiss
