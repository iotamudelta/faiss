/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hipblas.h>
#include <faiss/hip/utils/DeviceTensor.h>
#include <faiss/hip/utils/Float16.h>
#include <faiss/hip/utils/HostTensor.h>
#include <faiss/hip/utils/Tensor.h>

namespace faiss {
namespace gpu {

class GpuResources;

/// C = alpha * A * B + beta * C
/// Expects row major layout, not fortran/blas column major!
template <typename AT, typename BT>
void runMatrixMult(
        Tensor<float, 2, true>& c,
        bool transC,
        Tensor<AT, 2, true>& a,
        bool transA,
        Tensor<BT, 2, true>& b,
        bool transB,
        float alpha,
        float beta,
        hipblasHandle_t handle,
        hipStream_t stream);

/// C_i = alpha * A_i * B_i + beta * C_i
/// where `i` is the outermost dimension, via iterated gemm
/// Expects row major layout, not fortran/blas column major!
template <typename AT, typename BT>
void runIteratedMatrixMult(
        Tensor<float, 3, true>& c,
        bool transC,
        Tensor<AT, 3, true>& a,
        bool transA,
        Tensor<BT, 3, true>& b,
        bool transB,
        float alpha,
        float beta,
        hipblasHandle_t handle,
        hipStream_t stream);

} // namespace gpu
} // namespace faiss

#include <faiss/gpu/utils/MatrixMult-inl.cuh>