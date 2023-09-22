# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(FAISS_HIP_SRC
  GpuAutoTune.cpp
  GpuCloner.cpp
  GpuDistance.hip
  GpuIcmEncoder.hip
  GpuIndex.hip
  GpuIndexBinaryFlat.hip
  GpuIndexFlat.hip
  GpuIndexIVF.hip
  GpuIndexIVFFlat.hip
  GpuIndexIVFPQ.hip
  GpuIndexIVFScalarQuantizer.hip
  GpuResources.cpp
  StandardGpuResources.cpp
  impl/BinaryDistance.hip
  impl/BinaryFlatIndex.hip
  impl/BroadcastSum.hip
  impl/Distance.hip
  impl/FlatIndex.hip
  impl/IndexUtils.hip
  impl/IVFAppend.hip
  impl/IVFBase.hip
  impl/IVFFlat.hip
  impl/IVFFlatScan.hip
  impl/IVFInterleaved.hip
  impl/IVFPQ.hip
  impl/IVFUtils.hip
  impl/IVFUtilsSelect1.hip
  impl/IVFUtilsSelect2.hip
  impl/InterleavedCodes.cpp
  impl/L2Norm.hip
  impl/L2Select.hip
  impl/PQScanMultiPassPrecomputed.hip
  impl/RemapIndices.cpp
  impl/VectorResidual.hip
  impl/scan/IVFInterleaved1.hip
  impl/scan/IVFInterleaved32.hip
  impl/scan/IVFInterleaved64.hip
  impl/scan/IVFInterleaved128.hip
  impl/scan/IVFInterleaved256.hip
  impl/scan/IVFInterleaved512.hip
  impl/scan/IVFInterleaved1024.hip
  impl/scan/IVFInterleaved2048.hip
  impl/IcmEncoder.hip
  utils/BlockSelectFloat.hip
  utils/DeviceUtils.hip
  utils/StackDeviceMemory.cpp
  utils/Timer.cpp
  utils/WarpSelectFloat.hip
  utils/blockselect/BlockSelectFloat1.hip
  utils/blockselect/BlockSelectFloat32.hip
  utils/blockselect/BlockSelectFloat64.hip
  utils/blockselect/BlockSelectFloat128.hip
  utils/blockselect/BlockSelectFloat256.hip
  utils/blockselect/BlockSelectFloatF512.hip
  utils/blockselect/BlockSelectFloatF1024.hip
  utils/blockselect/BlockSelectFloatF2048.hip
  utils/blockselect/BlockSelectFloatT512.hip
  utils/blockselect/BlockSelectFloatT1024.hip
  utils/blockselect/BlockSelectFloatT2048.hip
  utils/warpselect/WarpSelectFloat1.hip
  utils/warpselect/WarpSelectFloat32.hip
  utils/warpselect/WarpSelectFloat64.hip
  utils/warpselect/WarpSelectFloat128.hip
  utils/warpselect/WarpSelectFloat256.hip
  utils/warpselect/WarpSelectFloatF512.hip
  utils/warpselect/WarpSelectFloatF1024.hip
  utils/warpselect/WarpSelectFloatF2048.hip
  utils/warpselect/WarpSelectFloatT512.hip
  utils/warpselect/WarpSelectFloatT1024.hip
  utils/warpselect/WarpSelectFloatT2048.hip
)

set(FAISS_HIP_HEADERS
  GpuAutoTune.h
  GpuCloner.h
  GpuClonerOptions.h
  GpuDistance.h
  GpuIcmEncoder.h
  GpuFaissAssert.h
  GpuIndex.h
  GpuIndexBinaryFlat.h
  GpuIndexFlat.h
  GpuIndexIVF.h
  GpuIndexIVFFlat.h
  GpuIndexIVFPQ.h
  GpuIndexIVFScalarQuantizer.h
  GpuIndicesOptions.h
  GpuResources.h
  StandardGpuResources.h
  impl/BinaryDistance.h
  impl/BinaryFlatIndex.h
  impl/BroadcastSum.h
  impl/Distance.h
  impl/DistanceUtils.h
  impl/FlatIndex.h
  impl/GeneralDistance.h
  impl/GpuScalarQuantizer.h
  impl/IndexUtils.h
  impl/IVFAppend.h
  impl/IVFBase.h
  impl/IVFFlat.h
  impl/IVFFlatScan.h
  impl/IVFInterleaved.h
  impl/IVFPQ.h
  impl/IVFUtils.h
  impl/InterleavedCodes.h
  impl/L2Norm.h
  impl/L2Select.h
  impl/PQCodeDistances-inl.h
  impl/PQCodeDistances.h
  impl/PQCodeLoad.h
  impl/PQScanMultiPassNoPrecomputed-inl.h
  impl/PQScanMultiPassNoPrecomputed.h
  impl/PQScanMultiPassPrecomputed.h
  impl/RemapIndices.h
  impl/VectorResidual.h
  impl/scan/IVFInterleavedImpl.h
  impl/IcmEncoder.h
  utils/BlockSelectKernel.h
  utils/Comparators.h
  utils/ConversionOperators.h
  utils/CopyUtils.h
  utils/DeviceDefs.h
  utils/DeviceTensor-inl.h
  utils/DeviceTensor.h
  utils/DeviceUtils.h
  utils/DeviceVector.h
  utils/Float16.h
  utils/HostTensor-inl.h
  utils/HostTensor.h
  utils/Limits.h
  utils/LoadStoreOperators.h
  utils/MathOperators.h
  utils/MatrixMult-inl.h
  utils/MatrixMult.h
  utils/MergeNetworkBlock.h
  utils/MergeNetworkUtils.h
  utils/MergeNetworkWarp.h
  utils/NoTypeTensor.h
  utils/Pair.h
  utils/PtxUtils.h
  utils/ReductionOperators.h
  utils/Reductions.h
  utils/Select.h
  utils/StackDeviceMemory.h
  utils/StaticUtils.h
  utils/Tensor-inl.h
  utils/Tensor.h
  utils/ThrustUtils.h
  utils/Timer.h
  utils/Transpose.h
  utils/WarpPackedBits.h
  utils/WarpSelectKernel.h
  utils/WarpShuffles.h
  utils/blockselect/BlockSelectImpl.h
  utils/warpselect/WarpSelectImpl.h
)

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_HIP_HEADERS ${FAISS_HIP_HEADERS} PARENT_SCOPE)

target_sources(faiss PRIVATE ${FAISS_HIP_SRC})
target_sources(faiss_avx2 PRIVATE ${FAISS_HIP_SRC})

foreach(header ${FAISS_HIP_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/hip/${dir}
  )
endforeach()

find_package(hip REQUIRED)
find_package(hsa-runtime64 REQUIRED)
find_package(hipblas REQUIRED)
target_link_libraries(faiss PRIVATE hip::amdhip64 hipblas)
target_link_libraries(faiss_avx2 PRIVATE hip::amdhip64 hipblas)
target_compile_options(faiss PRIVATE)
target_compile_options(faiss_avx2 PRIVATE)
