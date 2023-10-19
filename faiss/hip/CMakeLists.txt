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

function(generate_ivf_interleaved_code)
  set(SUB_CODEC_TYPE
    "faiss::hip::Codec<0, 1>"
    "faiss::hip::Codec<1, 1>"
    "faiss::hip::Codec<2, 1>"
    "faiss::hip::Codec<3, 1>"
    "faiss::hip::Codec<4, 1>"
    "faiss::hip::Codec<5, 1>"
    "faiss::hip::Codec<6, 1>"
    "faiss::hip::CodecFloat"
  )

  set(SUB_METRIC_TYPE
    "faiss::hip::IPDistance"
    "faiss::hip::L2Distance"
  )

  # Used for SUB_THREADS, SUB_NUM_WARP_Q, SUB_NUM_THREAD_Q
  set(THREADS_AND_WARPS
    "128|1024|8"
    "128|1|1"
    "128|128|3"
    "128|256|4"
    "128|64|2" #"128|32|2"     TODO won't compile with a warpsize of 64. Changed to 64
    "128|512|8"
    "128|64|3"
    "64|2048|8"
  )

  # Traverse through the Cartesian product of X and Y
  foreach(sub_codec ${SUB_CODEC_TYPE})
  foreach(metric_type ${SUB_METRIC_TYPE})
  foreach(threads_and_warps_str ${THREADS_AND_WARPS})
    string(REPLACE "|" ";" threads_and_warps ${threads_and_warps_str})
    list(GET threads_and_warps 0 sub_threads)
    list(GET threads_and_warps 1 sub_num_warp_q)
    list(GET threads_and_warps 2 sub_num_thread_q)

    # Define the output file name
    set(filename "template_${sub_codec}_${metric_type}_${sub_threads}_${sub_num_warp_q}_${sub_num_thread_q}")
    # Remove illegal characters from filename
    string(REGEX REPLACE "[^A-Za-z0-9_]" "" filename ${filename})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${filename}.hip")

    # Read the template file
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/impl/scan/IVFInterleavedScanKernelTemplate.hip" template_content)

    # Replace the placeholders
    string(REPLACE "SUB_CODEC_TYPE" "${sub_codec}" template_content "${template_content}")
    string(REPLACE "SUB_METRIC_TYPE" "${metric_type}" template_content "${template_content}")
    string(REPLACE "SUB_THREADS" "${sub_threads}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_WARP_Q" "${sub_num_warp_q}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_THREAD_Q" "${sub_num_thread_q}" template_content "${template_content}")

    # Write the modified content to the output file
    file(WRITE "${output_file}" "${template_content}")

    # Add the file to the sources
    list(APPEND FAISS_HIP_SRC "${output_file}")
  endforeach()
  endforeach()
  endforeach()
  # Propagate modified variable to the parent scope
  set(FAISS_HIP_SRC "${FAISS_HIP_SRC}" PARENT_SCOPE)
endfunction()

generate_ivf_interleaved_code()

if(FAISS_ENABLE_RAFT)
  list(APPEND FAISS_HIP_HEADERS
          impl/RaftFlatIndex.h)
  list(APPEND FAISS_HIP_SRC
          impl/RaftFlatIndex.hip)

  target_compile_definitions(faiss PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx2 PUBLIC USE_NVIDIA_RAFT=1)
endif()

# Export FAISS_HIP_HEADERS variable to parent scope.
set(FAISS_HIP_HEADERS ${FAISS_HIP_HEADERS} PARENT_SCOPE)

target_sources(faiss PRIVATE ${FAISS_HIP_SRC})
target_sources(faiss_avx2 PRIVATE ${FAISS_HIP_SRC})

foreach(header ${FAISS_HIP_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/hip/${dir}
  )
endforeach()

link_directories(/opt/rocm/lib/)
include_directories(/opt/rocm/include/)
find_package(hip REQUIRED)
find_package(hsa-runtime64 REQUIRED)
#find_package(hipblas REQUIRED)
target_link_libraries(faiss PRIVATE hip::amdhip64 ${HIPBLAS_LIBRARIES})
target_link_libraries(faiss_avx2 PRIVATE hip::amdhip64 ${HIPBLAS_LIBRARIES})
target_compile_options(faiss PRIVATE)
target_compile_options(faiss_avx2 PRIVATE)
