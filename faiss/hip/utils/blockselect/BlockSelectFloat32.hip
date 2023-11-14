/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/hip/utils/blockselect/BlockSelectImpl.h>

namespace faiss {
namespace hip {
#ifdef HIP_WF32
BLOCK_SELECT_IMPL(float, true, 32, 2);
BLOCK_SELECT_IMPL(float, false, 32, 2);
#endif
} // namespace hip
} // namespace faiss
