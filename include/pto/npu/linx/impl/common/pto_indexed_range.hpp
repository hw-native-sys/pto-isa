/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE_PTO_INDEXED_RANGE_HPP
#define PTO_LINX_NATIVE_PTO_INDEXED_RANGE_HPP

#include <cstdint>

namespace pto {
// PTO-ISA PR #424 at 1981bceffa82. Bounds are inclusive signed byte
// displacements of access starts, relative to the operation's GM base.
// Software certifies coverage and a nonwrapping address envelope. This is
// not an element count, row stride, or runtime bounds check.
struct IndexedByteRange {
  int64_t min;
  int64_t max;
};
} // namespace pto

#endif
