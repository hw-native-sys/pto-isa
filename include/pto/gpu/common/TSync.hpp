/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TSYNC_HPP
#define PTO_GPU_COMMON_TSYNC_HPP

#include <pto/common/type.hpp>
#include <pto/common/event.hpp>

namespace pto {

template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
    (void)sizeof(OpCode);
    // v1 GPU backend keeps advanced PTO event / pipeline semantics deferred.
}

} // namespace pto

#endif
