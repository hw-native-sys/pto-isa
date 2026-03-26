/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_HPP_KIRIN9030
#define TINSERT_HPP_KIRIN9030
#include "common.hpp"
#define bfloat16_t half
#define float8_e4m3_t int8_t
#define float8_e5m2_t int8_t
#define hifloat8_t int8_t
#define float8_e8m0_t int8_t
#define COPY_CC_TO_CUBF(dst, src, nSize, srcRow, dstStride, srcStride, QuantPre, reluMode, channelSplitEnable) \
    copy_matrix_cc_to_cbuf(dst, src, 0, nSize, srcRow, dstStride, srcStride, 0, 0, QuantPre, reluMode,         \
                           channelSplitEnable, false, 0, 0, false, false, 0, false, false, false, false, false, false)
#include "pto/npu/a5/TInsert.hpp"
#undef bfloat16_t
#undef float8_e4m3_t
#undef float8_e5m2_t
#undef hifloat8_t
#undef float8_e8m0_t
#endif