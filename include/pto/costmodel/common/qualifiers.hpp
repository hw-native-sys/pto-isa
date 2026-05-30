/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_COMMON_QUALIFIERS_HPP
#define PTO_MOCKER_COMMON_QUALIFIERS_HPP

#include <pto/common/cpu_stub.hpp>

using aclrtContext = void *;
using event_t = int;
using CceEventIdType = event_t;
using pad_t = int;
using addr_cal_mode_t = int;

namespace __cce_scalar {
using addr_cal_mode_t = ::addr_cal_mode_t;
}

// NOLINTNEXTLINE(bugprone-reserved-identifier): compatibility with legacy intrinsic qualifier tokens.
#ifndef __biasbuf__
// NOLINTNEXTLINE(bugprone-reserved-identifier): compatibility with legacy intrinsic qualifier tokens.
#define __biasbuf__
#endif

inline constexpr int ACL_STREAM_FAST_LAUNCH = 0;
inline constexpr int ACL_STREAM_FAST_SYNC = 0;
inline constexpr int ACL_STREAM_ATTR_FAILURE_MODE = 0;
inline constexpr int ACL_RT_CMO_TYPE_PREFETCH = 0;
inline constexpr int ONLY_INDEX = 0;
inline constexpr int VALUE_INDEX = 1;

using aclrtStreamAttrValue = int;

#ifdef aclFloat16ToFloat
#undef aclFloat16ToFloat
#endif
#define aclFloat16ToFloat(x) ((float)(x))
#ifdef __COSTMODEL
#ifdef dsb
#undef dsb
#endif
#ifdef SINGLE_CACHE_LINE
#undef SINGLE_CACHE_LINE
#endif
#ifdef DSB_DDR
#undef DSB_DDR
#endif
#ifdef DSB_ALL
#undef DSB_ALL
#endif
#ifdef DSB_UB
#undef DSB_UB
#endif
#ifdef set_flag
#undef set_flag
#endif
#ifdef wait_flag
#undef wait_flag
#endif
#ifdef set_mask_norm
#undef set_mask_norm
#endif
#ifdef set_vector_mask
#undef set_vector_mask
#endif
#define pipe_barrier(...) pto_costmodel_pipe_barrier(__VA_ARGS__)
#endif

#endif
