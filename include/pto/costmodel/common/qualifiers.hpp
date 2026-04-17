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

#include <cstdint>

#define __global__
#define AICORE
#define __aicore__
#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __tf__
#define __biasbuf__

using aclrtStream = void *;
using aclrtContext = void *;
using pipe_t = int;

inline constexpr pipe_t PIPE_S = 0;
inline constexpr pipe_t PIPE_V = 1;
inline constexpr pipe_t PIPE_MTE1 = 2;
inline constexpr pipe_t PIPE_MTE2 = 3;
inline constexpr pipe_t PIPE_MTE3 = 4;
inline constexpr pipe_t PIPE_M = 5;
inline constexpr pipe_t PIPE_ALL = 6;

using event_t = int;
using CceEventIdType = event_t;
using pad_t = int;
using addr_cal_mode_t = int;

namespace __cce_scalar {
using addr_cal_mode_t = ::addr_cal_mode_t;
}

inline constexpr event_t EVENT_ID0 = 0;
inline constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
inline constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
inline constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;
inline constexpr int ACL_MEMCPY_DEVICE_TO_DEVICE = 2;
inline constexpr int ACL_STREAM_FAST_LAUNCH = 0;
inline constexpr int ACL_STREAM_FAST_SYNC = 0;
inline constexpr int ACL_STREAM_ATTR_FAILURE_MODE = 0;
inline constexpr int ACL_RT_CMO_TYPE_PREFETCH = 0;
inline constexpr int ONLY_INDEX = 0;
inline constexpr int VALUE_INDEX = 1;

using aclrtStreamAttrValue = int;

#define aclFloat16ToFloat(x) ((float)(x))
#define __cce_get_tile_ptr(x) x

#endif
