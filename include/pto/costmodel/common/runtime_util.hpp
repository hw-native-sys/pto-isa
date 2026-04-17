/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_COMMON_RUNTIME_UTIL_HPP
#define PTO_MOCKER_COMMON_RUNTIME_UTIL_HPP

#include <cstdlib>
#include <type_traits>

#include <pto/costmodel/common/qualifiers.hpp>

inline CceEventIdType __pto_set_flag(pipe_t, pipe_t)
{
    return EVENT_ID0;
}
inline void __pto_wait_flag(pipe_t, pipe_t, CceEventIdType)
{}
inline void trap()
{
    std::abort();
}

inline int get_subblockid()
{
    return 0;
}
inline int get_rsvd_cnt()
{
    return 0;
}

template <typename T, typename U>
inline constexpr std::common_type_t<T, U> max(T lhs, U rhs)
{
    using Result = std::common_type_t<T, U>;
    return (lhs < rhs) ? static_cast<Result>(rhs) : static_cast<Result>(lhs);
}

template <typename T, typename U>
inline constexpr std::common_type_t<T, U> min(T lhs, U rhs)
{
    using Result = std::common_type_t<T, U>;
    return (lhs < rhs) ? static_cast<Result>(lhs) : static_cast<Result>(rhs);
}

#endif
