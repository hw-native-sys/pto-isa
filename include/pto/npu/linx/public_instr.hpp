/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_PUBLIC_INSTR_HPP
#define PTO_LINX_PUBLIC_INSTR_HPP

namespace pto {
namespace detail {
template <typename... WaitEvents>
PTO_INTERNAL void PtoWaitEvents(WaitEvents&...)
{
    static_assert(sizeof...(WaitEvents) == 0, "Linx does not support PTO pipe events yet");
}
} // namespace detail

#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...) linx::API##_IMPL TEMPLATE_ARGS(__VA_ARGS__)
#define MAP_INSTR_IMPL(API, ...) linx::API##_IMPL(__VA_ARGS__)
#define PTO_FORWARD_L2HINT_TO_IMPL 0
#include <pto/common/pto_instr_basic.hpp>
#undef PTO_FORWARD_L2HINT_TO_IMPL
#undef MAP_INSTR_IMPL_T
#undef PTO_TEMPLATE_ARGS
#undef MAP_INSTR_IMPL

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    linx::TSTORE_IMPL(dst, src);
    return {};
}
} // namespace pto
#endif
