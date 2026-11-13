/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE__INCLUDE_TYPE_H_
#define PTO_LINX_NATIVE__INCLUDE_TYPE_H_

template <typename E> struct type_traits {
  static constexpr int bits = sizeof(E) * 8;
};

#if defined(PTO_CPU_SIM_TARGET_LINX)
#include <pto/cpu/linx/type.hpp>
#else
#include <pto/npu/linx/impl/jcore/type.hpp>
#endif

#endif
