/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_DATATYPE_HPP_KIRINX90
#define PTO_DATATYPE_HPP_KIRINX90

#include <pto/npu/kirin9030/datatype.hpp>

namespace pto {
#if defined(__DAV_VEC__)
template <>
struct TypeGet<vector_bf16> {
    using T = vector_bf16;
};
#endif
} // namespace pto

#endif
