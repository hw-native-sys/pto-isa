/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "utils.h"
#include "aclrtlaunch_add_custom.h"

namespace ascendc_path {

at::Tensor run_add_custom(const at::Tensor& x, const at::Tensor& y)
{
    at::Tensor z = at::empty_like(x);
    // Define the number of blocks of vector core
    uint32_t blockDim = 20;
    uint32_t totalLength = 1;
    // Calculate the total number of elements
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    // Launch the custom kernel
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}
} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    // Declare the custom operator schema
    m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    // Register the custom operator implementation function
    m.impl("my_add", TORCH_FN(ascendc_path::run_add_custom));
}
} // namespace
