/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GEMM_BASIC_TORCH_CSRC_UTILS_H
#define PTO_GEMM_BASIC_TORCH_CSRC_UTILS_H

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace pto_path {

constexpr auto kNpuDevice = c10::DeviceType::PrivateUse1;

template <typename Arg>
decltype(auto) AdaptKernelArg(Arg &&arg)
{
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Arg>>, at::Tensor>) {
        return const_cast<void *>(arg.storage().data());
    } else {
        return std::forward<Arg>(arg);
    }
}

template <typename... Args>
auto AdaptKernelArgs(Args &&...args)
{
    return std::make_tuple(AdaptKernelArg(std::forward<Args>(args))...);
}

#define INVOKE_PTO_KERNEL(kernel_name, blk, ...)                                                                \
    do {                                                                                                        \
        auto __s = c10_npu::getCurrentNPUStream().stream(false);                                                \
        auto __p = AdaptKernelArgs(__VA_ARGS__);                                                                \
        auto __fn = [__s, blk, __p]() -> int {                                                                  \
            uint32_t __rc = 0;                                                                                  \
            std::apply([&](auto &&...__a) { __rc = ACLRT_LAUNCH_KERNEL(kernel_name)(blk, __s, __a...); }, __p); \
            return __rc;                                                                                        \
        };                                                                                                      \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, __fn);                                                \
    } while (false)

} // namespace pto_path

#endif
