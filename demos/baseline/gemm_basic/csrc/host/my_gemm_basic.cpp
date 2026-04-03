/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "utils.h"

#include "aclrtlaunch_gemm_basic_custom.h"

namespace pto_path {

constexpr uint32_t M = 512;
constexpr uint32_t K = 2048;
constexpr uint32_t N = 1536;
constexpr uint32_t DIM_2 = 2;

at::Tensor run_gemm_basic_custom(const at::Tensor &a, const at::Tensor &b_dn)
{
    TORCH_CHECK(a.device().type() == kNpuDevice, "a must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(b_dn.device().type() == kNpuDevice, "b_dn must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b_dn.scalar_type() == at::kHalf, "b_dn must be float16");
    TORCH_CHECK(a.dim() == DIM_2 && b_dn.dim() == DIM_2, "a and b_dn must be 2D tensors");
    TORCH_CHECK(a.size(0) == M && a.size(1) == K, "a shape must be [512, 2048]");
    TORCH_CHECK(b_dn.size(0) == N && b_dn.size(1) == K, "b_dn shape must be [1536, 2048] (b.t().contiguous())");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b_dn.is_contiguous(), "b_dn must be contiguous");

    auto out = at::empty({512, 1536}, a.options().dtype(at::kFloat));
    constexpr uint32_t blockDim = 24;
    INVOKE_PTO_KERNEL(gemm_basic_custom, blockDim, a, b_dn, out);
    return out;
}

} // namespace pto_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("pto_gemm_basic(Tensor a, Tensor b_dn) -> Tensor");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("pto_gemm_basic", TORCH_FN(pto_path::run_gemm_basic_custom));
}
} // namespace
