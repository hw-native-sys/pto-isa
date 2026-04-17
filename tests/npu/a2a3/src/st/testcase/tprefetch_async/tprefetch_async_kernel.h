/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_RUNTIME_STUB_HPP
#define PTO_MOCKER_RUNTIME_STUB_HPP

<<<<<<<< HEAD:tests/npu/a2a3/src/st/testcase/tprefetch_async/tprefetch_async_kernel.h
#pragma once

#include <cstddef>

// Single-card host runner for the public TPREFETCH_ASYNC GlobalTensor API
// correctness test. Defined in tprefetch_async_kernel.cpp.
template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(int deviceId);
========
#include <pto/costmodel/common/qualifiers.hpp>
#include <pto/costmodel/common/aclrt_stub.hpp>
#include <pto/costmodel/common/runtime_util.hpp>
#include <pto/costmodel/common/arch_select.hpp>

#endif
>>>>>>>> c610a3e5 (costmodel refactoring):include/pto/costmodel/runtime_stub.hpp
