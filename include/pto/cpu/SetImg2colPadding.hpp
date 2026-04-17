/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_COMMON_ARCH_SELECT_HPP
#define PTO_MOCKER_COMMON_ARCH_SELECT_HPP

#if !defined(__NPU_ARCH__)
#error "__NPU_ARCH__ must be defined for PTO costmodel."
#elif (__NPU_ARCH__ == 2201)
#include <pto/costmodel/a2a3/cce_costmodel.hpp>
#else
#error "PTO costmodel only supports __NPU_ARCH__ == 2201 (A2/A3)."
#endif

#endif
