/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_LINX_TARGET_HPP
#define PTO_NPU_LINX_TARGET_HPP

#include <pto/npu/linx/compat.hpp>

#if !defined(__linx) && !defined(PTO_LINX_CPU_SIM)
#error "The Linx backend requires __linx or PTO_LINX_CPU_SIM"
#endif

#if defined(__CPU_SIM) || defined(__COSTMODEL) || defined(__CCE_AICORE__)
#error "The Linx backend cannot be combined with CANN or PTO CPU-sim macros"
#endif

#if defined(__linx) && (defined(PTO_LINX_CPU_SIM) || defined(__cpu_sim__))
#error "Linx device and CPU simulator targets are mutually exclusive"
#endif
#if defined(__NPU_ARCH__) || defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || \
    defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90) || defined(PTO_NPU_ARCH_KIRIN9050)
#error "Select exactly one NPU target family for Linx"
#endif

#endif // PTO_NPU_LINX_TARGET_HPP
