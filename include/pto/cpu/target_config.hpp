// --------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

#ifndef PTO_CPU_TARGET_CONFIG_HPP
#define PTO_CPU_TARGET_CONFIG_HPP

#if defined(PTO_CPU_SIM_TARGET_A2A3) && defined(PTO_CPU_SIM_TARGET_A5)
#error "Select exactly one PTO CPU simulator target family"
#endif

#if defined(__linx)
#define PTO_NPU_ARCH_LINX
#endif

#if defined(PTO_LINX_CPU_SIM)
#ifndef __cpu_sim__
#error "PTO_LINX_CPU_SIM requires the Linx __cpu_sim__ backend"
#endif
#ifndef PTO_NPU_ARCH_LINX
#define PTO_NPU_ARCH_LINX
#endif
#endif

#if defined(PTO_CPU_SIM_TARGET_A2A3) || defined(PTO_CPU_SIM_TARGET_A5)
#ifndef __CPU_SIM
#error "PTO CPU simulator target flags require __CPU_SIM"
#endif
#if defined(__NPU_ARCH__) || defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || \
    defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90) || defined(PTO_NPU_ARCH_KIRIN9050) ||        \
    defined(PTO_NPU_ARCH_LINX)
#error "PTO CPU simulator target flags cannot be combined with NPU target flags"
#endif
#endif

// Fixed-family types and entry points have distinct C++ symbols across kernel libraries.
#if defined(PTO_CPU_SIM_TARGET_A5)
#define PTO_CPU_TARGET_NAMESPACE_BEGIN inline namespace cpu_target_a5 {
#define PTO_CPU_TARGET_NAMESPACE_END }
#elif defined(PTO_CPU_SIM_TARGET_A2A3)
#define PTO_CPU_TARGET_NAMESPACE_BEGIN inline namespace cpu_target_a2a3 {
#define PTO_CPU_TARGET_NAMESPACE_END }
#else
#define PTO_CPU_TARGET_NAMESPACE_BEGIN
#define PTO_CPU_TARGET_NAMESPACE_END
#endif

namespace pto {
enum class NPUArch { A2A3, A5 };
enum class CpuSimTarget { Legacy, A2A3, A5, Linx };
#if defined(PTO_CPU_SIM_TARGET_A5)
static constexpr CpuSimTarget CPU_SIM_TARGET = CpuSimTarget::A5;
#elif defined(PTO_CPU_SIM_TARGET_A2A3)
static constexpr CpuSimTarget CPU_SIM_TARGET = CpuSimTarget::A2A3;
#elif defined(PTO_LINX_CPU_SIM)
static constexpr CpuSimTarget CPU_SIM_TARGET = CpuSimTarget::Linx;
#else
static constexpr CpuSimTarget CPU_SIM_TARGET = CpuSimTarget::Legacy;
#endif
} // namespace pto

#endif // PTO_CPU_TARGET_CONFIG_HPP
