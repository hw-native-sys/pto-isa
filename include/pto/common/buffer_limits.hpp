/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_BUFFER_LIMITS_HPP
#define PTO_COMMON_BUFFER_LIMITS_HPP

#include <cstddef>

// =============================================================================
// On-chip buffer capacity and alignment per platform.
//
// Each macro can be overridden by the build system (-D flag) to accommodate
// non-standard configurations.  When not overridden, defaults are derived from
// the hardware reference for A2A3 / A5 / Kirin9030 / KirinX90.
//
// All values are in BYTES.
// =============================================================================

// ---- UB (Vec) ----
#ifndef PTO_UBUF_ALIGN_BYTES
#define PTO_UBUF_ALIGN_BYTES 32u
#endif

#ifndef PTO_UBUF_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A5)
#define PTO_UBUF_SIZE_BYTES (256u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3)
#define PTO_UBUF_SIZE_BYTES (192u * 1024u)
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_UBUF_SIZE_BYTES (128u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_UBUF_SIZE_BYTES (256u * 1024u)
#else
#error \
    "PTO_UBUF_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_UBUF_SIZE_BYTES manually."
#endif
#endif

// ---- L1 / CB (Mat) ----
#ifndef PTO_CBUF_ALIGN_BYTES
#define PTO_CBUF_ALIGN_BYTES 32u
#endif

#ifndef PTO_CBUF_SIZE_BYTES
#if defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_CBUF_SIZE_BYTES (1024u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
#define PTO_CBUF_SIZE_BYTES (512u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_CBUF_SIZE_BYTES (256u * 1024u)
#else
#error \
    "PTO_CBUF_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_CBUF_SIZE_BYTES manually."
#endif
#endif

// ---- L0A (Left / ScaleLeft) ----
#ifndef PTO_L0A_ALIGN_BYTES
#define PTO_L0A_ALIGN_BYTES 32u
#endif

#ifndef PTO_L0A_SIZE_BYTES
#if defined(PTO_NPU_ARCH_KIRIN9030)
#define PTO_L0A_SIZE_BYTES (32u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_L0A_SIZE_BYTES (64u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_L0A_SIZE_BYTES (128u * 1024u)
#else
#error \
    "PTO_L0A_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_L0A_SIZE_BYTES manually."
#endif
#endif

// ---- L0B (Right / ScaleRight) ----
#ifndef PTO_L0B_ALIGN_BYTES
#define PTO_L0B_ALIGN_BYTES 32u
#endif

#ifndef PTO_L0B_SIZE_BYTES
#if defined(PTO_NPU_ARCH_KIRIN9030)
#define PTO_L0B_SIZE_BYTES (32u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_L0B_SIZE_BYTES (64u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_L0B_SIZE_BYTES (128u * 1024u)
#else
#error \
    "PTO_L0B_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_L0B_SIZE_BYTES manually."
#endif
#endif

// ---- L0C (Acc) ----
#ifndef PTO_L0C_ALIGN_BYTES
#define PTO_L0C_ALIGN_BYTES 32u
#endif

#ifndef PTO_L0C_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A5)
#define PTO_L0C_SIZE_BYTES (256u * 1024u)
#elif defined(PTO_NPU_ARCH_KIRINX90) || defined(PTO_NPU_ARCH_A2A3)
#define PTO_L0C_SIZE_BYTES (128u * 1024u)
#elif defined(PTO_NPU_ARCH_KIRIN9030)
#define PTO_L0C_SIZE_BYTES (64u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_L0C_SIZE_BYTES (256u * 1024u)
#else
#error \
    "PTO_L0C_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_L0C_SIZE_BYTES manually."
#endif
#endif

// ---- Bias ----
#ifndef PTO_BIAS_ALIGN_BYTES
#define PTO_BIAS_ALIGN_BYTES 32u
#endif

#ifndef PTO_BIAS_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A5)
#define PTO_BIAS_SIZE_BYTES (4u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_BIAS_SIZE_BYTES (1u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_BIAS_SIZE_BYTES (8u * 1024u)
#else
#error \
    "PTO_BIAS_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_BIAS_SIZE_BYTES manually."
#endif
#endif

// ---- FBuffer (Scaling) ----
#ifndef PTO_FBUF_ALIGN_BYTES
#define PTO_FBUF_ALIGN_BYTES 32u
#endif

#ifndef PTO_FBUF_SIZE_BYTES
#if defined(PTO_NPU_ARCH_KIRIN9030)
#define PTO_FBUF_SIZE_BYTES (7u * 1024u)
#elif defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_FBUF_SIZE_BYTES (6u * 1024u)
#elif defined(PTO_NPU_ARCH_A5)
#define PTO_FBUF_SIZE_BYTES (4u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3)
#define PTO_FBUF_SIZE_BYTES (2u * 1024u)
#elif defined(PTO_GPU_BACKEND)
#define PTO_FBUF_SIZE_BYTES (8u * 1024u)
#else
#error \
    "PTO_FBUF_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_FBUF_SIZE_BYTES manually."
#endif
#endif

// ---- ScaleLeft (L0A, A5 only) ----
#ifndef PTO_SCALELEFT_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A5)
#define PTO_SCALELEFT_SIZE_BYTES (4u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_SCALELEFT_SIZE_BYTES 0u
#elif defined(PTO_GPU_BACKEND)
#define PTO_SCALELEFT_SIZE_BYTES (8u * 1024u)
#else
#error \
    "PTO_SCALELEFT_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_SCALELEFT_SIZE_BYTES manually."
#endif
#endif

// ---- ScaleRight (L0B, A5 only) ----
#ifndef PTO_SCALERIGHT_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A5)
#define PTO_SCALERIGHT_SIZE_BYTES (4u * 1024u)
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define PTO_SCALERIGHT_SIZE_BYTES 0u
#elif defined(PTO_GPU_BACKEND)
#define PTO_SCALERIGHT_SIZE_BYTES (8u * 1024u)
#else
#error \
    "PTO_SCALERIGHT_SIZE_BYTES: unknown NPU architecture. Define one of PTO_NPU_ARCH_{A2A3,A5,KIRIN9030,KIRINX90} or set PTO_SCALERIGHT_SIZE_BYTES manually."
#endif
#endif

#endif // PTO_COMMON_BUFFER_LIMITS_HPP
