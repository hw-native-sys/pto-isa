/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_INCLUDE_CPU_TYPE_HEADER
#define PTO_LINX_VENDOR_INCLUDE_CPU_TYPE_HEADER

#include <stdint.h>

#ifdef __clang__
#if __clang_major__ >= 15
typedef _Float16 __half;
#endif
#endif

/*
| 架构 | 类型名   | 最低 GCC 版本  |
|------|----------|----------------|
| x86  | _Float16 | 12.1           |
| ARM  | __fp16   | ARMv8以上 |
*/
#if defined(__GNUC__) && !defined(__clang__)
#ifdef __x86_64__
#if (__GNUC__ > 12) || (__GNUC__ == 12 && __GNUC_MINOR__ >= 1)
typedef _Float16 __half;
#endif
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include "arm_fp16.h"
typedef __fp16 __half;
#endif
#endif

typedef float __fp32;

#endif
