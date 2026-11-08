#ifndef __INCLUDE_CPU_TYPE_HEADER__
#define __INCLUDE_CPU_TYPE_HEADER__

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
