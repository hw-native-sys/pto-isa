/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPUSTUB_HPP
#define PTO_GPUSTUB_HPP

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#define PTO_GPU_BACKEND
#define PTO_GPU_INLINE_PTX 1

#define __aicore__ __device__
#define AICORE __device__
#define PTO_INLINE __forceinline__
#define PTO_INST AICORE PTO_INLINE
#define PTO_INTERNAL AICORE PTO_INLINE

#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __tf__

#ifndef __cce_get_tile_ptr
#define __cce_get_tile_ptr(x) (x)
#endif

typedef int pipe_t;
constexpr pipe_t PIPE_S = 0;
constexpr pipe_t PIPE_V = 1;
constexpr pipe_t PIPE_MTE1 = 2;
constexpr pipe_t PIPE_MTE2 = 3;
constexpr pipe_t PIPE_MTE3 = 4;
constexpr pipe_t PIPE_M = 5;
constexpr pipe_t PIPE_ALL = 6;
constexpr pipe_t PIPE_FIX = 7;

using event_t = int;
constexpr event_t EVENT_ID0 = 0;

PTO_INTERNAL void pipe_barrier(pipe_t)
{}

PTO_INTERNAL void set_flag(pipe_t, pipe_t, event_t)
{}

PTO_INTERNAL void wait_flag(pipe_t, pipe_t, event_t)
{}

#endif

#endif
