/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6R2_HEADER_HPP
#define PTO_NPU_A6R2_HEADER_HPP

// A6R2 (dav-920r2) reuses A6, plus MaxTrack TPUSH/TPOP and a wider TPipe ctor.
// 1) Inject 5-arg TPipe ctor into a5/TPush.hpp without A6R2 ifdefs there.
// 2) After a6/header.hpp, pull MaxTrack TPUSH / TPOP from a6r2.
#define PTO_A6R2_TPUSH_CTOR_ONLY
#define PTO_TPIPE_EXTRA_CTORS_HEADER "pto/npu/a6r2/TPush.hpp"

#include "pto/npu/a6/header.hpp"

#undef PTO_TPIPE_EXTRA_CTORS_HEADER
#undef PTO_A6R2_TPUSH_CTOR_ONLY

#include "pto/npu/a6r2/TMov.hpp"
#include "pto/npu/a6r2/TMatmul.hpp"
#include "pto/npu/a6r2/TPush.hpp"
#include "pto/npu/a6r2/TPop.hpp"

#endif // PTO_NPU_A6R2_HEADER_HPP
