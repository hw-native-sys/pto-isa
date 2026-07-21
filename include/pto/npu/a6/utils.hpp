/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_UTILS_A6_H
#define PTO_UTILS_A6_H

// A6 (dav-9201) reuses the A5 vector-store helpers (DistVST, GetDistVst,
// PSetWithType, GetScaleAddr, VECTOR_REG_WIDTH, ...). Forward to the A5
// implementation to avoid divergence; revisit once A6 adds distinct store modes.
#include "pto/npu/a5/utils.hpp"

#endif
