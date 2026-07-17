/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HEADER_HPP
#define HEADER_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

#include "pto/npu/a6/datatype.hpp"
#include "pto/npu/a6/TSync.hpp"

// A6 uses dedicated TLoad/TExtract/TMatmul implementations,
// while some other instructions still reuse A5.
#include "pto/npu/a5/TAssign.hpp"
#include "pto/npu/a5/SyncAll.hpp"
#include "pto/npu/a5/TAdd.hpp"
#include "pto/npu/a6/TLoad.hpp"
#include "pto/npu/a5/TStore.hpp"
#include "pto/npu/a6/TExtract.hpp"
#include "pto/npu/a6/TMatmul.hpp"

#endif
