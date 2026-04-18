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
#define bfloat16_t half
#define float8_e4m3_t int8_t
#define float8_e5m2_t int8_t
#define hifloat8_t int8_t
#define float8_e8m0_t int8_t
#define float4_e2m1x2_t int64_t
#define float4_e1m2x2_t int64_t
#include "pto/npu/a2a3/TAssign.hpp"
#include "pto/npu/kirin9030/TSync.hpp"
#include "pto/npu/kirin9030/TAdd.hpp"
#include "pto/npu/kirin9030/TAddS.hpp"
#include "pto/npu/kirin9030/TDivS.hpp"
#include "pto/npu/kirin9030/TMulS.hpp"
#include "pto/npu/kirin9030/TSub.hpp"
#include "pto/npu/kirin9030/TMin.hpp"
#include "pto/npu/kirin9030/TMax.hpp"
#include "pto/npu/kirin9030/TLoad.hpp"
#ifdef __DAV_VEC__
#include "pto/npu/kirin9030/TCvt.hpp"
#endif
#include "pto/npu/kirin9030/TStore.hpp"
#include "pto/npu/kirin9030/TMrgSort.hpp"
#include "pto/npu/kirin9030/TMatmul.hpp"
#include "pto/npu/kirin9030/TCmps.hpp"
#include "pto/npu/kirin9030/TColSum.hpp"
#include "pto/npu/kirin9030/TReshape.hpp"
#include "pto/npu/kirin9030/TRowReduce.hpp"
#include "pto/npu/kirin9030/TFillPad.hpp"
#include "pto/npu/kirin9030/TTrans.hpp"
#include "pto/npu/kirin9030/Tci.hpp"
#include "pto/npu/kirin9030/TSel.hpp"
#include "pto/npu/kirin9030/TSort32.hpp"
#include "pto/npu/kirin9030/TExtract.hpp"
#include "pto/npu/kirin9030/TMov.hpp"
#include "pto/npu/kirin9030/TRowExpand.hpp"
#include "pto/npu/kirin9030/TPartAdd.hpp"
#include "pto/npu/kirin9030/TPartMax.hpp"
#include "pto/npu/kirin9030/TInsert.hpp"
#include "pto/npu/kirin9030/TPartMin.hpp"
#include "pto/npu/kirin9030/TGather.hpp"
#include "pto/npu/kirin9030/TRsqrt.hpp"
#include "pto/npu/kirin9030/TUnaryOp.hpp"
#include "pto/npu/kirin9030/TBinSOp.hpp"
#include "pto/npu/kirin9030/TDiv.hpp"
#include "pto/npu/kirin9030/TMul.hpp"
#include "pto/npu/a5/TCmp.hpp"
#include "pto/npu/a5/TExpandS.hpp"
#include "pto/npu/a5/TColExpand.hpp"
#include "pto/npu/a5/TMaxs.hpp"
#include "pto/npu/a5/TMins.hpp"
#undef bfloat16_t
#undef float8_e4m3_t
#undef float8_e5m2_t
#undef hifloat8_t
#undef float8_e8m0_t
#undef float4_e2m1x2_t
#undef float4_e1m2x2_t
#endif
