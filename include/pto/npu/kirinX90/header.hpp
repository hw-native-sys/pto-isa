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
#define hifloat8_t int8_t
#define float8_e4m3_t int8_t
#define float8_e5m2_t int8_t
#define float8_e8m0_t int8_t
#define float4_e2m1x2_t int64_t
#define float4_e1m2x2_t int64_t
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/kirinX90/common.hpp"
#include "pto/npu/kirin9030/utils.hpp"
#include "pto/npu/kirinX90/datatype.hpp"
#include "pto/npu/a2a3/TSubView.hpp"
#include "pto/npu/kirinX90/TLoad.hpp"
#include "pto/npu/kirinX90/TStore.hpp"
#include "pto/npu/kirinX90/TExtract.hpp"
#include "pto/npu/kirinX90/TInsert.hpp"
#include "pto/npu/kirinX90/TMov.hpp"
#include "pto/npu/a2a3/TImg2col.hpp"
#include "pto/npu/a2a3/SetFmatrix.hpp"
#include "pto/npu/a2a3/SetImg2colRpt.hpp"
#include "pto/npu/a2a3/SetImg2colPadding.hpp"
#ifdef __DAV_VEC__
#include "pto/npu/kirinX90/TCvt.hpp"
#endif
#include "pto/npu/a2a3/TAssign.hpp"
#include "pto/npu/kirin9030/TSync.hpp"
#include "pto/npu/a5/TAdd.hpp"
#include "pto/npu/a5/TAddS.hpp"
#include "pto/npu/a5/TDivS.hpp"
#include "pto/npu/a5/TMulS.hpp"
#include "pto/npu/a5/TSub.hpp"
#include "pto/npu/a5/TMin.hpp"
#include "pto/npu/a5/TMax.hpp"
#include "pto/npu/a5/TMrgSort.hpp"
#include "pto/npu/kirin9030/TMatmul.hpp"
#include "pto/npu/a5/TCmps.hpp"
#include "pto/npu/a5/TColSum.hpp"
#include "pto/npu/a5/TReshape.hpp"
#include "pto/npu/a5/TRowReduce.hpp"
#include "pto/npu/a5/TFillPad.hpp"
#include "pto/npu/a5/TTrans.hpp"
#include "pto/npu/a5/Tci.hpp"
#include "pto/npu/a5/TSel.hpp"
#include "pto/npu/a5/TSort32.hpp"
#include "pto/npu/a5/TRowExpand.hpp"
#include "pto/npu/a5/TPartAdd.hpp"
#include "pto/npu/a5/TPartMax.hpp"
#include "pto/npu/a5/TPartMin.hpp"
#include "pto/npu/kirin9030/TGather.hpp"
#include "pto/npu/kirin9030/TQuant.hpp"
#include "pto/npu/a5/TRsqrt.hpp"
#include "pto/npu/a5/TUnaryOp.hpp"
#include "pto/npu/a5/TBinSOp.hpp"
#include "pto/npu/a5/TAnd.hpp"
#include "pto/npu/a5/TAndS.hpp"
#include "pto/npu/a5/TOr.hpp"
#include "pto/npu/a5/TOrS.hpp"
#include "pto/npu/a5/TXor.hpp"
#include "pto/npu/a5/TXorS.hpp"
#include "pto/npu/a5/TShl.hpp"
#include "pto/npu/a5/TShlS.hpp"
#include "pto/npu/a5/TShr.hpp"
#include "pto/npu/a5/TShrS.hpp"
#include "pto/npu/a5/TAxpy.hpp"
#include "pto/npu/a5/TPrelu.hpp"
#include "pto/npu/a5/TLRelu.hpp"
#include "pto/npu/kirin9030/TSubS.hpp"
#include "pto/npu/a5/TMaxs.hpp"
#include "pto/npu/a5/TMins.hpp"
#include "pto/npu/a5/TCmp.hpp"
#include "pto/npu/a5/TSels.hpp"
#include "pto/npu/a5/TRowExpandAdd.hpp"
#include "pto/npu/a5/TRowExpandSub.hpp"
#include "pto/npu/a5/TRowExpandMul.hpp"
#include "pto/npu/a5/TRowExpandDiv.hpp"
#include "pto/npu/a5/TRowExpandMax.hpp"
#include "pto/npu/a5/TRowExpandMin.hpp"
#include "pto/npu/a5/TRowProd.hpp"
#include "pto/npu/a5/TRowReduceIdx.hpp"
#include "pto/npu/a5/TColProd.hpp"
#include "pto/npu/a5/TColMax.hpp"
#include "pto/npu/a5/TColMin.hpp"
#include "pto/npu/a5/TColReduceIdx.hpp"
#include "pto/npu/a5/TColExpand.hpp"
#include "pto/npu/a5/TColExpandAdd.hpp"
#include "pto/npu/a5/TColExpandSub.hpp"
#include "pto/npu/a5/TColExpandMul.hpp"
#include "pto/npu/a5/TColExpandDiv.hpp"
#include "pto/npu/a5/TColExpandMax.hpp"
#include "pto/npu/a5/TColExpandMin.hpp"
#include "pto/npu/a5/TConcat.hpp"
#include "pto/npu/a5/TExpandS.hpp"
#include "pto/npu/a5/TTri.hpp"
#include "pto/npu/a5/TPartMul.hpp"
#include "pto/npu/a5/TPartArgMax.hpp"
#include "pto/npu/a5/TPartArgMin.hpp"
#include "pto/npu/a5/TPow.hpp"
#include "pto/npu/a5/TGatherB.hpp"
#include "pto/npu/a5/TScatter.hpp"
#include "pto/npu/a5/TDiv.hpp"
#include "pto/npu/a5/TMul.hpp"
#include "pto/npu/a5/TFMod.hpp"
#include "pto/npu/a5/TFModS.hpp"
#include "pto/npu/a5/TColProd.hpp"
#include "pto/npu/a5/TRowExpandExpdif.hpp"
#include "pto/npu/a5/TColExpandExpdif.hpp"
#undef bfloat16_t
#undef hifloat8_t
#undef float8_e4m3_t
#undef float8_e5m2_t
#undef float8_e8m0_t
#undef float4_e2m1x2_t
#undef float4_e1m2x2_t
#endif
