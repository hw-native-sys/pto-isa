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

// A6 (dav-9201) shares the A5 core, so the whole A5 instruction set is reusable.
// Instructions with hardware-level differences have dedicated A6 implementations
// (TLoad/TExtract/TMatmul/TQuant/TSync/SyncAll/TShl/TShlS/TShr/TShrS); all the
// rest are A5 headers.
#include "pto/npu/a6/common.hpp"
#include "pto/npu/a6/datatype.hpp"
#include "pto/npu/a6/utils.hpp"
#include "pto/npu/a6/TSync.hpp"
#include "pto/npu/a6/SyncAll.hpp"
#include "pto/npu/a6/TLoad.hpp"
#include "pto/npu/a6/TExtract.hpp"
#include "pto/npu/a6/TMatmul.hpp"
#include "pto/npu/a6/TQuant.hpp"
#include "pto/npu/a6/TReshape.hpp"
#include "pto/npu/a6/TShl.hpp"
#include "pto/npu/a6/TShlS.hpp"
#include "pto/npu/a6/TShr.hpp"
#include "pto/npu/a6/TShrS.hpp"
#include "pto/npu/a6/TInterleave.hpp"
#include "pto/npu/a6/TDeInterleave.hpp"

#include "pto/npu/a5/MGather.hpp"
#include "pto/npu/a5/MScatter.hpp"
#include "pto/npu/a5/SetFmatrix.hpp"
#include "pto/npu/a5/SetImg2colPadding.hpp"
#include "pto/npu/a5/SetImg2colRpt.hpp"
#include "pto/npu/a5/SetQuantScalar.hpp"
#include "pto/npu/a5/SetQuantVector.hpp"
#include "pto/npu/a5/TAdd.hpp"
#include "pto/npu/a5/TAddS.hpp"
#include "pto/npu/a5/TAlloc.hpp"
#include "pto/npu/a5/TAnd.hpp"
#include "pto/npu/a5/TAndS.hpp"
#include "pto/npu/a5/TAssign.hpp"
#include "pto/npu/a5/TAxpy.hpp"
#include "pto/npu/a5/TBinSOp.hpp"
#include "pto/npu/a5/TCmp.hpp"
#include "pto/npu/a5/TCmps.hpp"
#include "pto/npu/a5/TColExpand.hpp"
#include "pto/npu/a5/TColExpandAdd.hpp"
#include "pto/npu/a5/TColExpandDiv.hpp"
#include "pto/npu/a5/TColExpandExpdif.hpp"
#include "pto/npu/a5/TColExpandMax.hpp"
#include "pto/npu/a5/TColExpandMin.hpp"
#include "pto/npu/a5/TColExpandMul.hpp"
#include "pto/npu/a5/TColExpandSub.hpp"
#include "pto/npu/a5/TColMax.hpp"
#include "pto/npu/a5/TColMin.hpp"
#include "pto/npu/a5/TColProd.hpp"
#include "pto/npu/a5/TColReduceIdx.hpp"
#include "pto/npu/a5/TColSum.hpp"
#include "pto/npu/a5/TConcat.hpp"
#ifdef PTO_COMPILE_VEC
#include "pto/npu/a6/TCvt.hpp"
#endif
#include "pto/npu/a5/TDeQuant.hpp"
#include "pto/npu/a5/TDiv.hpp"
#include "pto/npu/a5/TDivS.hpp"
#include "pto/npu/a5/TExpandS.hpp"
#include "pto/npu/a5/TFMod.hpp"
#include "pto/npu/a5/TFModS.hpp"
#include "pto/npu/a5/TFillPad.hpp"
#include "pto/npu/a5/TFree.hpp"
#include "pto/npu/a5/TGather.hpp"
#include "pto/npu/a5/TGatherB.hpp"
#include "pto/npu/a5/THistogram.hpp"
#include "pto/npu/a5/TImg2col.hpp"
#include "pto/npu/a5/TInsert.hpp"
#include "pto/npu/a5/TLRelu.hpp"
#include "pto/npu/a5/TMAdd.hpp"
#include "pto/npu/a5/TMax.hpp"
#include "pto/npu/a5/TMaxs.hpp"
#include "pto/npu/a5/TMin.hpp"
#include "pto/npu/a5/TMins.hpp"
#include "pto/npu/a5/TMov.hpp"
#include "pto/npu/a5/TMrgSort.hpp"
#include "pto/npu/a5/TMul.hpp"
#include "pto/npu/a5/TMulS.hpp"
#include "pto/npu/a5/TMula.hpp"
#include "pto/npu/a5/TOr.hpp"
#include "pto/npu/a5/TOrS.hpp"
#include "pto/npu/a5/TPartAdd.hpp"
#include "pto/npu/a5/TPartArgMax.hpp"
#include "pto/npu/a5/TPartArgMin.hpp"
#include "pto/npu/a5/TPartMax.hpp"
#include "pto/npu/a5/TPartMin.hpp"
#include "pto/npu/a5/TPartMul.hpp"
#include "pto/npu/a5/TPop.hpp"
#include "pto/npu/a5/TPow.hpp"
#include "pto/npu/a5/TPrefetch.hpp"
#include "pto/npu/a5/TPrelu.hpp"
#ifdef _DEBUG
#include "pto/npu/a5/TPrint.hpp"
#endif
#include "pto/npu/a5/TPush.hpp"
#include "pto/npu/a5/TRandom.hpp"
#include "pto/npu/a5/TRem.hpp"
#include "pto/npu/a5/TRemS.hpp"
#include "pto/npu/a5/TRowExpand.hpp"
#include "pto/npu/a5/TRowExpandAdd.hpp"
#include "pto/npu/a5/TRowExpandDiv.hpp"
#include "pto/npu/a5/TRowExpandExpdif.hpp"
#include "pto/npu/a5/TRowExpandMax.hpp"
#include "pto/npu/a5/TRowExpandMin.hpp"
#include "pto/npu/a5/TRowExpandMul.hpp"
#include "pto/npu/a5/TRowExpandSub.hpp"
#include "pto/npu/a5/TRowProd.hpp"
#include "pto/npu/a5/TRowReduce.hpp"
#include "pto/npu/a5/TRowReduceIdx.hpp"
#include "pto/npu/a5/TRsqrt.hpp"
#include "pto/npu/a5/TScatter.hpp"
#include "pto/npu/a5/TSel.hpp"
#include "pto/npu/a5/TSels.hpp"
#include "pto/npu/a5/TSort32.hpp"
#include "pto/npu/a5/TStore.hpp"
#include "pto/npu/a5/TSub.hpp"
#include "pto/npu/a5/TSubS.hpp"
#include "pto/npu/a5/TTrans.hpp"
#include "pto/npu/a5/TTri.hpp"
#include "pto/npu/a5/TUnaryOp.hpp"
#include "pto/npu/a5/TXor.hpp"
#include "pto/npu/a5/TXorS.hpp"
#include "pto/npu/a5/Tci.hpp"

// Async L2 cache prefetch via SDMA CMO (arch-neutral SDMA-backed implementation).
#if defined(__CCE_AICORE__) && !(defined(__CPU_SIM) || defined(__COSTMODEL))
#include "pto/npu/a5/TPrefetchAsync.hpp"
#endif

#endif
