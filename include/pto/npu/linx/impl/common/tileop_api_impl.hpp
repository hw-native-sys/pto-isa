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

#ifndef PTO_LINX_VENDOR_TILEOP_API_IMPL_HPP
#define PTO_LINX_VENDOR_TILEOP_API_IMPL_HPP

#ifdef __linx
#include "pto/npu/linx/impl/jcore/template_asm.hpp"

#elif defined(__cpu_sim__)
#include "pto/npu/linx/impl/cpu_sim/MatMacc.hpp"
#include "pto/npu/linx/impl/cpu_sim/MatMul.hpp"
#include "pto/npu/linx/impl/cpu_sim/TAbs.hpp"
#include "pto/npu/linx/impl/cpu_sim/TAdd.hpp"
#include "pto/npu/linx/impl/cpu_sim/TAdds.hpp"
#include "pto/npu/linx/impl/cpu_sim/TAnd.hpp"
#include "pto/npu/linx/impl/cpu_sim/TAssemble.hpp"
#include "pto/npu/linx/impl/cpu_sim/TCast.hpp"
#include "pto/npu/linx/impl/cpu_sim/TCI.hpp"
#include "pto/npu/linx/impl/cpu_sim/TCmp.hpp"
#include "pto/npu/linx/impl/cpu_sim/TCopy.hpp"
#include "pto/npu/linx/impl/cpu_sim/TLoadBackend.hpp"
#include "pto/npu/linx/impl/cpu_sim/TStoreBackend.hpp"
#include "pto/npu/linx/impl/cpu_sim/TCvt.hpp"
#include "pto/npu/linx/impl/cpu_sim/TDiv.hpp"
#include "pto/npu/linx/impl/cpu_sim/TDivs.hpp"
#include "pto/npu/linx/impl/cpu_sim/TExp.hpp"
#include "pto/npu/linx/impl/cpu_sim/TExpdif.hpp"
#include "pto/npu/linx/impl/cpu_sim/TExpandRow.hpp"
#include "pto/npu/linx/impl/cpu_sim/TExtract.hpp"
#include "pto/npu/linx/impl/cpu_sim/TFillPad.hpp"
#include "pto/npu/linx/impl/cpu_sim/TGather.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMax.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMaxs.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMin.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMins.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMul.hpp"
#include "pto/npu/linx/impl/cpu_sim/TMuls.hpp"
#include "pto/npu/linx/impl/cpu_sim/TOr.hpp"
#include "pto/npu/linx/impl/cpu_sim/TXor.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSll.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSrl.hpp"
#include "pto/npu/linx/impl/cpu_sim/TPad.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRSqrt.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRecip.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRem.hpp"
#include "pto/npu/linx/impl/cpu_sim/TReshape.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRowMax.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRowMaxExpand.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRowSum.hpp"
#include "pto/npu/linx/impl/cpu_sim/TRowSumExpand.hpp"
#include "pto/npu/linx/impl/cpu_sim/TScatter.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSelect.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSqrt.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSub.hpp"
#include "pto/npu/linx/impl/cpu_sim/TSubs.hpp"
#include "pto/npu/linx/impl/cpu_sim/TTrans.hpp"

#else
#error "__linx or __cpu_sim__ must be defined"

#endif
#endif
