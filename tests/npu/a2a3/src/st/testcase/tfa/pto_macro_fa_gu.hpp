/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_GU_HPP
#define PTO_MACRO_FA_GU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename reducedTileData, typename svTileData>
PTO_INTERNAL void pto_macro_fa_gu(
    svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile, reducedTileData __in__ exp_max)
{
    pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
    pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
}

template <typename reducedTileData, typename svTileData>
PTO_INTERNAL void pto_macro_fa_gu_last(
    svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile, reducedTileData __in__ exp_max,
    reducedTileData __in__ new_global_sum)
{
    pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
    pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
    pto::TROWEXPANDDIV(prev_sv_tile, prev_sv_tile, new_global_sum);
    // pto::TCVT(prev_sv_nd_tile, prev_sv_tile, RoundMode::CAST_RINT);
}

} // namespace pto
#endif // TGU_PTO_H