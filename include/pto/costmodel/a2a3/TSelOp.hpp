/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELOP_HPP
#define TSELOP_HPP

#include <pto/common/constants.hpp>

namespace pto {
enum class SELMODE : uint8_t
{
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE = 1,
    VSEL_TENSOR_TENSOR_MODE = 2,
};

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL std::vector<CostModelStats> runTSelOp(DstTile &dst)
{
    std::vector<CostModelStats> stats;
    static_assert(sizeof(typename DstTile::DType) == 4 || sizeof(typename DstTile::DType) == 2,
                  "Fix: TSEL only support 16B and 32B data type.");
    static_assert(std::is_same_v<typename DstTile::DType, typename Src0Tile::DType> ||
                      std::is_same_v<typename DstTile::DType, typename Src1Tile::DType>,
                  "Fix: TSEL only support same data type between dst, src0, and src1.");
    static_assert(DstTile::isRowMajor && Src0Tile::isRowMajor && Src1Tile::isRowMajor,
                  "Fix: TSEL only support RowMajor layout type.");

    return stats;
}
} // namespace pto
#endif