/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGATHER_HPP
#define PTO_COMM_TGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TGATHER_IMPL: Gather operation - root collects data from all ranks
//
// The calling NPU is the root and gathers data from all ranks, concatenating
// the results along DIM_3 (row dimension) into a local output buffer.
//
// Each rank r contributes data of shape (D0, D1, D2, H, W). The destination
// tensor has shape (D0, D1, D2, N*H, W), where rank r's data is placed at
// rows [r*H, (r+1)*H).
//
// When the per-rank GlobalTensor exceeds the UB tile capacity in rows and/or
// columns, the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, per-rank DIM_3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, DIM_4 must be divisible by ValidCol.
//   - All source tensors in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                               TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    // CPU Logic is not implemented yet
}

// ============================================================================
// TGATHER_IMPL (ping-pong): Gather with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD of the next
// chunk (MTE2) with TSTORE of the current chunk (MTE3).
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Constraints: same as TGATHER_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &pingTile,
                               TileData &pongTile)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    // CPU Logic is not implemented yet
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGATHER_HPP
