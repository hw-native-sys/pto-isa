/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"

static constexpr size_t kCcuCount = 1024;

__global__ __aicore__ void treduce_ccu_trigger_kernel(uint64_t ckeVA, uint32_t mask)
{
    if (get_block_idx() != 0)
        return;

    using ShapeS = pto::Shape<1, 1, 1, 1, kCcuCount>;
    using StrideS = pto::Stride<kCcuCount, kCcuCount, kCcuCount, kCcuCount, 1>;
    using GmType = pto::GlobalTensor<float, ShapeS, StrideS, pto::Layout::ND>;
    using TileT = pto::Tile<pto::TileType::Vec, float, 1, kCcuCount, pto::BLayout::RowMajor, -1, -1>;

    GmType dstGm;
    TileT accTile;
    TileT recvTile;

    GmType rankTensors[1];
    pto::comm::ParallelGroup<GmType> group(rankTensors, 1, 0);

    pto::comm::CcuTriggerContext ctx{ckeVA, mask};

    pto::comm::TREDUCE<pto::comm::CollEngine::CCU>(group, dstGm, accTile, recvTile, pto::comm::ReduceOp::Sum, ctx);
}

extern "C" __attribute__((visibility("default"))) int treduce_ccu_trigger_launch(void *stream, uint64_t ckeVA,
                                                                                 uint32_t mask)
{
    treduce_ccu_trigger_kernel<<<1, nullptr, stream>>>(ckeVA, mask);
    return 0;
}
