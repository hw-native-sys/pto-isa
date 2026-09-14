/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_GATHER_SCATTER_GUARD_KERNEL_H
#define INT64_GATHER_SCATTER_GUARD_KERNEL_H

// Operation: indexed gather, pattern gather, indexed scatter, row scatter, column scatter.
template <
    typename T, int Operation, int Rows, int SrcCols, int ValidRows, int ValidCols, int DstCols,
    pto::MaskPattern Pattern = pto::MaskPattern::P1010>
__global__ AICORE void RunInt64GatherScatterGuard(__gm__ T* out, __gm__ T* input, __gm__ uint32_t* indices)
{
    using namespace pto;
    constexpr int times = GetTimesByMask<Pattern>();
    constexpr int dstRows = Operation == 4 ? Rows * times : Rows;
    constexpr int dstValidRows = Operation == 4 ? ValidRows * times : ValidRows;
    constexpr int dstValidCols = Operation == 3 ? ValidCols * times : ValidCols;
    constexpr int indexCols = (ValidCols + 7) / 8 * 8;
    constexpr int outputElements = dstRows * DstCols + 64;
    using SrcTile = Tile<TileType::Vec, T, Rows, SrcCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, dstRows, DstCols, BLayout::RowMajor, -1, -1>;
    using IndexTile = Tile<TileType::Vec, uint32_t, Rows, indexCols, BLayout::RowMajor, -1, -1>;
    using OutputTile = Tile<TileType::Vec, T, 1, outputElements>;
    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, ValidRows, ValidCols>, pto::Stride<1, 1, 1, SrcCols, 1>>;
    using IndexGlobal =
        GlobalTensor<uint32_t, Shape<1, 1, 1, ValidRows, ValidCols>, pto::Stride<1, 1, 1, indexCols, 1>>;
    using OutputGlobal = GlobalTensor<T, Shape<1, 1, 1, 1, outputElements>, pto::Stride<1, 1, 1, outputElements, 1>>;
    SrcTile src(ValidRows, ValidCols);
    DstTile dst(dstValidRows, Operation == 1 ? DstCols : dstValidCols);
    IndexTile index(ValidRows, ValidCols);
    IndexTile tmp(ValidRows, ValidCols);
    OutputTile entireOutput;
    TASSIGN(src, 0);
    TASSIGN(index, 16384);
    TASSIGN(tmp, 24576);
    TASSIGN(dst, 32768);
    TASSIGN(entireOutput, 32768);
    SrcGlobal srcGlobal(input);
    IndexGlobal indexGlobal(indices);
    OutputGlobal outputGlobal(out);
    TLOAD(src, srcGlobal);
    TLOAD(entireOutput, outputGlobal);
    if constexpr (Operation == 0 || Operation == 2) {
        TLOAD(index, indexGlobal);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (Operation == 0) {
        TGATHER(dst, src, index, tmp);
    } else if constexpr (Operation == 1) {
        TGATHER<DstTile, SrcTile, Pattern>(dst, src);
    } else if constexpr (Operation == 2) {
        TSCATTER(dst, src, index);
    } else if constexpr (Operation == 3) {
        TSCATTER<Pattern, ScatterAxis::SCATTER_ROW>(dst, src);
    } else {
        TSCATTER<Pattern, ScatterAxis::SCATTER_COL>(dst, src);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    // Read the physical destination, its row padding, and the following allocation.
    TSTORE(outputGlobal, entireOutput);
}

template <
    typename T, int Operation, int Rows, int SrcCols, int ValidRows, int ValidCols, int DstCols,
    pto::MaskPattern Pattern = pto::MaskPattern::P1010>
void LaunchInt64GatherScatterGuard(T* out, T* input, uint32_t* indices, void* stream)
{
    RunInt64GatherScatterGuard<T, Operation, Rows, SrcCols, ValidRows, ValidCols, DstCols, Pattern>
        <<<1, nullptr, stream>>>(out, input, indices);
}

#endif
