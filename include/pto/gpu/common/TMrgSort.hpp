/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TMRGSORT_HPP
#define PTO_GPU_COMMON_TMRGSORT_HPP

namespace pto {

struct MrgSortExecutedNumList {
};

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &, MrgSortExecutedNumList &, TmpTileData &, Src0TileData &, Src1TileData &)
{
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &, MrgSortExecutedNumList &, TmpTileData &, Src0TileData &, Src1TileData &,
                                Src2TileData &, Src3TileData &)
{
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, typename Src4TileData, typename Src5TileData,
          typename Src6TileData, typename Src7TileData>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &, MrgSortExecutedNumList &, TmpTileData &, Src0TileData &, Src1TileData &,
                                Src2TileData &, Src3TileData &, Src4TileData &, Src5TileData &, Src6TileData &,
                                Src7TileData &)
{
}

} // namespace pto

#endif
