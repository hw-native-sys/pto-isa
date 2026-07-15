/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SET_QUANT_VECTOR_HPP
#define SET_QUANT_VECTOR_HPP

namespace pto {

template <typename FpTileData>
__tf__ PTO_INTERNAL void SET_QUANT_VECTOR(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType* dstAddrFp = (__fbuf__ typename FpTileData::DType*)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

template <typename FpTileData>
PTO_INTERNAL void SET_QUANT_VECTOR_IMPL(FpTileData& fpTile)
{
    static_assert(FpTileData::Loc == TileType::Scaling, "Fix: SET_QUANT_VECTOR only supports Scaling input tile type.");
    SET_QUANT_VECTOR<FpTileData>(fpTile.data());
}

} // namespace pto

#endif // SET_QUANT_VECTOR_HPP
