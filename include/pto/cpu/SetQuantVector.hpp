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
    uint64_t fpAddres = reinterpret_cast<uint64_t>(fp);
    uint8_t* quantSrc = reinterpret_cast<uint8_t*>(&fpAddres);
    uint8_t* reg_base = reinterpret_cast<uint8_t*>(NPUMemoryModel::Instance().GetREGBase());
    uint8_t* scalarReg = reg_base + QUANT_VECTOR_REG_OFFSET * sizeof(uint64_t);
    std::copy(quantSrc, quantSrc + sizeof(uint64_t), scalarReg);
}

template <typename FpTileData>
PTO_INTERNAL void SET_QUANT_VECTOR_IMPL(FpTileData& fpTile)
{
    static_assert(FpTileData::Loc == TileType::Scaling, "Fix: SET_QUANT_VECTOR only supports Scaling input tile type.");
    SET_QUANT_VECTOR<FpTileData>(fpTile.data());
}

} // namespace pto

#endif // SET_QUANT_VECTOR_HPP
