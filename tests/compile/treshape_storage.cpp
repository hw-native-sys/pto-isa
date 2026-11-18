/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile from the repository root with the A5 device compiler:
// bisheng -xcce --cce-aicore-arch=dav-c310-vec --cce-aicore-only -DREGISTER_BASE -std=c++17 -I include
//         -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note tests/compile/treshape_storage.cpp
#include <pto/pto-inst.hpp>

using namespace pto;

using Packed = Tile<TileType::Vec, float4_e2m1x2_t, 1, 64>;
using Bytes32 = Tile<TileType::Vec, uint8_t, 1, 32>;
using Bytes64 = Tile<TileType::Vec, uint8_t, 1, 64>;
using NZ = Tile<TileType::Vec, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
using PaddedNZ = Tile<
    TileType::Vec, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;

AICORE void checkPackedViews(Packed& src, Bytes32& exact, Bytes64& larger)
{
    TRESHAPE(exact, src);
    TRESHAPE(larger, src); // expected-note {{in instantiation}}
    // expected-error@pto/npu/a2a3/TReshape.hpp:* {{TRESHAPE: Total byte size must match.}}
}

AICORE void checkPanelPadding(NZ& src, PaddedNZ& larger)
{
    TRESHAPE(larger, src); // expected-note {{in instantiation}}
    // expected-error@pto/npu/a2a3/TReshape.hpp:* {{TRESHAPE: Total byte size must match.}}
}
