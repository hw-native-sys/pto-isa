/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <pto/pto-inst.hpp>

using namespace pto;

void check(bool value)
{
    if (!value) {
        std::cerr << "Linx imported layout-conversion regression failed\n";
        std::abort();
    }
}

template <int Rows, int Cols, int ValidRows = Rows, int ValidCols = Cols>
void testConversion(bool toNz)
{
    using Col = Tile<TileType::Vec, float, Rows, Cols, BLayout::ColMajor, ValidRows, ValidCols>;
    using Nz = Tile<TileType::Vec, float, Rows, Cols, BLayout::ColMajor, ValidRows, ValidCols, SLayout::RowMajor>;
    static_assert(Nz::InnerRows == 16 && Nz::InnerCols == 8);
    Col col;
    Nz nz;
    constexpr float sentinel = -999;
    auto value = [](int row, int column) { return float(row * Cols + column + 1); };
    for (auto& x : col.data())
        x = sentinel;
    for (auto& x : nz.data())
        x = sentinel;

    // Fill each source independently: a roundtrip could hide matching errors in
    // both conversion directions. NZ stores row-major 16x8 cells column-first.
    for (int r = 0; r < ValidRows; ++r)
        for (int c = 0; c < ValidCols; ++c)
            if (toNz)
                col.data()[c * Rows + r] = value(r, c);
    for (int i = 0; i < Rows * Cols; ++i) {
        int r = (i / 128 % (Rows / 16)) * 16 + i % 128 / 8;
        int c = (i / (Rows * 8)) * 8 + i % 8;
        if (!toNz && r < ValidRows && c < ValidCols)
            nz.data()[i] = value(r, c);
    }

    // This is a regression for shipped imported internals, not public pto::TCVT
    // support. The public CPU adapter still rejects boxed and CUBE layouts.
    if (toNz)
        TCVT_Impl(nz, col);
    else
        TCVT_Impl(col, nz);

    for (int i = 0; i < Rows * Cols; ++i) {
        int r = toNz ? (i / 128 % (Rows / 16)) * 16 + i % 128 / 8 : i % Rows;
        int c = toNz ? (i / (Rows * 8)) * 8 + i % 8 : i / Rows;
        float expected = r < ValidRows && c < ValidCols ? value(r, c) : sentinel;
        check((toNz ? nz.data()[i] : col.data()[i]) == expected);
    }
}

int main(int argc, char** argv)
{
    check(argc == 2);
    std::string_view direction = argv[1];
    check(direction == "to-nz" || direction == "to-col");
    bool toNz = direction == "to-nz";
    testConversion<16, 8>(toNz);
    testConversion<16, 32>(toNz);
    testConversion<32, 16>(toNz);
    testConversion<16, 32, 13, 19>(toNz);
    testConversion<32, 16, 17, 11>(toNz);
    std::cout << "Linx imported " << direction << " conversion tests passed\n";
}
