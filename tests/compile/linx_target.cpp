/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Public headers must work on either side of the umbrella include.
#include <pto/common/type.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>

using namespace pto;
static_assert(CPU_SIM_TARGET == CpuSimTarget::Linx);
static_assert(static_cast<int>(CmpMode::LE) == 3 && static_cast<int>(CmpMode::GT) == 4);
static_assert(cmp_mode_code(CmpMode::GT) == 3 && cmp_mode_code(CmpMode::LE) == 4);

void check(bool value)
{
    if (!value) {
        std::cerr << "Linx CPU-sim numerical regression failed\n";
        std::abort();
    }
}

template <BLayout Layout>
void testVector()
{
    // Asymmetric active region, padded tile and non-unit GM column stride.
    using T = Tile<TileType::Vec, float, 4, 8, Layout, 3, 5>;
    using M = Tile<TileType::Vec, uint8_t, 4, 8, Layout, 3, 5>;
    using G = GlobalTensor<float, Shape<1, 1, 1, 3, 5>, Stride<48, 48, 48, 16, 2>>;
    float a[48], b[48], out[48];
    for (int i = 0; i < 48; ++i) {
        a[i] = -123;
        b[i] = -456;
        out[i] = -999;
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 5; ++c) {
            a[r * 16 + c * 2] = r * 5 + c + 1;
            b[r * 16 + c * 2] = 8;
        }
    G ga(a), gb(b), go(out);
    T ta, tb, dst, tmp;
    M mask;
    for (auto& x : dst.data())
        x = -777;
    TLOAD(ta, ga);
    TLOAD(tb, gb);
    static_assert(std::is_same_v<decltype(TADD(dst, ta, tb)), RecordEvent>);
    TADD(dst, ta, tb);
    TMUL(dst, dst, tb);
    TSUB(dst, dst, ta);
    TSTORE(go, dst);
    for (int i = 0; i < 48; ++i) {
        bool active = i / 16 < 3 && i % 16 < 10 && i % 2 == 0;
        check(out[i] == (active ? a[i] * 7 + 64 : -999));
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c)
            if (r >= 3 || c >= 5)
                check(dst.data()[Layout == BLayout::RowMajor ? r * 8 + c : c * 4 + r] == -777);
    for (auto mode : {CmpMode::EQ, CmpMode::NE, CmpMode::LT, CmpMode::GT, CmpMode::LE, CmpMode::GE}) {
        TLOAD(ta, ga);
        TCMP(mask, ta, tb, mode);
        // Aliasing true source exercises the native TSEL adapter's snapshot.
        TSEL(ta, mask, ta, tb, tmp);
        TSTORE(go, ta);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 5; ++c) {
                int i = r * 16 + c * 2;
                bool pred = mode == CmpMode::EQ ? a[i] == 8 :
                            mode == CmpMode::NE ? a[i] != 8 :
                            mode == CmpMode::LT ? a[i] < 8 :
                            mode == CmpMode::GT ? a[i] > 8 :
                            mode == CmpMode::LE ? a[i] <= 8 :
                                                  a[i] >= 8;
                check(out[i] == (pred ? a[i] : 8));
            }
    }
    TLOAD(ta, ga);
    TDIV(dst, ta, tb);
    TSQRT(dst, dst);
    TSTORE(go, dst);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 5; ++c) {
            int i = r * 16 + c * 2;
            check(std::abs(out[i] - std::sqrt(a[i] / 8)) < 1e-5f);
        }
}

void testUnaryAndLogical()
{
    using F = Tile<TileType::Vec, float, 2, 3>;
    F a, b, dst;
    for (int i = 0; i < 6; ++i) {
        a.data()[i] = i - 3;
        b.data()[i] = 1;
    }
    TABS(dst, a);
    for (int i = 0; i < 6; ++i)
        check(dst.data()[i] == std::abs(i - 3));
    TMAX(dst, a, b);
    for (int i = 0; i < 6; ++i)
        check(dst.data()[i] == std::max(i - 3, 1));
    TMIN(dst, a, b);
    for (int i = 0; i < 6; ++i)
        check(dst.data()[i] == std::min(i - 3, 1));
    TEXP(dst, a);
    for (int i = 0; i < 6; ++i)
        check(std::abs(dst.data()[i] - std::exp(float(i - 3))) < 1e-5f);
    using I = Tile<TileType::Vec, int32_t, 2, 3>;
    I ia, ib, out;
    for (int i = 0; i < 6; ++i) {
        ia.data()[i] = i + 1;
        ib.data()[i] = 10;
    }
    TAND(out, ia, ib);
    for (int i = 0; i < 6; ++i)
        check(out.data()[i] == ((i + 1) & 10));
    TOR(out, ia, ib);
    for (int i = 0; i < 6; ++i)
        check(out.data()[i] == ((i + 1) | 10));
}

void testMatrix()
{
    using A = Tile<TileType::Left, float, 3, 4, BLayout::ColMajor>;
    using B = Tile<TileType::Right, float, 4, 5>;
    using C = Tile<TileType::Acc, float, 3, 5>;
    A a;
    B b;
    C c;
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 4; ++k)
            a.data()[k * 3 + r] = r + k + 1;
    for (int k = 0; k < 4; ++k)
        for (int n = 0; n < 5; ++n)
            b.data()[k * 5 + n] = 2 * k - n;
    TMATMUL(c, a, b);
    // sum_k (r+k+1)*(2*k-n) = 12*r + 40 - n*(4*r+10)
    for (int r = 0; r < 3; ++r)
        for (int n = 0; n < 5; ++n)
            check(c.data()[r * 5 + n] == 12 * r + 40 - n * (4 * r + 10));
    Tile<TileType::Vec, float, 5, 3, BLayout::ColMajor> trans, tmp;
    TTRANS(trans, c, tmp);
    for (int r = 0; r < 5; ++r)
        for (int n = 0; n < 3; ++n)
            check(trans.data()[n * 5 + r] == c.data()[n * 5 + r]);
}

int main()
{
    testVector<BLayout::RowMajor>();
    testVector<BLayout::ColMajor>();
    testUnaryAndLogical();
    testMatrix();
    std::cout << "Linx CPU-sim numerical tests passed\n";
}
