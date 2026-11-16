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
#include <algorithm>
#include <cmath>
#include <limits>
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

// Check each operation independently so another operation cannot hide a missed
// write. Inspect the physical padding as well as every active element.
template <BLayout Layout, int Rows, int Cols, int ValidRows = Rows, int ValidCols = Cols>
void testElementwise()
{
    using F = Tile<TileType::Vec, float, Rows, Cols, Layout, ValidRows, ValidCols>;
    using I = Tile<TileType::Vec, int32_t, Rows, Cols, Layout, ValidRows, ValidCols>;
    using M = Tile<TileType::Vec, uint8_t, Rows, Cols, Layout, ValidRows, ValidCols>;
    auto offset = [](int r, int c) { return Layout == BLayout::RowMajor ? r * Cols + c : c * Rows + r; };
    F a, b, dst, tmp;
    I ia, ib, out;
    M mask;
    for (int r = 0; r < Rows; ++r)
        for (int c = 0; c < Cols; ++c) {
            int i = offset(r, c);
            a.data()[i] = float((r * Cols + c) % 7 - 3);
            b.data()[i] = 2;
            ia.data()[i] = r * Cols + c + 1;
            ib.data()[i] = 10;
        }
    auto verify = [&](auto& result, auto operation, auto expected) {
        for (auto& x : result.data())
            x = 99;
        operation();
        for (int r = 0; r < Rows; ++r)
            for (int c = 0; c < Cols; ++c) {
                int i = offset(r, c);
                double value = r < ValidRows && c < ValidCols ? expected(i) : 99;
                check(std::abs(double(result.data()[i]) - value) <= 1e-6 * std::max(1.0, std::abs(value)));
            }
    };
    verify(dst, [&] { TADD(dst, a, b); }, [&](int i) { return a.data()[i] + b.data()[i]; });
    verify(dst, [&] { TSUB(dst, a, b); }, [&](int i) { return a.data()[i] - b.data()[i]; });
    verify(dst, [&] { TMUL(dst, a, b); }, [&](int i) { return a.data()[i] * b.data()[i]; });
    verify(dst, [&] { TDIV(dst, a, b); }, [&](int i) { return a.data()[i] / b.data()[i]; });
    verify(dst, [&] { TMAX(dst, a, b); }, [&](int i) { return std::max(a.data()[i], b.data()[i]); });
    verify(dst, [&] { TMIN(dst, a, b); }, [&](int i) { return std::min(a.data()[i], b.data()[i]); });
    verify(dst, [&] { TABS(dst, a); }, [&](int i) { return std::abs(a.data()[i]); });
    verify(dst, [&] { TEXP(dst, a); }, [&](int i) { return std::exp(a.data()[i]); });
    verify(out, [&] { TAND(out, ia, ib); }, [&](int i) { return ia.data()[i] & ib.data()[i]; });
    verify(out, [&] { TOR(out, ia, ib); }, [&](int i) { return ia.data()[i] | ib.data()[i]; });
    for (auto mode : {CmpMode::EQ, CmpMode::NE, CmpMode::LT, CmpMode::GT, CmpMode::LE, CmpMode::GE}) {
        auto predicate = [&](int i) {
            auto x = a.data()[i], y = b.data()[i];
            return mode == CmpMode::EQ ? x == y :
                   mode == CmpMode::NE ? x != y :
                   mode == CmpMode::LT ? x < y :
                   mode == CmpMode::GT ? x > y :
                   mode == CmpMode::LE ? x <= y :
                                         x >= y;
        };
        verify(mask, [&] { TCMP(mask, a, b, mode); }, predicate);
        verify(
            dst, [&] { TSEL(dst, mask, a, b, tmp); }, [&](int i) { return predicate(i) ? a.data()[i] : b.data()[i]; });
    }
    for (auto& x : a.data())
        x = std::abs(x) + 0.25f;
    verify(dst, [&] { TSQRT(dst, a); }, [&](int i) { return std::sqrt(a.data()[i]); });
}

template <BLayout Layout>
void testDoubleSqrt()
{
    using T = Tile<TileType::Vec, double, 2, 4, Layout>;
    const double values[] = {1e300, 1e-300,
                             2.0,   0.0,
                             -0.0,  std::numeric_limits<double>::infinity(),
                             -1.0,  std::numeric_limits<double>::quiet_NaN()};
    T src, dst;
    for (int i = 0; i < 8; ++i)
        src.data()[i] = values[i];
    TSQRT(dst, src);
    for (int i = 0; i < 8; ++i) {
        double expected = std::sqrt(values[i]), actual = dst.data()[i];
        if (std::isnan(expected))
            check(std::isnan(actual));
        else if (std::isinf(expected) || expected == 0)
            check(actual == expected && std::signbit(actual) == std::signbit(expected));
        else
            check(std::isfinite(actual) && std::abs(actual / expected - 1) < 1e-15);
    }
}

template <BLayout Layout>
void testDynamicStride()
{
    using T = Tile<TileType::Vec, float, 4, 8, Layout, 3, 5>;
    using S = Stride<1, 1, 1, DYNAMIC, DYNAMIC>;
    using G = GlobalTensor<float, Shape<1, 1, 1, 3, 5>, S>;
    for (int colStride : {1, 2, 3}) {
        // Different source/destination pitches exercise both transfer directions.
        float input[60], output[60];
        for (int i = 0; i < 60; ++i) {
            input[i] = float(i + 1);
            output[i] = -999;
        }
        G src(input, {}, S(19, colStride)), dst(output, {}, S(20, colStride));
        T tile;
        for (auto& x : tile.data())
            x = -777;
        TLOAD(tile, src);
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c)
                check(
                    tile.data()[Layout == BLayout::RowMajor ? r * 8 + c : c * 4 + r] ==
                    (r < 3 && c < 5 ? input[r * 19 + c * colStride] : -777));
        TSTORE(dst, tile);
        for (int i = 0; i < 60; ++i) {
            int r = i / 20, c = (i % 20) / colStride;
            bool active = i % 20 % colStride == 0 && c < 5;
            check(output[i] == (active ? input[r * 19 + c * colStride] : -999));
        }
    }
}

template <typename DType, BLayout Layout>
void testIntegerAbs()
{
    using T = Tile<TileType::Vec, DType, 4, 8, Layout, 3, 5>;
    constexpr DType min = std::numeric_limits<DType>::min(), max = std::numeric_limits<DType>::max();
    const DType values[] = {min, DType(min + 1), -1, 0, 1, max};
    const DType expected[] = {min, max, 1, 0, 1, max};
    auto offset = [](int r, int c) { return Layout == BLayout::RowMajor ? r * 8 + c : c * 4 + r; };
    T src, dst;
    for (auto& x : src.data())
        x = -7;
    for (auto& x : dst.data())
        x = 99;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 5; ++c)
            src.data()[offset(r, c)] = values[(r * 5 + c) % 6];
    auto verify = [&](const T& result, DType padding) {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c)
                check(result.data()[offset(r, c)] == (r < 3 && c < 5 ? expected[(r * 5 + c) % 6] : padding));
    };
    TABS(dst, src);
    verify(dst, 99);
    TABS(src, src);
    verify(src, -7);
}

template <typename DType, BLayout Layout>
void testExpTypes()
{
    using T = Tile<TileType::Vec, DType, 4, 8, Layout, 3, 5>;
    const double values[] = {
        -700,
        -100,
        -20,
        -1,
        -0.25,
        0,
        0.25,
        1,
        12,
        20,
        100,
        700,
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()};
    auto offset = [](int r, int c) { return Layout == BLayout::RowMajor ? r * 8 + c : c * 4 + r; };
    T src, dst;
    for (auto& x : src.data())
        x = 0;
    for (auto& x : dst.data())
        x = 99;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 5; ++c)
            src.data()[offset(r, c)] = static_cast<DType>(values[r * 5 + c]);
    TEXP(dst, src);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c) {
            int i = offset(r, c);
            if (r >= 3 || c >= 5) {
                check(dst.data()[i] == DType(99));
                continue;
            }
            double expected = static_cast<DType>(std::exp(static_cast<double>(src.data()[i])));
            double actual = static_cast<double>(dst.data()[i]);
            if (std::isnan(expected))
                check(std::isnan(actual));
            else if (std::isinf(expected) || expected == 0)
                check(actual == expected);
            else {
                constexpr double tolerance = std::is_same_v<DType, double> ? 1e-14 :
                                             std::is_same_v<DType, float>  ? 1e-6 :
                                                                             0;
                check(std::isfinite(actual) && std::abs(actual / expected - 1) <= tolerance);
            }
        }
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
    testElementwise<BLayout::RowMajor, 2, 3>();
    testElementwise<BLayout::ColMajor, 2, 3>();
    testElementwise<BLayout::RowMajor, 8, 4>();
    testElementwise<BLayout::ColMajor, 8, 4>();
    testElementwise<BLayout::RowMajor, 4, 8, 3, 5>();
    testElementwise<BLayout::ColMajor, 4, 8, 3, 5>();
    testElementwise<BLayout::RowMajor, 8, 4, 5, 3>();
    testElementwise<BLayout::ColMajor, 8, 4, 5, 3>();
    testDoubleSqrt<BLayout::RowMajor>();
    testDoubleSqrt<BLayout::ColMajor>();
    testIntegerAbs<int8_t, BLayout::RowMajor>();
    testIntegerAbs<int8_t, BLayout::ColMajor>();
    testIntegerAbs<int16_t, BLayout::RowMajor>();
    testIntegerAbs<int16_t, BLayout::ColMajor>();
    testIntegerAbs<int32_t, BLayout::RowMajor>();
    testIntegerAbs<int32_t, BLayout::ColMajor>();
    testIntegerAbs<int64_t, BLayout::RowMajor>();
    testIntegerAbs<int64_t, BLayout::ColMajor>();
    testExpTypes<__half, BLayout::RowMajor>();
    testExpTypes<__half, BLayout::ColMajor>();
    testExpTypes<__bf16, BLayout::RowMajor>();
    testExpTypes<__bf16, BLayout::ColMajor>();
    testExpTypes<float, BLayout::RowMajor>();
    testExpTypes<float, BLayout::ColMajor>();
    testExpTypes<double, BLayout::RowMajor>();
    testExpTypes<double, BLayout::ColMajor>();
    testDynamicStride<BLayout::RowMajor>();
    testDynamicStride<BLayout::ColMajor>();
    testMatrix();
    std::cout << "Linx CPU-sim numerical tests passed\n";
}
