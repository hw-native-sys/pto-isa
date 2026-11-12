/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <cmath>
#include <cstdint>
using namespace pto;
template <typename T, int R = 4, int C = 32>
using V = Tile<Location::Vec, T, R, C>;
template <typename T, int R = 4, int C = 32>
using M = global_tensor<T, RowMajor<R, C>>;
void k_TADD(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TADD(d, a, b);
    TSTORE(md, d);
}
void k_TSUB(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TSUB(d, a, b);
    TSTORE(md, d);
}
void k_TMUL(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TMUL(d, a, b);
    TSTORE(md, d);
}
void k_TDIV(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TDIV(d, a, b);
    TSTORE(md, d);
}
void k_TREM(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TREM(d, a, b);
    TSTORE(md, d);
}
void k_TMAX(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TMAX(d, a, b);
    TSTORE(md, d);
}
void k_TMIN(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TMIN(d, a, b);
    TSTORE(md, d);
}
void k_TEXPDIF(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TEXPDIF(d, a, b);
    TSTORE(md, d);
}
void k_TABS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TABS(d, a);
    TSTORE(md, d);
}
void k_TNEG(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TNEG(d, a);
    TSTORE(md, d);
}
void k_TEXP(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TEXP(d, a);
    TSTORE(md, d);
}
void k_TLOG(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TLOG(d, a);
    TSTORE(md, d);
}
void k_TRECIP(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TRECIP(d, a);
    TSTORE(md, d);
}
void k_TSQRT(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TSQRT(d, a);
    TSTORE(md, d);
}
void k_TRSQRT(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TRSQRT(d, a);
    TSTORE(md, d);
}
void k_TRELU(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TRELU(d, a);
    TSTORE(md, d);
}
void k_TADDS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TADDS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TSUBS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TSUBS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TMULS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TMULS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TDIVS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TDIVS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TREMS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TREMS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TMAXS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TMAXS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TMINS(float* x, float* z)
{
    V<float> a, d;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TMINS(d, a, 2.f);
    TSTORE(md, d);
}
void k_TAND(uint32_t* x, uint32_t* y, uint32_t* z)
{
    V<uint32_t> a, b, d;
    M<uint32_t> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TAND(d, a, b);
    TSTORE(md, d);
}
void k_TOR(uint32_t* x, uint32_t* y, uint32_t* z)
{
    V<uint32_t> a, b, d;
    M<uint32_t> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TOR(d, a, b);
    TSTORE(md, d);
}
void k_TXOR(uint32_t* x, uint32_t* y, uint32_t* z)
{
    V<uint32_t> a, b, d;
    M<uint32_t> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TXOR(d, a, b);
    TSTORE(md, d);
}
void k_TSHL(uint32_t* x, uint32_t* y, uint32_t* z)
{
    V<uint32_t> a, b, d;
    M<uint32_t> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TSHL(d, a, b);
    TSTORE(md, d);
}
void k_TSHR(uint32_t* x, uint32_t* y, uint32_t* z)
{
    V<uint32_t> a, b, d;
    M<uint32_t> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TSHR(d, a, b);
    TSTORE(md, d);
}
void k_TROWSUM(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<float, 4, 1> md(z);
    TLOAD(a, ma);
    TROWSUM(d, a);
    TSTORE(md, d);
}
void k_TROWPROD(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<float, 4, 1> md(z);
    TLOAD(a, ma);
    TROWPROD(d, a);
    TSTORE(md, d);
}
void k_TROWMAX(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<float, 4, 1> md(z);
    TLOAD(a, ma);
    TROWMAX(d, a);
    TSTORE(md, d);
}
void k_TROWMIN(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<float, 4, 1> md(z);
    TLOAD(a, ma);
    TROWMIN(d, a);
    TSTORE(md, d);
}
void k_TCOLSUM(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<float, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLSUM(d, a);
    TSTORE(md, d);
}
void k_TCOLPROD(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<float, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLPROD(d, a);
    TSTORE(md, d);
}
void k_TCOLMAX(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<float, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLMAX(d, a);
    TSTORE(md, d);
}
void k_TCOLMIN(float* x, float* z)
{
    V<float> a;
    Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<float, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLMIN(d, a);
    TSTORE(md, d);
}
void k_TROWARGMAX(float* x, uint32_t* z)
{
    V<float> a;
    Tile<Location::Vec, uint32_t, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<uint32_t, 4, 1> md(z);
    TLOAD(a, ma);
    TROWARGMAX(d, a);
    TSTORE(md, d);
}
void k_TROWARGMIN(float* x, uint32_t* z)
{
    V<float> a;
    Tile<Location::Vec, uint32_t, 4, 32, BLayout::RowMajor, 4, 1> d;
    M<float> ma(x);
    M<uint32_t, 4, 1> md(z);
    TLOAD(a, ma);
    TROWARGMIN(d, a);
    TSTORE(md, d);
}
void k_TCOLARGMAX(float* x, uint32_t* z)
{
    V<float> a;
    Tile<Location::Vec, uint32_t, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<uint32_t, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLARGMAX(d, a);
    TSTORE(md, d);
}
void k_TCOLARGMIN(float* x, uint32_t* z)
{
    V<float> a;
    Tile<Location::Vec, uint32_t, 4, 32, BLayout::RowMajor, 1, 32> d;
    M<float> ma(x);
    M<uint32_t, 1, 32> md(z);
    TLOAD(a, ma);
    TCOLARGMIN(d, a);
    TSTORE(md, d);
}
void k_broadcast(float* x, float* z)
{
    V<float> a, d;
    V<float, 4, 1> row;
    V<float, 1, 32> col;
    M<float> ma(x), md(z);
    TLOAD(a, ma);
    TROWSUM(row, a);
    TROWEXPANDADD(d, a, row);
    TSTORE(md, d);
    TCOLSUM(col, a);
    TCOLEXPANDMUL(d, a, col);
    M<float> out(z + 128);
    TSTORE(out, d);
}
void k_fma(float* x, float* y, float* z)
{
    V<float> a, b, d;
    M<float> ma(x), mb(y), md(z);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TFMA(d, a, b, a);
    TSTORE(md, d);
}
void k_regions(int32_t* x, int32_t* y, int32_t* z)
{
    using P = VecTileM32<int32_t, 32, 8>;
    using F = VecTileM32<int32_t, 32, 1>;
    P parent;
    F rhs;
    M<int32_t, 32, 8> mx(x), mz(z);
    M<int32_t, 32, 1> my(y);
    TLOAD(parent, mx);
    TLOAD(rhs, my);
    auto views = TPARTVIEW<F, 1, 8>(parent);
    TileArray<F, 1, 8> result;
    for (unsigned i = 0; i < 8; ++i) {
        auto view = views[0][i];
        TADD(result[0][i], view, rhs);
    }
    auto out = TASSEMBLY<P>(std::move(result));
    TSTORE(mz, out);
}
void k_broadcast_ops(unsigned operation, bool column, float* x, float* values, float* output)
{
    V<float> a, d;
    V<float, 4, 1> row;
    V<float, 1, 32> col;
    M<float> in(x), out(output);
    M<float, 4, 1> mr(values);
    M<float, 1, 32> mc(values);
    TLOAD(a, in);
    if (column) {
        TLOAD(col, mc);
        switch (operation) {
            case 0:
                TCOLEXPAND(d, col);
                break;
            case 1:
                TCOLEXPANDADD(d, a, col);
                break;
            case 2:
                TCOLEXPANDSUB(d, a, col);
                break;
            case 3:
                TCOLEXPANDMUL(d, a, col);
                break;
            case 4:
                TCOLEXPANDDIV(d, a, col);
                break;
            case 5:
                TCOLEXPANDMAX(d, a, col);
                break;
            case 6:
                TCOLEXPANDMIN(d, a, col);
                break;
            default:
                TCOLEXPANDEXPDIF(d, a, col);
                break;
        }
    } else {
        TLOAD(row, mr);
        switch (operation) {
            case 0:
                TROWEXPAND(d, row);
                break;
            case 1:
                TROWEXPANDADD(d, a, row);
                break;
            case 2:
                TROWEXPANDSUB(d, a, row);
                break;
            case 3:
                TROWEXPANDMUL(d, a, row);
                break;
            case 4:
                TROWEXPANDDIV(d, a, row);
                break;
            case 5:
                TROWEXPANDMAX(d, a, row);
                break;
            case 6:
                TROWEXPANDMIN(d, a, row);
                break;
            default:
                TROWEXPANDEXPDIF(d, a, row);
                break;
        }
    }
    TSTORE(out, d);
}

void k_compare_move(float* x, float* y, float* output)
{
    V<float> a, b, d;
    V<float> predicate;
    M<float> mx(x), my(y), out(output);
    TPREFETCH(mx, 32, 4);
    TLOAD(a, mx);
    TLOAD(b, my);
    TMOV(d, a);
    TCMP<CmpMode::GE>(predicate, a, b);
    TSEL(d, predicate, b);
    TSTORE(out, d);
}

#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
int main()
{
    std::array<float, 256> x{}, y{}, z{};
    std::array<uint32_t, 128> ux{}, uy{}, uz{};
    for (unsigned i = 0; i < 128; ++i) {
        x[i] = float((i * 7) % 16 + 1) / 4.f;
        y[i] = float(i % 8 + 1) / 4.f;
        ux[i] = 0xdead0000 + i;
        uy[i] = i % 37;
    }
    k_TADD(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (a + b), "TADD");
    }
    k_TSUB(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (a - b), "TSUB");
    }
    k_TMUL(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (a * b), "TMUL");
    }
    k_TDIV(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (a / b), "TDIV");
    }
    k_TREM(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (std::fmod(a, b)), "TREM");
    }
    k_TMAX(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (std::fmax(a, b)), "TMAX");
    }
    k_TMIN(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (std::fmin(a, b)), "TMIN");
    }
    k_TEXPDIF(x.data(), y.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i], b = y[i];
        check(z[i] == (float(std::exp(double(a) - b))), "TEXPDIF");
    }
    k_TABS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::fabs(a)), "TABS");
    }
    k_TNEG(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (-a), "TNEG");
    }
    k_TEXP(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (float(std::exp(double(a)))), "TEXP");
    }
    k_TLOG(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == std::log2(a) * 0x1.62e430p-1f, "TLOG rounds log2 before ln(2) multiply");
    }
    k_TRECIP(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (1.f / a), "TRECIP");
    }
    k_TSQRT(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::sqrt(a)), "TSQRT");
    }
    k_TRSQRT(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == 1.f / std::sqrt(a), "TRSQRT rounds sqrt before division");
    }
    k_TRELU(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::fmax(a, 0.f)), "TRELU");
    }
    k_TADDS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (a + 2.f), "TADDS");
    }
    k_TSUBS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (a - 2.f), "TSUBS");
    }
    k_TMULS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (a * 2.f), "TMULS");
    }
    k_TDIVS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (a / 2.f), "TDIVS");
    }
    k_TREMS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::fmod(a, 2.f)), "TREMS");
    }
    k_TMAXS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::fmax(a, 2.f)), "TMAXS");
    }
    k_TMINS(x.data(), z.data());
    for (int i = 0; i < 128; ++i) {
        float a = x[i];
        check(z[i] == (std::fmin(a, 2.f)), "TMINS");
    }
    k_TAND(ux.data(), uy.data(), uz.data());
    for (int i = 0; i < 128; ++i) {
        uint32_t a = ux[i], b = uy[i];
        check(uz[i] == (a & b), "TAND");
    }
    k_TOR(ux.data(), uy.data(), uz.data());
    for (int i = 0; i < 128; ++i) {
        uint32_t a = ux[i], b = uy[i];
        check(uz[i] == (a | b), "TOR");
    }
    k_TXOR(ux.data(), uy.data(), uz.data());
    for (int i = 0; i < 128; ++i) {
        uint32_t a = ux[i], b = uy[i];
        check(uz[i] == (a ^ b), "TXOR");
    }
    k_TSHL(ux.data(), uy.data(), uz.data());
    for (int i = 0; i < 128; ++i) {
        uint32_t a = ux[i], b = uy[i];
        check(uz[i] == (a << (b % 32)), "TSHL");
    }
    k_TSHR(ux.data(), uy.data(), uz.data());
    for (int i = 0; i < 128; ++i) {
        uint32_t a = ux[i], b = uy[i];
        check(uz[i] == (a >> (b % 32)), "TSHR");
    }
    k_TROWSUM(x.data(), z.data());
    for (int i = 0; i < 4; ++i) {
        float v = 0.f;
        for (int j = 0; j < 32; ++j) {
            float a = x[i * 32 + j];
            v += a;
        }
        check(z[i] == v, "TROWSUM");
    }
    k_TROWPROD(x.data(), z.data());
    for (int i = 0; i < 4; ++i) {
        float v = 1.f;
        for (int j = 0; j < 32; ++j) {
            float a = x[i * 32 + j];
            v *= a;
        }
        check(z[i] == v, "TROWPROD");
    }
    k_TROWMAX(x.data(), z.data());
    for (int i = 0; i < 4; ++i) {
        float v = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 32; ++j) {
            float a = x[i * 32 + j];
            v = std::fmax(v, a);
        }
        check(z[i] == v, "TROWMAX");
    }
    k_TROWMIN(x.data(), z.data());
    for (int i = 0; i < 4; ++i) {
        float v = std::numeric_limits<float>::infinity();
        for (int j = 0; j < 32; ++j) {
            float a = x[i * 32 + j];
            v = std::fmin(v, a);
        }
        check(z[i] == v, "TROWMIN");
    }
    k_TCOLSUM(x.data(), z.data());
    for (int i = 0; i < 32; ++i) {
        float v = 0.f;
        for (int j = 0; j < 4; ++j) {
            float a = x[j * 32 + i];
            v += a;
        }
        check(z[i] == v, "TCOLSUM");
    }
    k_TCOLPROD(x.data(), z.data());
    for (int i = 0; i < 32; ++i) {
        float v = 1.f;
        for (int j = 0; j < 4; ++j) {
            float a = x[j * 32 + i];
            v *= a;
        }
        check(z[i] == v, "TCOLPROD");
    }
    k_TCOLMAX(x.data(), z.data());
    for (int i = 0; i < 32; ++i) {
        float v = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float a = x[j * 32 + i];
            v = std::fmax(v, a);
        }
        check(z[i] == v, "TCOLMAX");
    }
    k_TCOLMIN(x.data(), z.data());
    for (int i = 0; i < 32; ++i) {
        float v = std::numeric_limits<float>::infinity();
        for (int j = 0; j < 4; ++j) {
            float a = x[j * 32 + i];
            v = std::fmin(v, a);
        }
        check(z[i] == v, "TCOLMIN");
    }
    k_TROWARGMAX(x.data(), uz.data());
    for (int i = 0; i < 4; ++i) {
        unsigned best = 0;
        for (int j = 1; j < 32; ++j)
            if (x[i * 32 + j] > x[i * 32 + best])
                best = j;
        check(uz[i] == best, "TROWARGMAX");
    }
    k_TROWARGMIN(x.data(), uz.data());
    for (int i = 0; i < 4; ++i) {
        unsigned best = 0;
        for (int j = 1; j < 32; ++j)
            if (x[i * 32 + j] < x[i * 32 + best])
                best = j;
        check(uz[i] == best, "TROWARGMIN");
    }
    k_TCOLARGMAX(x.data(), uz.data());
    for (int i = 0; i < 32; ++i) {
        unsigned best = 0;
        for (int j = 1; j < 4; ++j)
            if (x[j * 32 + i] > x[best * 32 + i])
                best = j;
        check(uz[i] == best, "TCOLARGMAX");
    }
    k_TCOLARGMIN(x.data(), uz.data());
    for (int i = 0; i < 32; ++i) {
        unsigned best = 0;
        for (int j = 1; j < 4; ++j)
            if (x[j * 32 + i] < x[best * 32 + i])
                best = j;
        check(uz[i] == best, "TCOLARGMIN");
    }
    k_broadcast(x.data(), z.data());
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned c = 0; c < 32; ++c) {
            float sum = 0;
            for (unsigned j = 0; j < 32; ++j)
                sum += x[r * 32 + j];
            check(z[r * 32 + c] == x[r * 32 + c] + sum, "row broadcast");
            sum = 0;
            for (unsigned i = 0; i < 4; ++i)
                sum += x[i * 32 + c];
            check(z[128 + r * 32 + c] == x[r * 32 + c] * sum, "column broadcast");
        }
    k_fma(x.data(), y.data(), z.data());
    for (unsigned i = 0; i < 128; ++i)
        check(z[i] == std::fma(x[i], y[i], x[i]), "TFMA");
    std::array<int32_t, 256> left{}, right{}, out{};
    for (unsigned i = 0; i < 256; ++i)
        left[i] = i;
    for (unsigned i = 0; i < 32; ++i)
        right[i] = -int(i);
    k_regions(left.data(), right.data(), out.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 8; ++c)
            check(out[r * 8 + c] == int(r * 8 + c) - int(r), "TileArray/subview output assembly");
    for (unsigned column = 0; column < 2; ++column)
        for (unsigned op = 0; op < 8; ++op) {
            k_broadcast_ops(op, column, x.data(), y.data(), z.data());
            for (unsigned i = 0; i < 128; ++i) {
                const float a = x[i], b = y[column ? i % 32 : i / 32];
                const float expected[] = {b,     a + b,           a - b,           a * b,
                                          a / b, std::fmax(a, b), std::fmin(a, b), float(std::exp(double(a) - b))};
                check(z[i] == expected[op], "broadcast operation");
            }
        }
    k_compare_move(x.data(), y.data(), z.data());
    for (unsigned i = 0; i < 128; ++i)
        check(z[i] == std::fmin(x[i], y[i]), "TCMP/TMOV/TSEL");
    std::cout << "Linx vector and region execution checks passed\n";
}
#endif
