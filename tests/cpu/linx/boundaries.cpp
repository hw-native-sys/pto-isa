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
#include <cstdint>
using namespace pto;
template <typename T, int R, int C>
using Mem = global_tensor<T, RowMajor<R, C>>;
void indexedRows(float* input, uint16_t* indices, float* output)
{
    using T = Tile<Location::Vec, float, 4, 32>;
    using I = Tile<Location::Vec, uint16_t, 4, 32>;
    T a, d;
    I ix;
    Mem<float, 4, 32> src(input), out(output);
    Mem<uint16_t, 4, 32> index(indices);
    TLOAD(a, src);
    TLOAD(ix, index);
    TGATHER(d, a, ix);
    TSTORE(out, d);
    TSCATTER(a, a, ix);
    Mem<float, 4, 32> scattered(output + 128);
    TSTORE(scattered, a);
}
void selectAndBits(uint32_t* input, uint32_t* output)
{
    using T = Tile<Location::Vec, uint32_t, 4, 32>;
    T a, b, p;
    Mem<uint32_t, 4, 32> src(input), out(output);
    TLOAD(a, src);
    TCMPS<CmpMode::GE>(p, a, 64u);
    TSELS(b, p, 999u, a);
    TSTORE(out, b);
    TNOT(b, a);
    Mem<uint32_t, 4, 32> negated(output + 128);
    TSTORE(negated, b);
    TANDS(b, a, 0xfu);
    TORS(b, b, 0x100u);
    TXORS(b, b, 0x10u);
    TSHLS(b, b, 2u);
    TSHRS(b, b, 1u);
    Mem<uint32_t, 4, 32> scalar(output + 256);
    TSTORE(scalar, b);
}
void generators(int32_t* output)
{
    using T = Tile<Location::Vec, int32_t, 4, 32>;
    T a;
    Mem<int32_t, 4, 32> out(output);
    Tile<Location::Vec, int32_t, 1, 32> seq;
    TCI<decltype(seq), int32_t, 1>(seq, 100);
    Mem<int32_t, 1, 32> seqOut(output);
    TSTORE(seqOut, seq);
    TTRI(a);
    Mem<int32_t, 4, 32> triangle(output + 128);
    TSTORE(triangle, a);
    CubeTileM32<int32_t, 32, 4> cube;
    TCI_2D<decltype(cube), int32_t, -1, 1>(cube, 7);
    Mem<int32_t, 32, 4> grid(output + 256);
    TSTORE(grid, cube);
}
void dynamic(float* input, float* output, unsigned rows, unsigned columns)
{
    using T = Tile<Location::Vec, float, 4, 32, BLayout::RowMajor, -1, -1>;
    T a(rows, columns), d(rows, columns);
    Mem<float, 4, 32> src(input), out(output);
    TLOAD(a, src);
    TADDS(d, a, 3.f);
    TSTORE(out, d);
}
void padded(float* input, float* output)
{
    using P = VecTileM32<float, 32, 4, 3, 2>;
    P tile;
    Mem<float, 3, 2> src(input);
    TLOAD_CUBE<TmaPadValue::Max>(tile, src);
    // The native carrier is shared by equal-capacity Tile specializations.
    VecTileM32<float, 32, 4> full;
    full.data() = tile.data();
    Mem<float, 32, 4> out(output);
    TSTORE(out, full);
}
void quantized(float* left, float* right, int8_t* output)
{
    CubeTileM32<float, 32, 8> a;
    CubeTileN8<float, 8, 8> b;
    CubeAccumulatorM32<int8_t, 32, 8> d;
    Mem<float, 32, 8> ma(left);
    Mem<float, 8, 8> mb(right);
    Mem<int8_t, 32, 8> md(output);
    TLOAD(a, ma);
    TLOAD(b, mb);
    // FP19 scale 0.5 and signed S9 offset +3.
    TMATMUL(d, a, b, fixp::s8(uint64_t(0x3f000000) | (uint64_t(3) << 37)));
    TSTORE(md, d);
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
template <typename F>
void rejects(F fn, const char* message)
{
    bool failed = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    check(failed, message);
}
int main()
{
    std::array<float, 256> in{}, out{};
    std::array<uint16_t, 128> indices{};
    for (unsigned i = 0; i < 128; ++i) {
        in[i] = i;
        indices[i] = (i / 32 + i % 2) % 3;
    }
    indexedRows(in.data(), indices.data(), out.data());
    std::array<float, 128> expected{};
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned c = 0; c < 32; ++c) {
            check(out[r * 32 + c] == in[indices[r * 32 + c] * 32 + c], "TGATHER indexes rows, retaining columns");
            expected[indices[r * 32 + c] * 32 + c] = in[r * 32 + c];
        }
    for (unsigned i = 0; i < 128; ++i)
        check(out[128 + i] == expected[i], "TSCATTER snapshot, duplicate order, and zero fill");
    std::array<uint32_t, 128> u{};
    std::array<uint32_t, 384> v{};
    for (unsigned i = 0; i < 128; ++i)
        u[i] = i;
    selectAndBits(u.data(), v.data());
    for (unsigned i = 0; i < 128; ++i) {
        check(v[i] == (i >= 64 ? i : 999), "TSELS mask and scalar order");
        check(v[128 + i] == ~i, "TNOT");
        check(v[256 + i] == (((i & 15) | 0x100) ^ 0x10) * 2, "scalar bitwise and shifts");
    }
    std::array<int32_t, 384> g{};
    generators(g.data());
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned c = 0; c < 32; ++c) {
            if (r == 0)
                check(g[c] == 100 - int(c), "descending TCI");
            check(g[128 + r * 32 + c] == int(c <= r), "TTRI");
        }
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 4; ++c)
            check(g[256 + r * 4 + c] == 7 - r + c, "signed TCI_2D increments");
    out.fill(-9);
    dynamic(in.data(), out.data(), 3, 17);
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned c = 0; c < 32; ++c)
            check(
                out[r * 32 + c] == (r < 3 && c < 17 ? in[r * 32 + c] + 3 : -9),
                "runtime valid shape leaves GM tail untouched");
    rejects([&] { dynamic(in.data(), out.data(), 5, 17); }, "invalid runtime rows");
    padded(in.data(), out.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 4; ++c)
            check(
                out[r * 4 + c] == (r < 3 && c < 2 ? in[r * 2 + c] : std::numeric_limits<float>::max()),
                "CUBE Max padding");
    std::array<float, 64> right{};
    right.fill(1);
    for (unsigned i = 0; i < 256; ++i)
        in[i] = float(int(i % 17) - 8);
    std::array<int8_t, 256> q{};
    quantized(in.data(), right.data(), q.data());
    for (unsigned r = 0; r < 32; ++r) {
        float sum = 0;
        for (unsigned k = 0; k < 8; ++k)
            sum += in[r * 8 + k];
        const int8_t ref = int(std::nearbyint(sum * .5f)) + 3;
        for (unsigned n = 0; n < 8; ++n)
            check(q[r * 8 + n] == ref, "matrix scalar prequantization and signed offset");
    }
    in.fill(100.f);
    quantized(in.data(), right.data(), q.data());
    for (auto value : q)
        check(value == 2, "S9 intermediate clipping followed by offset and S8 wrap");
    using T = Tile<Location::Vec, uint32_t, 4, 32>;
    T a, b, d;
    TEXPANDS(a, 10u);
    TEXPANDS(b, 0u);
    rejects([&] { TDIV(d, a, b); }, "integer division by zero");
    using I = Tile<Location::Vec, int32_t, 4, 32>;
    I offsets;
    TEXPANDS(offsets, 512);
    Mem<uint32_t, 4, 32> mem(u.data());
    rejects([&] { MGATHER(d, mem, offsets, IndexedByteRange{0, 508}); }, "certified indexed range");
    TEXPANDS(offsets, 1);
    rejects([&] { MGATHER_ADD(d, reinterpret_cast<uint64_t>(u.data()), offsets, a, 32, 4); }, "atomic alignment");
    SharedTile<T> shared;
    rejects([&] { TMOV_S2L_BROADCAST(d, shared); }, "missing Shared producer");
    using Parent = Tile<Location::Vec, float, 64, 4>;
    SharedTile<Parent> parent;
    VecTileM32<float, 32, 4> writer;
    TEXPANDS(writer, 1.f);
    auto first = range::assemble<4, 0>(parent);
    TMOV_CUBE_TO_SHARED_ND(first, writer);
    auto overlap = range::assemble_last<4, 0>(parent);
    rejects([&] { TMOV_CUBE_TO_SHARED_ND(overlap, writer); }, "overlapping Shared writer");
    std::cout << "Linx boundary, alias, shape, and protocol checks passed\n";
}
#endif
