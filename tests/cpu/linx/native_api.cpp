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

template <typename T, int R, int C>
using Memory = global_tensor<T, RowMajor<R, C>>;
template <typename T, int R, int C>
using Vector = Tile<Location::Vec, T, R, C>;

void elementwise(float* input, float* output)
{
    Vector<float, 4, 32> a, b, c;
    Memory<float, 4, 32> gm(input), out(output);
    TLOAD(a, gm);
    TEXPANDS(b, 2.f);
    TADD(c, a, b);
    TMUL(c, c, b);
    TSUBS(c, c, 1.f);
    TSTORE(out, c);
}
void reduction(float* input, float* output)
{
    Vector<float, 1, 32> a;
    Vector<float, 1, 1> out;
    Memory<float, 1, 32> gm(input);
    Memory<float, 1, 1> result(output);
    TLOAD(a, gm);
    TROWSUM(out, a);
    TSTORE(result, out);
}
void cubeRoundtrip(uint64_t* input, uint64_t* output)
{
    VecTileM32<uint64_t, 32, 4> a;
    Memory<uint64_t, 32, 4> in(input), out(output);
    TLOAD(a, in);
    TSTORE(out, a);
}
void predicates(uint32_t* input, uint32_t* output)
{
    VecTileM32<uint32_t, 32, 4> a, b, c;
    Memory<uint32_t, 32, 4> in(input), out(output);
    TLOAD(a, in);
    TEXPANDS(b, 99u);
    TEXPANDS(c, 7u);
    auto predicate = predicate::compare_scalar<CmpMode::GE>(a, 20u);
    predicate::select(c, predicate, a, b);
    TSTORE(out, c);
}
void matrixMultiply(float* left, float* right, float* output)
{
    CubeTileM32<float, 32, 8> a;
    Tile<Location::Right, float, 8, 8, BLayout::CubeN8> b;
    CubeAccumulatorM32<float, 32, 8> c;
    Memory<float, 32, 8> ga(left), gc(output);
    Memory<float, 8, 8> gb(right);
    TLOAD(a, ga);
    TLOAD(b, gb);
    TMATMUL(c, a, b);
    TSTORE(gc, c);
}
void atomicGather(uint32_t* memory, int64_t* offsets, uint32_t* values, uint32_t* observed)
{
    Vector<int64_t, 1, 32> idx;
    Vector<uint32_t, 1, 32> val, old;
    Memory<int64_t, 1, 32> gi(offsets);
    Memory<uint32_t, 1, 32> gv(values), go(observed);
    TLOAD(idx, gi);
    TLOAD(val, gv);
    MGATHER_ADD(old, reinterpret_cast<uint64_t>(memory), idx, val, 32);
    TSTORE(go, old);
}
void conversion(float* input, __half* output)
{
    Tile<Location::Vec, float, 1, 64, BLayout::RowMajor, 1, 32> a;
    Tile<Location::Vec, __half, 1, 64, BLayout::RowMajor, 1, 32> b;
    Memory<float, 1, 32> ga(input);
    Memory<__half, 1, 32> gb(output);
    TLOAD(a, ga);
    TCVT(b, a);
    TSTORE(gb, b);
}

void collective(float* input, float* output)
{
    Vector<float, 1, 32> a, b;
    Memory<float, 1, 32> in(input + get_thread_idx() * 32), out(output + get_thread_idx() * 32);
    TLOAD(a, in);
    GMOV(b, (get_thread_idx() + 1) % 4, a);
    TSTORE(out, b);
}
static void scalarRounding(float* input, int32_t* output)
{
    for (unsigned i = 0; i < 6; ++i) {
        linx_cvt<LINX_RNE>(output[i], input[i]);
        linx_cvt<LINX_RTZ>(output[6 + i], input[i]);
        linx_cvt<LINX_RDN>(output[12 + i], input[i]);
        linx_cvt<LINX_RUP>(output[18 + i], input[i]);
        linx_cvt<LINX_RNA>(output[24 + i], input[i]);
        linx_cvt<LINX_RTO>(output[30 + i], input[i]);
        linx_cvt<LINX_RHB>(output[36 + i], input[i]);
    }
}
static void packedBf16(uint32_t* output)
{
    __bf16x2 pair{};
    linx_cvt_packed(pair, 1.f, 2.f);
    output[0] = pair.data;
    output[1] = blkv_bf16x2_add(pair, pair).data;
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
void scalarConversion(float input, __bf16& output) { linx_cvt(output, input); }
#include <array>
#include <iostream>
#include <stdexcept>
static void check(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
int main()
{
    std::array<float, 256> a{}, b{}, c{};
    for (unsigned i = 0; i < a.size(); ++i)
        a[i] = i * .25f - 17.f;
    elementwise(a.data(), c.data());
    for (unsigned i = 0; i < 128; ++i)
        check(c[i] == (a[i] + 2.f) * 2.f - 1.f, "elementwise or alias snapshot");
    a.fill(0);
    a[0] = 16777216.f;
    a[1] = 1.f;
    a[2] = -16777216.f;
    reduction(a.data(), c.data());
    check(c[0] == 0.f, "TROWSUM must round every FP32 step");
    std::array<uint64_t, 128> wideIn{}, wideOut{};
    for (unsigned i = 0; i < 128; ++i)
        wideIn[i] = 0x123456789abc0000ULL + i;
    cubeRoundtrip(wideIn.data(), wideOut.data());
    check(wideIn == wideOut, "M32 B64 plane layout");
    std::array<uint32_t, 128> uintIn{}, uintOut{};
    for (unsigned i = 0; i < 128; ++i)
        uintIn[i] = i;
    predicates(uintIn.data(), uintOut.data());
    for (unsigned i = 0; i < 128; ++i)
        check(uintOut[i] == (i >= 20 ? i : 99), "PredicateCell selection");
    for (unsigned i = 0; i < 256; ++i)
        a[i] = (int(i) % 11 - 5) * .5f;
    for (unsigned i = 0; i < 64; ++i)
        b[i] = (int(i) % 7 - 3) * .25f;
    matrixMultiply(a.data(), b.data(), c.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned n = 0; n < 8; ++n) {
            float sum = 0;
            for (unsigned k = 0; k < 8; ++k) {
                float product = a[r * 8 + k] * b[k * 8 + n];
                sum += product;
            }
            check(c[r * 8 + n] == sum, "matrix CUBE layout and accumulation");
        }
    std::array<int64_t, 32> offsets{};
    std::array<uint32_t, 32> values{}, old{};
    uint32_t counter = 10;
    values.fill(1);
    atomicGather(&counter, offsets.data(), values.data(), old.data());
    check(counter == 42, "atomic repeated address");
    for (unsigned i = 0; i < 32; ++i)
        check(old[i] == 10 + i, "atomic observed old value");
    std::array<__half, 32> half{};
    conversion(a.data(), half.data());
    for (unsigned i = 0; i < 32; ++i)
        check(half[i] == static_cast<__half>(a[i]), "FP32 to FP16");
    __bf16 bf;
    scalarConversion(1.00390625f, bf);
    check(bf.data == 0x3f80, "BF16 ties to even");
    for (unsigned i = 0; i < 128; ++i)
        a[i] = i;
    linx_cpu::runCore([&] { collective(a.data(), c.data()); });
    for (unsigned pe = 0; pe < 4; ++pe)
        for (unsigned i = 0; i < 32; ++i)
            check(c[pe * 32 + i] == a[((pe + 1) % 4) * 32 + i], "Core4 GMOV peer snapshot");
    std::cout << "Linx native API CPU execution checks passed\n";
}
#endif
