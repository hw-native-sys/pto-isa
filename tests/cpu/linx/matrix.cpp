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
void matrixAux(float* left, float* right, __half* output, __half* rowOutput, __half* groupOutput)
{
    CubeTileM32<float, 32, 8> a;
    CubeTileN8<float, 8, 8> b;
    CubeAccumulatorM32<__half, 32, 8> d;
    CubeAccumulatorM32<__half, 32, 2, 32, 1> row, group;
    Mem<float, 32, 8> ma(left);
    Mem<float, 8, 8> mb(right);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TMATMUL(d, a, b, fixp::f16().relu().row_max(row).group_max<8>(group));
    Mem<__half, 32, 8> md(output);
    Mem<__half, 32, 1> mr(rowOutput), mg(groupOutput);
    TSTORE(md, d);
    TSTORE(mr, row);
    TSTORE(mg, group);
}
void matrixMX(__fp8_e4m3* left, __fp8_e5m2* right, float* output)
{
    CubeTileM16<__fp8_e4m3, 16, 32> a;
    CubeTileN8<__fp8_e5m2, 32, 8> b;
    Tile<Location::Scaling, __fp8_e8m0, 32, 4, BLayout::CubeM32, 16, 1> sa;
    Tile<Location::Scaling, __fp8_e8m0, 32, 4, BLayout::CubeM32, 8, 1> sb;
    CubeAccumulatorM16<float, 16, 8> d, c;
    CubeBias<float, 8> bias;
    Mem<__fp8_e4m3, 16, 32> ma(left);
    Mem<__fp8_e5m2, 32, 8> mb(right);
    TLOAD(a, ma);
    TLOAD(b, mb);
    __fp8_e8m0 x[16]{}, y[8]{};
    for (auto& value : x)
        value.data = 128;
    for (auto& value : y)
        value.data = 129;
    Mem<__fp8_e8m0, 16, 1> sx(x);
    Mem<__fp8_e8m0, 8, 1> sy(y);
    TLOAD(sa, sx);
    TLOAD(sb, sy);
    TMATMUL_MX(d, a, sa, b, sb);
    Mem<float, 16, 8> md(output);
    TSTORE(md, d);
    TEXPANDS(c, 2.f);
    TMATMUL_MX_ACC(d, c, a, sa, b, sb);
    Mem<float, 16, 8> mc(output + 128);
    TSTORE(mc, d);
    float biasData[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    Mem<float, 1, 8> biasMemory(biasData);
    TLOAD(bias, biasMemory);
    TMATMUL_MX_BIAS(d, a, sa, b, sb, bias);
    Mem<float, 16, 8> mo(output + 256);
    TSTORE(mo, d);
}
void matrixAccScale(float* left, float* right, float* output)
{
    CubeTileM32<float, 32, 8> a;
    CubeTileN8<float, 8, 8> b;
    CubeAccumulatorM32<float, 32, 8> d;
    VecTileM32<uint8_t, 32, 4, 32, 1> scale;
    Mem<float, 32, 8> ma(left), md(output);
    Mem<float, 8, 8> mb(right);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TEXPANDS(d, 8.f);
    TEXPANDS(scale, uint8_t(2));
    TMATMUL_ACC(d, d, a, b, fixp::keep_acc().cscale(scale));
    TSTORE(md, d);
}
void matrixShared(float* left, float* right, float* output)
{
    using A = SharedMatrixLeft<float, 64, 8>;
    using B = SharedMatrixRight<float, 8, 8>;
    Mem<float, 64, 8> ma(left);
    Mem<float, 8, 8> mb(right);
    auto a = TLOAD<A, 8>(ma);
    auto b = TLOAD<B, 8>(mb);
    CubeAccumulatorM16<float, 16, 8> d;
    TMATMUL(d, a, b);
    Mem<float, 16, 8> md(output + get_thread_idx() * 128);
    TSTORE(md, d);
}
void gemv(__half* vector, __half* matrix, float* output)
{
    CubeTileM16<__half, 1, 32> a;
    CubeTileN8<__half, 32, 8> b;
    CubeAccumulatorM16<float, 1, 8> d, c;
    CubeBias<float, 8> bias;
    Mem<__half, 1, 32> ma(vector);
    Mem<__half, 32, 8> mb(matrix);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TGEMV(d, b, a);
    Mem<float, 1, 8> md(output);
    TSTORE(md, d);
    TEXPANDS(c, 2.f);
    TGEMV_ACC(d, c, b, a);
    Mem<float, 1, 8> mc(output + 8);
    TSTORE(mc, d);
    float biasData[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    Mem<float, 1, 8> biasMemory(biasData);
    TLOAD(bias, biasMemory);
    TGEMV_BIAS(d, b, a, bias);
    Mem<float, 1, 8> mo(output + 16);
    TSTORE(mo, d);
    TGEMV_MX(d, b, a);
    Mem<float, 1, 8> mx(output + 24);
    TSTORE(mx, d);
    TGEMV_MX_ACC(d, c, b, a);
    Mem<float, 1, 8> mxa(output + 32);
    TSTORE(mxa, d);
    TGEMV_MX_BIAS(d, b, a, bias);
    Mem<float, 1, 8> mxb(output + 40);
    TSTORE(mxb, d);
}
void matrixHif4(__fp4_hif4x2* left, __fp4_hif4x2* right, float* output)
{
    CubeTileM16<__fp4_hif4x2, 16, 64> a;
    CubeTileN8<__fp4_hif4x2, 64, 8> b;
    Tile<Location::Scaling, uint32_t, 32, 1, BLayout::CubeM32, 16, 1> sa;
    Tile<Location::Scaling, uint32_t, 32, 1, BLayout::CubeM32, 8, 1> sb;
    CubeAccumulatorM16<float, 16, 8> d;
    Mem<__fp4_hif4x2, 16, 64> ma(left);
    Mem<__fp4_hif4x2, 64, 8> mb(right);
    Mem<float, 16, 8> md(output);
    TLOAD(a, ma);
    TLOAD(b, mb);
    TEXPANDS(sa, 0xffffffc0u);
    TEXPANDS(sb, 0xc0u);
    TMATMUL_MX(d, a, sa, b, sb);
    TSTORE(md, d);
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
int main()
{
    std::array<float, 512> a{}, b{}, out{};
    std::array<__half, 256> half{};
    std::array<__half, 32> row{}, group{};
    for (unsigned i = 0; i < a.size(); ++i)
        a[i] = (int(i) % 11 - 5) * .1111f;
    for (unsigned i = 0; i < 64; ++i)
        b[i] = (int(i) % 7 - 3) * .2222f;
    matrixAux(a.data(), b.data(), half.data(), row.data(), group.data());
    for (unsigned r = 0; r < 32; ++r) {
        __half maximum = 0;
        for (unsigned n = 0; n < 8; ++n) {
            float expected = 0;
            for (unsigned k = 0; k < 8; ++k) {
                float product = a[r * 8 + k] * b[k * 8 + n];
                expected += product;
            }
            __half rounded = static_cast<__half>(std::fmax(expected, 0.f));
            check(half[r * 8 + n] == rounded, "matrix FP16 postprocess");
            if (rounded > maximum)
                maximum = rounded;
        }
        check(row[r] == maximum && group[r] == maximum, "RowMax/GroupMax use final FP16 D values");
    }
    matrixAccScale(a.data(), b.data(), out.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned n = 0; n < 8; ++n) {
            float expected = 2;
            for (unsigned k = 0; k < 8; ++k) {
                float product = a[r * 8 + k] * b[k * 8 + n];
                expected += product;
            }
            check(out[r * 8 + n] == expected, "ACC CScale and D == C snapshot");
        }
    std::array<__fp8_e4m3, 512> ma{};
    std::array<__fp8_e5m2, 256> mb{};
    for (auto& x : ma)
        x.data = 0x3c;
    for (auto& x : mb)
        x.data = 0x40;
    matrixMX(ma.data(), mb.data(), out.data());
    for (unsigned i = 0; i < 128; ++i) {
        check(out[i] == 768, "MX E4M3/E5M2 scaled dot");
        check(out[128 + i] == 770, "MX ACC");
        check(out[256 + i] == 771, "MX BIAS");
    }
    linx_cpu::runCore([&] { matrixShared(a.data(), b.data(), out.data()); });
    for (unsigned r = 0; r < 64; ++r)
        for (unsigned n = 0; n < 8; ++n) {
            float expected = 0;
            for (unsigned k = 0; k < 8; ++k) {
                float product = a[r * 8 + k] * b[n * 8 + k];
                expected += product;
            }
            check(out[r * 8 + n] == expected, "Shared B NK storage and cooperative M shards");
        }
    std::array<__half, 32> va{};
    std::array<__half, 256> vb{};
    va.fill(1);
    vb.fill(2);
    gemv(va.data(), vb.data(), out.data());
    for (unsigned n = 0; n < 8; ++n) {
        check(out[n] == 64, "TGEMV operand order");
        check(out[8 + n] == 66, "TGEMV ACC");
        check(out[16 + n] == 67, "TGEMV BIAS");
        check(out[24 + n] == 64, "TGEMV MX without scales");
        check(out[32 + n] == 66, "TGEMV MX ACC");
        check(out[40 + n] == 67, "TGEMV MX BIAS");
    }
    std::array<__fp4_hif4x2, 512> hifA{};
    std::array<__fp4_hif4x2, 256> hifB{};
    for (auto& v : hifA)
        v.data = 0x44;
    for (auto& v : hifB)
        v.data = 0x44;
    matrixHif4(hifA.data(), hifB.data(), out.data());
    for (unsigned i = 0; i < 128; ++i)
        check(out[i] == 256, "HIF4 E1M2 lanes with hierarchical E6M2/E1 scales");
    std::cout << "Linx matrix, MX, postprocess, and GEMV execution checks passed\n";
}
#endif
