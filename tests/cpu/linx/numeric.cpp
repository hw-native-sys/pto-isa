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
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace pto;
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <typename T>
static void stagedTileRounding(
    uint16_t aBits, uint16_t bBits, uint16_t expBits, uint16_t sqrtBits, uint16_t reciprocalBits)
{
    Tile<Location::Vec, T, 1, 32> a, b, d;
    T output[32]{};
    global_tensor<T, RowMajor<1, 32>> out(output);
    TEXPANDS(a, std::bit_cast<T>(aBits));
    TEXPANDS(b, std::bit_cast<T>(bBits));
    TEXPDIF(d, a, b);
    TSTORE(out, d);
    for (auto value : output)
        check(std::bit_cast<uint16_t>(value) == expBits, "TEXPDIF must round the intermediate subtraction");
    TEXPANDS(a, std::bit_cast<T>(sqrtBits));
    TRSQRT(d, a);
    TSTORE(out, d);
    for (auto value : output)
        check(
            std::bit_cast<uint16_t>(value) == reciprocalBits,
            "TRSQRT must follow the documented pinned-model contract");
}

template <typename T>
static void roundtrip(unsigned count)
{
    for (unsigned raw = 0; raw < count; ++raw) {
        T input{};
        std::memcpy(&input, &raw, sizeof(T));
        float value;
        linx_cvt(value, input);
        if (!std::isfinite(value))
            continue;
        T result{};
        linx_cvt(result, value);
        unsigned actual = 0;
        std::memcpy(&actual, &result, sizeof(T));
        if (actual != raw) {
            std::cerr << "roundtrip " << type_traits<T>::TypeCode << " raw=" << raw << " actual=" << actual
                      << " value=" << value << '\n';
            throw std::runtime_error("scalar roundtrip");
        }
    }
}
int main()
{
    roundtrip<__half>(65536);
    roundtrip<__bf16>(65536);
    roundtrip<__hif8>(256);
    roundtrip<__fp8_e4m3>(256);
    roundtrip<__fp8_e5m2>(256);
    roundtrip<__fp6_e3m2>(64);
    roundtrip<__fp6_e2m3>(64);
    roundtrip<__fp8_e8m0>(255);
    roundtrip<__fp8_e6m2>(255);
    // Integer rounding expectations do not depend on the host rounding mode.
    constexpr float inputs[] = {2.5f, 3.5f, -2.5f, -3.5f, 2.2f, -2.2f};
    constexpr int expected[][6] = {{2, 4, -2, -4, 2, -2}, {2, 3, -2, -3, 2, -2}, {2, 3, -3, -4, 2, -3},
                                   {3, 4, -2, -3, 3, -2}, {3, 4, -3, -4, 2, -2}, {3, 3, -3, -3, 3, -3},
                                   {3, 4, -2, -3, 2, -2}};
    for (unsigned i = 0; i < 6; ++i) {
        int out;
        linx_cvt<LINX_RNE>(out, inputs[i]);
        check(out == expected[0][i], "RNE");
        linx_cvt<LINX_RTZ>(out, inputs[i]);
        check(out == expected[1][i], "RTZ");
        linx_cvt<LINX_RDN>(out, inputs[i]);
        check(out == expected[2][i], "RTM");
        linx_cvt<LINX_RUP>(out, inputs[i]);
        check(out == expected[3][i], "RTP");
        linx_cvt<LINX_RNA>(out, inputs[i]);
        check(out == expected[4][i], "RNA");
        linx_cvt<LINX_RTO>(out, inputs[i]);
        check(out == expected[5][i], "RTO jam");
        linx_cvt<LINX_RHB>(out, inputs[i]);
        check(out == expected[6][i], "RHB");
    }
    __bf16 a{}, b{}, c{};
    linx_cvt(a, 2.f);
    linx_cvt(b, 3.f);
    linx_cvt(c, 4.f);
    auto number = [](__bf16 x) {
        float out;
        linx_cvt(out, x);
        return out;
    };
    check(number(blkv_bf16_add(a, b)) == 5, "BF16 add");
    check(number(blkv_bf16_sub(a, b)) == -1, "BF16 sub");
    check(number(blkv_bf16_mul(a, b)) == 6, "BF16 mul");
    check(number(blkv_bf16_div(c, a)) == 2, "BF16 div");
    check(number(blkv_bf16_sqrt(c)) == 2, "BF16 sqrt");
    check(number(blkv_bf16_madd(a, b, c)) == 10, "BF16 fmadd");
    check(number(blkv_bf16_msub(a, b, c)) == 2, "BF16 fmsub");
    check(number(blkv_bf16_max(a, b)) == 3, "BF16 fmax");
    __bf16 zero{};
    check(number(blkv_bf16_exp(zero)) == 1, "BF16 exp");
    auto one = std::bit_cast<__bf16>(uint16_t{0x3f80});
    auto tiny = std::bit_cast<__bf16>(uint16_t{1});
    auto minusOne = std::bit_cast<__bf16>(uint16_t{0xbf80});
    check(blkv_bf16_add<BLKV_RUP>(one, tiny).data == 0x3f81, "BF16 directed rounding retains tiny addend");
    check(blkv_bf16_add<BLKV_RNE>(one, tiny).data == 0x3f80, "BF16 nearest ignores distant addend");
    check(blkv_bf16_add<BLKV_RDN>(one, minusOne).data == 0x8000, "BF16 exact cancellation in RTM");
    // A float halfway between adjacent BF16 numbers rounds to the even carrier.
    for (unsigned raw = 0x3f000000; raw < 0x40000000; raw += 0x1000) {
        float in = std::bit_cast<float>(raw);
        __bf16 out;
        linx_cvt(out, in);
        check(out.data == ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16), "BF16 RNE midpoint grid");
    }
    unsigned char narrow;
    linx_cvt(narrow, 300.f);
    check(narrow == 44, "finite NOSAT integer wrap");
    linx_cvt<LINX_RNE, LINX_SAT>(narrow, 300.f);
    check(narrow == 255, "SAT integer clipping");
    __bf16x2 pair{};
    linx_cvt_packed(pair, 1.f, 2.f);
    check(pair.data == 0x40003f80u, "packed BF16 lane order");
    auto pairSum = blkv_bf16x2_add(pair, pair);
    check(pairSum.data == 0x40804000u, "packed BF16 addition");
    __fp16x2 halfPair{0x40003c00u};
    __fp4_e2m1x2 fp4;
    linx_cvt(fp4, halfPair);
    check(fp4.data == 0x42, "FP4 packed conversion");
    __fp16x2 restored;
    linx_cvt(restored, fp4);
    check(restored.data == halfPair.data, "FP4 unpacked conversion");
    halfPair.data = 0x48004800u; // Both FP16 lanes are 8.0.
    linx_cvt(fp4, halfPair);
    check(fp4.data == 0x66, "pinned E2M1 overflow encoding");
    __fp8_e4m3 fp8;
    linx_cvt(fp8, 449.f);
    check(fp8.data == 0x7f, "E4M3 finite overflow encoding");
    linx_cvt<LINX_RNE, LINX_SAT>(fp8, 449.f);
    check(fp8.data == 0x7e, "E4M3 saturated maximum");
    // Truncated mantissas retain sign/exponent and zero all unused low bits.
    __tf32 tf;
    __hf32 hf;
    linx_cvt(tf, 1.0007f);
    linx_cvt(hf, 1.0007f);
    check((tf.data & 8191) == 0 && (hf.data & 4095) == 0, "TF32/HF32 carrier low bits");
    stagedTileRounding<__half>(15360, 38272, 0x4171, 12290, 0x41a6);
    stagedTileRounding<__bf16>(16256, 47944, 0x402e, 12293, 0x4731);
    std::cout << "Linx scalar conversion and BF16 execution checks passed\n";
}
