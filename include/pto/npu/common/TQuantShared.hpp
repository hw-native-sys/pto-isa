/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
//
// Arch-independent helpers shared by the a5 and a6 TQuant implementations.
// Every function here is byte-for-byte equivalent on both targets (the only
// historical difference was the CCE_VL / REPEAT_BYTE spelling, both 256 B on
// dav-c310-vec and dav-920r1-vec). This header must be included AFTER the
// per-arch common.hpp/utils.hpp so the CCE intrinsics and CreatePredicate are
// visible; it deliberately includes nothing arch-specific itself.
#ifndef PTO_NPU_COMMON_TQUANT_SHARED_HPP
#define PTO_NPU_COMMON_TQUANT_SHARED_HPP

#include <type_traits>

namespace pto {
namespace tquant_detail {

constexpr int16_t kF32StorageBits = 32;
constexpr int16_t kF32MantissaBits = 23;
constexpr uint32_t kF32PositiveInfBits = 0x7F800000u;
constexpr int16_t kBf16StorageBits = 16;
constexpr int16_t kBf16MantissaBits = 7;
constexpr uint16_t kBf16PositiveInfBits = 0x7F80u;
constexpr int16_t kIeee754ExponentBits = 8;
constexpr int16_t kFp4CodeBits = 4;
constexpr int16_t kPackedByteBits = 8;
constexpr int16_t kVectorLaneBits = 32;
constexpr int16_t kSignBitCount = 1;
constexpr uint32_t kSingleBitMask = 1u;

/*
 * Floating-point bit fields used by the vector bit operations below.
 *
 * FP32 lane:
 *   bit 31       30..23        22..0
 *   +------------+-------------+--------------------+
 *   | sign       | exponent    | mantissa           |
 *   +------------+-------------+--------------------+
 *                 ^ exponentOffset == mantissaBits
 *
 * BF16 uses the same exponent width in a 16-bit container:
 *   bit 15       14..7         6..0
 *   +------------+-------------+--------------------+
 *   | sign       | exponent    | mantissa           |
 *   +------------+-------------+--------------------+
 */
template <typename BitsT, int16_t StorageBits, int16_t ExponentBits, int16_t MantissaBits, BitsT PositiveInfBits>
struct FloatBitFieldLayout {
    using BitsType = BitsT;

    static constexpr int16_t storageBits = StorageBits;
    static constexpr int16_t exponentBits = ExponentBits;
    static constexpr int16_t mantissaBits = MantissaBits;
    static constexpr int16_t signBitOffset = storageBits - kSignBitCount;
    static constexpr int16_t signClearShift = kSignBitCount;
    static constexpr int16_t exponentOffset = mantissaBits;
    static constexpr uint32_t exponentBias = (kSingleBitMask << (exponentBits - kSignBitCount)) - kSingleBitMask;
    static constexpr int32_t negativeExponentBias = -static_cast<int32_t>(exponentBias);
    static constexpr BitsT signMask = static_cast<BitsT>(BitsT{kSingleBitMask} << signBitOffset);
    static constexpr BitsT absMask = static_cast<BitsT>(~signMask);
    static constexpr BitsT positiveInfBits = PositiveInfBits;
};

using F32BitFieldLayout =
    FloatBitFieldLayout<uint32_t, kF32StorageBits, kIeee754ExponentBits, kF32MantissaBits, kF32PositiveInfBits>;
using Bf16BitFieldLayout =
    FloatBitFieldLayout<uint16_t, kBf16StorageBits, kIeee754ExponentBits, kBf16MantissaBits, kBf16PositiveInfBits>;

/*
 * FP4 values are packed as two 4-bit codes per byte.
 *
 * Packed byte:
 *   bit 7..4       bit 3..0
 *   +--------------+--------------+
 *   | odd code     | even code    |
 *   +--------------+--------------+
 *
 * The vector shift intrinsics operate on 32-bit lanes.  left/right by
 * lowCodeShift extracts the low code bits; right by highCodeShift moves
 * the odd code into byte bits [7:4].
 */
template <int16_t CodeBits, int16_t PackedByteBits = kPackedByteBits, int16_t VectorLaneBits = kVectorLaneBits>
struct PackedSubBytePairLayout {
    static constexpr int16_t codeBits = CodeBits;
    static constexpr int16_t packedByteBits = PackedByteBits;
    static constexpr int16_t vectorLaneBits = VectorLaneBits;
    static constexpr int16_t lowCodeShift = vectorLaneBits - codeBits;
    static constexpr int16_t highCodeShift = vectorLaneBits - packedByteBits;
};

using Fp4PackedPairLayout = PackedSubBytePairLayout<kFp4CodeBits>;

/*
 * E2M1 code layout used after source values have been scaled to FP32 lanes.
 *
 * 4-bit code:
 *   bit 3        bit 2..1       bit 0
 *   +------------+--------------+------------+
 *   | sign       | exponent     | mantissa   |
 *   +------------+--------------+------------+
 *
 * maxBiasedExponent clamps the FP32 exponent into the finite E2M1 range.
 * magicRoundingExponentOffset builds the FP32 addend used to round while
 * retaining one E2M1 mantissa bit.  negativeCodeOffset maps positive
 * magnitude codes into signed 4-bit code space before the pack step.
 */
template <typename SourceFloatLayout, typename PackedPairLayout>
struct Fp4E2M1CodeLayout {
    static constexpr int16_t exponentBits = 2;
    static constexpr int16_t mantissaBits = 1;
    static constexpr uint32_t maxExponentDelta = (1u << exponentBits) - 2u;
    static constexpr uint32_t maxBiasedExponent = SourceFloatLayout::exponentBias + maxExponentDelta;
    static constexpr int32_t magicRoundingExponentOffset = SourceFloatLayout::mantissaBits - mantissaBits;
    static constexpr int16_t magnitudeCodeShift = mantissaBits;
    static constexpr uint32_t signBitMask = 1u << (PackedPairLayout::codeBits - 1);
    static constexpr uint32_t maxMagnitudeCode = signBitMask - 1u;
    static constexpr int32_t negativeCodeOffset = -static_cast<int32_t>(signBitMask);
};

using Fp4E2M1Code = Fp4E2M1CodeLayout<F32BitFieldLayout, Fp4PackedPairLayout>;

/*
 * TQUANT flattens auxiliary exp/max/scaling tiles to a single logical row
 * while keeping their valid extents runtime-sized.
 */
struct FlatTile1DLayout {
    static constexpr int rows = 1;
    static constexpr int runtimeValidExtent = -1;
    static constexpr int sFractalSize = TileConfig::fractalABSize;
};

} // namespace tquant_detail

template <typename T, typename U>
PTO_INTERNAL MaskReg TQuantPSetTyped(U dist)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return pset_b32(dist);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        return pset_b16(dist);
    } else {
        return pset_b8(dist);
    }
}

// Convert two FP16 vectors to BF16 while explicitly preserving Inf/NaN values.
// The saturating f16->bf16 cast used by the quantization path would otherwise
// silently change NaN/Inf behavior; stage 2 relies on the reduced max having
// bf16 exp==0x7F80 to emit the E8M0 NaN/Inf sentinel.
template <typename SrcHalfVec, typename DstBf16Vec>
PTO_INTERNAL void Fp16ToBf16PreserveSpecial(
    SrcHalfVec& src0, SrcHalfVec& src1, DstBf16Vec& dst0, DstBf16Vec& dst1, MaskReg& preg_vl0, MaskReg& preg_vl1)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    constexpr uint16_t kFp16ExpMask = 0x7C00;
    constexpr uint16_t kFp16MantissaMask = 0x03FF;
    constexpr uint16_t kFp16InfBits = 0x7C00;
    constexpr uint16_t kBf16InfBits = 0x7F80;
    constexpr uint16_t kBf16NanBits = 0x7FC0;

    RegTensor<uint16_t> vu16_abs_0, vu16_abs_1;
    RegTensor<uint16_t> vu16_exp_0, vu16_exp_1;
    RegTensor<uint16_t> vu16_mantissa_0, vu16_mantissa_1;
    RegTensor<uint16_t> vu16_abs_mask, vu16_exp_mask, vu16_mantissa_mask;
    RegTensor<uint16_t> vu16_bf16_inf, vu16_bf16_nan;
    vector_bool preg_special_0, preg_special_1, preg_nan_0, preg_nan_1, preg_inf_0, preg_inf_1;

    vbr(vu16_abs_mask, kBf16AbsMask);
    vbr(vu16_exp_mask, kFp16ExpMask);
    vbr(vu16_mantissa_mask, kFp16MantissaMask);
    vbr(vu16_bf16_inf, kBf16InfBits);
    vbr(vu16_bf16_nan, kBf16NanBits);
    vand(vu16_abs_0, (vector_u16&)src0, vu16_abs_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_abs_1, (vector_u16&)src1, vu16_abs_mask, preg_vl1, MODE_ZEROING);
    vand(vu16_exp_0, vu16_abs_0, vu16_exp_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_exp_1, vu16_abs_1, vu16_exp_mask, preg_vl1, MODE_ZEROING);
    vand(vu16_mantissa_0, vu16_abs_0, vu16_mantissa_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_mantissa_1, vu16_abs_1, vu16_mantissa_mask, preg_vl1, MODE_ZEROING);
    vcmps_eq(preg_special_0, vu16_exp_0, kFp16ExpMask, preg_vl0);
    vcmps_eq(preg_special_1, vu16_exp_1, kFp16ExpMask, preg_vl1);
    vcmps_ne(preg_nan_0, vu16_mantissa_0, 0, preg_special_0);
    vcmps_ne(preg_nan_1, vu16_mantissa_1, 0, preg_special_1);
    vcmps_eq(preg_inf_0, vu16_abs_0, kFp16InfBits, preg_vl0);
    vcmps_eq(preg_inf_1, vu16_abs_1, kFp16InfBits, preg_vl1);
    vcvt(dst0, src0, preg_vl0, ROUND_Z);
    vcvt(dst1, src1, preg_vl1, ROUND_Z);
    vsel((vector_u16&)dst0, vu16_bf16_inf, (vector_u16&)dst0, preg_inf_0);
    vsel((vector_u16&)dst1, vu16_bf16_inf, (vector_u16&)dst1, preg_inf_1);
    vsel((vector_u16&)dst0, vu16_bf16_nan, (vector_u16&)dst0, preg_nan_0);
    vsel((vector_u16&)dst1, vu16_bf16_nan, (vector_u16&)dst1, preg_nan_1);
}

// Reduce one 256-element DINTLV_B16 window to 8 per-block abs raw maxima.
// OCP follows dynamic_mx_quant_tail_axis_fp8: FP16 is first converted to BF16
// then both FP16/BF16 paths reduce BF16 abs bits. NV/cuBLAS reduces FP16 as
// numeric FP16, so callers can disable that conversion.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_DintlvWindow(
    __ubuf__ T* srcPtr, uint32_t offset, uint32_t remaining, RegTensor<T>& vb16_max)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    RegTensor<T> vb16_in_1, vb16_in_2;
    RegTensor<uint16_t> vu16_abs_1, vu16_abs_2, vu16_bf16_abs_mask;
    uint32_t even_count = (remaining + 1) / 2;
    uint32_t odd_count = remaining / 2;
    MaskReg preg_vl0 = CreatePredicate<T>(even_count);
    MaskReg preg_vl1 = CreatePredicate<T>(odd_count);
    vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16);

    vbr(vu16_bf16_abs_mask, kBf16AbsMask);
    if constexpr (std::is_same<T, half>::value && fp16AsBf16ForMax) {
        vector_bf16 vb16_bf16_1, vb16_bf16_2;
        Fp16ToBf16PreserveSpecial(vb16_in_1, vb16_in_2, vb16_bf16_1, vb16_bf16_2, preg_vl0, preg_vl1);
        vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
        vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
    } else {
        vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
        vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
    }

    vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
    vcgmax((vector_u16&)vb16_max, vu16_abs_1, preg_vl0, MODE_ZEROING);
}

// Assumption: input total size is a multiple of 32 VLs.
// Uses 2 VLs per inner iteration (1 DINTLV + 1 vcgmax + 1 vstus) to avoid
// WAW hazard on the vstus auto-increment scalar register when using 2 vstus per iteration.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_ND_largesizes(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned vl_count, unsigned total_elements_count)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    RegTensor<T> vb16_in_1, vb16_in_2, vb16_max_1;
    RegTensor<uint16_t> vu16_abs_1, vu16_abs_2, vu16_bf16_abs_mask;
    vector_bf16 vb16_bf16_1, vb16_bf16_2;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = CCE_VL / sizeof(T);     // 256 B / 2 B = 128 elements per VL
    constexpr uint32_t grps_per_vl = elements_per_vl / grp_size; // 128 / 32 = 4 groups per VL
    constexpr uint32_t num_vl_per_inner_loop = 2;                // 2 VLs per inner loop (1 DINTLV load)
    constexpr uint32_t num_vl_per_outer_loop = 32;
    constexpr uint32_t grps_per_inner_loop = num_vl_per_inner_loop * grps_per_vl; // 2 * 4 = 8 grps per inner loop
    constexpr uint32_t grps_per_outer_loop = num_vl_per_outer_loop * grps_per_vl; // 32 * 4 = 128
    constexpr uint32_t blks_per_vl = CCE_VL / BLOCK_BYTE_SIZE;                    // 8 blocks per VL
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    vbr(vu16_bf16_abs_mask, kBf16AbsMask);
    uint16_t outerLoopLimit = static_cast<uint16_t>(vl_count / num_vl_per_outer_loop);
    constexpr uint16_t innerLoopLimit = static_cast<uint16_t>(num_vl_per_outer_loop / num_vl_per_inner_loop);
    for (uint16_t i = 0; i < outerLoopLimit; ++i) {     // 32 VLs per outer loop
        for (uint16_t j = 0; j < innerLoopLimit; ++j) { // 2 VLs per inner loop
            MaskReg preg_vl0 = CreatePredicate<T>(total_count);
            MaskReg preg_vl1 = CreatePredicate<T>(total_count);
            uint32_t offset = (i * num_vl_per_outer_loop + j * num_vl_per_inner_loop) * elements_per_vl;
            uint32_t grp_offset = grps_per_outer_loop * i + grps_per_inner_loop * j;
            vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16); // loads 2 VLs (256 bf16 elements)

            if constexpr (std::is_same<T, half>::value && fp16AsBf16ForMax) {
                Fp16ToBf16PreserveSpecial(vb16_in_1, vb16_in_2, vb16_bf16_1, vb16_bf16_2, preg_vl0, preg_vl1);
                vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
                vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
            } else {
                vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
                vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
            }

            vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
            vcgmax((vector_u16&)vb16_max_1, vu16_abs_1, preg_vl0, MODE_ZEROING);
            vstus(ureg_max, blks_per_vl, vb16_max_1, maxPtr + grp_offset);
        }
        vstas(ureg_max, maxPtr + grps_per_outer_loop * i, 0);
    }
}

// FP16-specific scaling+multiply for one FP8 window. The scaling buffer stores
// bf16 reciprocal scales even for fp16 input; widen to fp32 and multiply in fp32
// so the mantissa precision is preserved before downcast to fp8.
PTO_INTERNAL void ApplyHalfScalingToFP8Window(
    vector_f32& vb32_cvt_1, vector_f32& vb32_cvt_2, vector_f32& vb32_cvt_3, vector_f32& vb32_cvt_4,
    RegTensor<half>& vb16_in_1, RegTensor<half>& vb16_in_2, __ubuf__ half* scalingPtr, uint16_t i, MaskReg& preg_b16_1,
    MaskReg& preg_b16_2, MaskReg& preg_f32_1_even, MaskReg& preg_f32_1_odd, MaskReg& preg_f32_2_even,
    MaskReg& preg_f32_2_odd)
{
    vector_bf16 vb16_scaling_bf16;
    vector_f32 vb32_scaling;
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    vlds((vector_u16&)vb16_scaling_bf16, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
    vcvt(vb32_scaling, vb16_scaling_bf16, preg_all_b16, PART_EVEN);
    vcvt(vb32_cvt_1, vb16_in_1, preg_b16_1, PART_EVEN);
    vcvt(vb32_cvt_2, vb16_in_1, preg_b16_1, PART_ODD);
    vcvt(vb32_cvt_3, vb16_in_2, preg_b16_2, PART_EVEN);
    vcvt(vb32_cvt_4, vb16_in_2, preg_b16_2, PART_ODD);
    vmul(vb32_cvt_1, vb32_cvt_1, vb32_scaling, preg_f32_1_even, MODE_ZEROING);
    vmul(vb32_cvt_2, vb32_cvt_2, vb32_scaling, preg_f32_1_odd, MODE_ZEROING);
    vmul(vb32_cvt_3, vb32_cvt_3, vb32_scaling, preg_f32_2_even, MODE_ZEROING);
    vmul(vb32_cvt_4, vb32_cvt_4, vb32_scaling, preg_f32_2_odd, MODE_ZEROING);
}

// B16 (BF16/FP16) -> FP8. FP16 uses BF16 reciprocal scale, matching dynamic_mx_quant:
// convert input and BF16 scale to fp32, multiply in fp32, then downcast to fp8.
// Quantize one 256-element DINTLV_B16 window to FP8: scale via broadcast of
// 8 per-group scaling values, upcast b16->fp32 (EVEN/ODD), downcast fp32->fp8
// (PART_P0-P3 pack mod-4 bytes), OR-combine, and store.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_B16_Window(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, uint16_t i, uint32_t offset_b16,
    uint32_t remaining)
{
    constexpr uint32_t elementsPerVL_b8 = CCE_VL / sizeof(uint8_t);
    RegTensor<T> vb16_scaling, vb16_in_1, vb16_in_2, vb16_out_1, vb16_out_2;
    vector_f32 vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4;
    vector_f8e4m3 vb8_or1, vb8_or2, vb8_out, vb8_p0, vb8_p1, vb8_p2, vb8_p3;
    uint32_t evenCount = (remaining + 1) / 2;
    uint32_t oddCount = remaining / 2;
    uint32_t b8Count = remaining;
    uint32_t b16Count1 = evenCount;
    uint32_t b16Count2 = oddCount;
    uint32_t f32Count1Even = (evenCount + 1) / 2;
    uint32_t f32Count1Odd = evenCount / 2;
    uint32_t f32Count2Even = (oddCount + 1) / 2;
    uint32_t f32Count2Odd = oddCount / 2;
    MaskReg preg_b16_1 = CreatePredicate<T>(b16Count1);
    MaskReg preg_b16_2 = CreatePredicate<T>(b16Count2);
    MaskReg preg_f32_1_even = CreatePredicate<float>(f32Count1Even);
    MaskReg preg_f32_1_odd = CreatePredicate<float>(f32Count1Odd);
    MaskReg preg_f32_2_even = CreatePredicate<float>(f32Count2Even);
    MaskReg preg_f32_2_odd = CreatePredicate<float>(f32Count2Odd);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(b8Count);
    vlds(vb16_in_1, vb16_in_2, srcPtr, offset_b16, DINTLV_B16);
    if constexpr (std::is_same<T, half>::value) {
        ApplyHalfScalingToFP8Window(
            vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4, vb16_in_1, vb16_in_2, (__ubuf__ half*)scalingPtr, i,
            preg_b16_1, preg_b16_2, preg_f32_1_even, preg_f32_1_odd, preg_f32_2_even, preg_f32_2_odd);
    } else {
        vlds((vector_u16&)vb16_scaling, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
        vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
        vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
        vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN);
        vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);
        vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN);
        vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);
    }
    // fp32->fp8 P0..P3 writes to bytes 0..3 of each 32-bit slot; pair with mod-4 index.
    vcvt(vb8_p0, vb32_cvt_1, preg_f32_1_even, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(vb8_p1, vb32_cvt_3, preg_f32_2_even, ROUND_R, RS_ENABLE, PART_P1);
    vcvt(vb8_p2, vb32_cvt_2, preg_f32_1_odd, ROUND_R, RS_ENABLE, PART_P2);
    vcvt(vb8_p3, vb32_cvt_4, preg_f32_2_odd, ROUND_R, RS_ENABLE, PART_P3);
    vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
    vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
    vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
    vsts((vector_u8&)vb8_out, (__ubuf__ uint8_t*)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
}

PTO_INTERNAL void CalcE2M1SignedCodeI32(vector_s32& signedCode, vector_f32 scaled, MaskReg& preg_f32)
{
    using F32 = tquant_detail::F32BitFieldLayout;
    using E2M1 = tquant_detail::Fp4E2M1Code;

    vector_u32 vu32_abs_bits, vu32_exp, vu32_tmp;
    vector_bool preg_sign, preg_nan;

    vshrs(vu32_tmp, (vector_u32&)scaled, F32::signBitOffset, preg_f32, MODE_ZEROING);
    vcmps_ne(preg_sign, vu32_tmp, (uint32_t)0, preg_f32);
    vshls(vu32_abs_bits, (vector_u32&)scaled, F32::signClearShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_abs_bits, vu32_abs_bits, F32::signClearShift, preg_f32, MODE_ZEROING);
    vcmps_gt(preg_nan, vu32_abs_bits, F32::positiveInfBits, preg_f32);

    vshrs(vu32_exp, vu32_abs_bits, F32::exponentOffset, preg_f32, MODE_ZEROING);
    vmaxs(vu32_exp, vu32_exp, F32::exponentBias, preg_f32, MODE_ZEROING);
    vmins(vu32_exp, vu32_exp, E2M1::maxBiasedExponent, preg_f32, MODE_ZEROING);

    vadds((vector_s32&)vu32_tmp, (vector_s32&)vu32_exp, E2M1::magicRoundingExponentOffset, preg_f32, MODE_ZEROING);
    vshls(vu32_tmp, vu32_tmp, F32::exponentOffset, preg_f32, MODE_ZEROING);
    vadd(scaled, (vector_f32&)vu32_abs_bits, (vector_f32&)vu32_tmp, preg_f32, MODE_ZEROING);
    vsub(vu32_abs_bits, (vector_u32&)scaled, vu32_tmp, preg_f32);

    vadds((vector_s32&)vu32_exp, (vector_s32&)vu32_exp, F32::negativeExponentBias, preg_f32, MODE_ZEROING);
    vshls(vu32_exp, vu32_exp, E2M1::magnitudeCodeShift, preg_f32, MODE_ZEROING);
    vadd(vu32_abs_bits, vu32_abs_bits, vu32_exp, preg_f32, MODE_ZEROING);
    vmins(vu32_abs_bits, vu32_abs_bits, E2M1::maxMagnitudeCode, preg_f32, MODE_ZEROING);

    vadds(signedCode, (vector_s32&)vu32_abs_bits, E2M1::negativeCodeOffset, preg_f32, MODE_ZEROING);
    vsel(signedCode, signedCode, (vector_s32&)vu32_abs_bits, preg_sign);

    vsel(signedCode, (vector_s32&)vu32_abs_bits, signedCode, preg_nan);
}

PTO_INTERNAL void PackE2M1SignedCodeBytes(
    vector_u8& packedBytes, vector_s32 evenCode, vector_s32 oddCode, vector_u8& packIndex, MaskReg& preg_f32)
{
    using Pack = tquant_detail::Fp4PackedPairLayout;

    vector_u32 vu32_even, vu32_odd;

    vshls(vu32_even, (vector_u32&)evenCode, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_even, vu32_even, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshls(vu32_odd, (vector_u32&)oddCode, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_odd, vu32_odd, Pack::highCodeShift, preg_f32, MODE_ZEROING);
    vor(vu32_even, vu32_even, vu32_odd, preg_f32, MODE_ZEROING);
    vselr(packedBytes, (vector_u8&)vu32_even, packIndex);
}

PTO_INTERNAL void SaturateBf16NaNToPosInf(vector_u16& value, MaskReg& preg_b16)
{
    using Bf16 = tquant_detail::Bf16BitFieldLayout;

    vector_u16 v_abs, v_abs_mask, v_inf;
    vector_bool preg_nan;

    vbr(v_abs_mask, Bf16::absMask);
    vbr(v_inf, Bf16::positiveInfBits);
    vand(v_abs, value, v_abs_mask, preg_b16, MODE_ZEROING);
    vcmps_gt(preg_nan, v_abs, Bf16::positiveInfBits, preg_b16);
    vsel(value, v_inf, value, preg_nan);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Half_Window(
    __ubuf__ half* srcPtr, __ubuf__ half* scalingPtr, __ubuf__ uint8_t* dstPtr, uint16_t window, vector_u8& packIndex)
{
    constexpr uint32_t kElementsPerWindow = 256;
    constexpr uint32_t kPackedBytesPerWindow = kElementsPerWindow / 2;
    constexpr uint32_t kB16LanesPerReg = CCE_VL / sizeof(half);
    constexpr uint32_t kF32LanesPerReg = CCE_VL / sizeof(float);
    uint32_t b16LanesPerReg = kB16LanesPerReg;
    uint32_t f32LanesPerReg = kF32LanesPerReg;
    uint32_t packedBytesPerWindow = kPackedBytesPerWindow;
    MaskReg preg_b16 = CreatePredicate<half>(b16LanesPerReg);
    MaskReg preg_f32 = CreatePredicate<float>(f32LanesPerReg);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(packedBytesPerWindow);
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    RegTensor<half> v_input_0, v_input_1;
    vector_bf16 v_scaling_bf16;
    vector_f32 v_scaling_f32, v_mod_even, v_mod_odd;
    vector_s32 v_even_code, v_odd_code;
    vector_u8 v_pair01, v_pair23, v_output, v_scratch;

    vlds(v_input_0, v_input_1, srcPtr, window * kElementsPerWindow, DINTLV_B16);
    vlds((vector_u16&)v_scaling_bf16, (__ubuf__ uint16_t*)scalingPtr, 8 * window, E2B_B16);
    vcvt(v_scaling_f32, v_scaling_bf16, preg_all_b16, PART_EVEN);

    vcvt(v_mod_even, v_input_0, preg_b16, PART_EVEN);
    vcvt(v_mod_odd, v_input_1, preg_b16, PART_EVEN);
    vmul(v_mod_even, v_mod_even, v_scaling_f32, preg_f32, MODE_ZEROING);
    vmul(v_mod_odd, v_mod_odd, v_scaling_f32, preg_f32, MODE_ZEROING);
    CalcE2M1SignedCodeI32(v_even_code, v_mod_even, preg_f32);
    CalcE2M1SignedCodeI32(v_odd_code, v_mod_odd, preg_f32);
    PackE2M1SignedCodeBytes(v_pair01, v_even_code, v_odd_code, packIndex, preg_f32);

    vcvt(v_mod_even, v_input_0, preg_b16, PART_ODD);
    vcvt(v_mod_odd, v_input_1, preg_b16, PART_ODD);
    vmul(v_mod_even, v_mod_even, v_scaling_f32, preg_f32, MODE_ZEROING);
    vmul(v_mod_odd, v_mod_odd, v_scaling_f32, preg_f32, MODE_ZEROING);
    CalcE2M1SignedCodeI32(v_even_code, v_mod_even, preg_f32);
    CalcE2M1SignedCodeI32(v_odd_code, v_mod_odd, preg_f32);
    PackE2M1SignedCodeBytes(v_pair23, v_even_code, v_odd_code, packIndex, preg_f32);

    vintlv(
        (RegTensor<uint8_t>&)v_output, (RegTensor<uint8_t>&)v_scratch, (RegTensor<uint8_t>&)v_pair01,
        (RegTensor<uint8_t>&)v_pair23);
    vsts((RegTensor<uint8_t>&)v_output, (__ubuf__ uint8_t*)dstPtr, window * kPackedBytesPerWindow, NORM_B8, preg_b8);
}

// Quantize the tail groups (1..7) of a BF16->MXFP4 E2M1 tile after the full
// 8-group windows have been processed.
PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Bf16_Tail(
    __ubuf__ bfloat16_t* srcTailPtr, __ubuf__ bfloat16_t* scalingTailPtr, __ubuf__ uint8_t* dstWritePtr,
    uint32_t tailGroups, MaskReg& preg_b16_group, vector_u8& v_idx)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    UnalignReg ureg_out;
    for (uint32_t group = 0; group < tailGroups; ++group) {
        vector_bf16 v_input;
        vector_bf16 v_scale;
        vector_bf16 v_scaled;
        vector_f4e2m1x2 v_output_p0, v_output;

        vlds(v_input, srcTailPtr, group * kGroupSize, NORM);
        vlds((vector_u16&)v_scale, (__ubuf__ uint16_t*)scalingTailPtr, group, BRC_B16);
        vmul(v_scaled, v_input, v_scale, preg_b16_group, MODE_ZEROING);
        SaturateBf16NaNToPosInf((vector_u16&)v_scaled, preg_b16_group);
        vcvt(v_output_p0, v_scaled, preg_b16_group, ROUND_R, PART_P0);
        vselr((RegTensor<uint8_t>&)v_output, (RegTensor<uint8_t>&)v_output_p0, (RegTensor<uint8_t>&)v_idx);
        mem_bar(VST_VST);
        vstus(ureg_out, kPackedBytesPerGroup, (RegTensor<uint8_t>&)v_output, dstWritePtr, POST_UPDATE);
    }
    vstas(ureg_out, dstWritePtr, 0, POST_UPDATE);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Bf16(
    __ubuf__ bfloat16_t* srcPtr, __ubuf__ bfloat16_t* scalingPtr, __ubuf__ uint8_t* dstPtr, uint32_t totalGroups)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    constexpr uint32_t kGroupsPerWindow = 8;
    constexpr uint32_t kElementsPerWindow = kGroupSize * kGroupsPerWindow;
    constexpr uint32_t kPackedBytesPerWindow = kElementsPerWindow / 2;
    constexpr uint32_t kPackedBytesPerHalfWindow = kPackedBytesPerWindow / 2;
    uint32_t groupSize = kGroupSize;
    MaskReg preg_b16_window = pset_b16(PAT_ALL);
    MaskReg preg_b16_group = CreatePredicate<bfloat16_t>(groupSize);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    vector_u8 v_idx;
    vci((RegTensor<int8_t>&)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t>&)v_idx, (RegTensor<int16_t>&)v_idx, (int16_t)4, preg_idx);

    uint32_t windowCount = totalGroups / kGroupsPerWindow;
    for (uint32_t window = 0; window < windowCount; ++window) {
        vector_bf16 v_input_0, v_input_1;
        vector_bf16 v_intlv_0, v_intlv_1;
        vector_bf16 v_scale;
        vector_f4e2m1x2 v_output_0, v_output_1;

        vlds(v_input_0, v_input_1, srcPtr, window * kElementsPerWindow, DINTLV_B16);
        vlds((vector_u16&)v_scale, (__ubuf__ uint16_t*)scalingPtr, window * kGroupsPerWindow, E2B_B16);
        vmul(v_input_0, v_input_0, v_scale, preg_b16_window, MODE_ZEROING);
        vmul(v_input_1, v_input_1, v_scale, preg_b16_window, MODE_ZEROING);
        SaturateBf16NaNToPosInf((vector_u16&)v_input_0, preg_b16_window);
        SaturateBf16NaNToPosInf((vector_u16&)v_input_1, preg_b16_window);
        vintlv(v_intlv_0, v_intlv_1, v_input_0, v_input_1);
        vcvt(v_output_0, v_intlv_0, preg_b16_window, ROUND_R, PART_P0);
        vcvt(v_output_1, v_intlv_1, preg_b16_window, ROUND_R, PART_P0);
        vsts((RegTensor<uint8_t>&)v_output_0, dstPtr, window * kPackedBytesPerWindow, PK4_B32, preg_b16_window);
        vsts(
            (RegTensor<uint8_t>&)v_output_1, dstPtr, window * kPackedBytesPerWindow + kPackedBytesPerHalfWindow,
            PK4_B32, preg_b16_window);
    }

    uint32_t tailGroups = totalGroups - windowCount * kGroupsPerWindow;
    if (tailGroups == 0) {
        return;
    }
    CalcQuantizedFP4E2M1Values_Bf16_Tail(
        srcPtr + windowCount * kElementsPerWindow, scalingPtr + windowCount * kGroupsPerWindow,
        dstPtr + windowCount * kPackedBytesPerWindow, tailGroups, preg_b16_group, v_idx);
}

// Pad a tightly-packed max write pointer up to BLOCK_BYTE alignment and close the
// unaligned store sequence.
template <typename T>
PTO_INTERNAL void PadMaxWriteToAlignment(
    RegTensor<T>& vb16_max, vector_align& ureg_max, __ubuf__ T*& writePtr, uint32_t groupsWritten)
{
    uint32_t alignGroups = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t paddedGroups = CeilDivision(groupsWritten, alignGroups) * alignGroups;
    uint32_t padCount = paddedGroups - groupsWritten;
    if (padCount > 0)
        vstus(ureg_max, padCount, vb16_max, writePtr, POST_UPDATE);
    vstas(ureg_max, writePtr, 0, POST_UPDATE);
}

// Packed 2D AbsReduceMax: rows are read with srcCols stride, but max is written
// tightly as [validRows, validGroupsPerRow]. This keeps max/scaling scratch sized
// by the real tail-axis shape instead of the static tile tail.
template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND_2D_Packed(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned validCols, unsigned srcCols)
{
    RegTensor<T> vb16_max;
    vector_align ureg_max;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = CCE_VL / sizeof(T);
    constexpr uint32_t elements_per_dintlv = 2 * elements_per_vl;
    constexpr uint32_t grps_per_dintlv = elements_per_dintlv / grp_size;
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, grp_size);
    uint32_t elemsPerRow = groupsPerRow * grp_size;
    uint16_t loopNumPerRow = (uint16_t)CeilDivision(elemsPerRow, elements_per_dintlv);
    if (loopNumPerRow == 1) {
        __ubuf__ T* writePtr = maxPtr;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t srcRowOff = (uint32_t)row * srcCols;
            if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
                AbsReduceMax_b16_DintlvWindow<T, false>(srcPtr, srcRowOff, elemsPerRow, vb16_max);
            else
                AbsReduceMax_b16_DintlvWindow<T>(srcPtr, srcRowOff, elemsPerRow, vb16_max);
            uint32_t outCount = groupsPerRow;
            vstus(ureg_max, outCount, vb16_max, writePtr, POST_UPDATE);
        }
        PadMaxWriteToAlignment<T>(vb16_max, ureg_max, writePtr, (uint32_t)validRows * groupsPerRow);
        return;
    }
    __ubuf__ T* writePtr = maxPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t srcRowOff = (uint32_t)row * srcCols;
        for (uint16_t i = 0; i < loopNumPerRow; ++i) {
            uint32_t colOff = (uint32_t)i * elements_per_dintlv;
            uint32_t remaining = (elemsPerRow > colOff) ? (elemsPerRow - colOff) : 0;
            if (remaining > elements_per_dintlv)
                remaining = elements_per_dintlv;
            if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
                AbsReduceMax_b16_DintlvWindow<T, false>(srcPtr, srcRowOff + colOff, remaining, vb16_max);
            else
                AbsReduceMax_b16_DintlvWindow<T>(srcPtr, srcRowOff + colOff, remaining, vb16_max);

            uint32_t grpsWritten = (uint32_t)i * grps_per_dintlv;
            uint32_t grpsRemaining = (groupsPerRow > grpsWritten) ? (groupsPerRow - grpsWritten) : 0;
            uint32_t grpsThisIter = (grpsRemaining > grps_per_dintlv) ? grps_per_dintlv : grpsRemaining;
            vstus(ureg_max, grpsThisIter, vb16_max, writePtr, POST_UPDATE);
        }
    }
    vstas(ureg_max, writePtr, 0, POST_UPDATE);
}

// Zero-pad columns [validCols, StaticCols) of a 16-bit
// source tile at VL-aligned offsets (full-VL vlds -> vsel ->
// vsts). Sub-VL stores at non-VL-aligned offsets are
// unreliable on some hardware revisions. Requires StaticCols
// | elemPerVL. Must be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadColumns_VLAligned(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elemPerVL = CCE_VL / sizeof(T);
    static_assert(
        elemPerVL % StaticCols == 0, "StaticCols must evenly divide "
                                     "elements-per-VL for VL-aligned padding");
    constexpr unsigned rowsPerVL = elemPerVL / StaticCols;

    MaskReg pg_all = TQuantPSetTyped<T>(PAT_ALL);

    // Build a periodic predicate: bit p is set iff (p %
    // StaticCols) < validCols. Row 0 contributes positions
    // [0, validCols).
    uint32_t vc = (uint32_t)validCols;
    MaskReg preg_valid = CreatePredicate<T>(vc);
    for (uint16_t r = 1; r < (uint16_t)rowsPerVL; ++r) {
        uint32_t rangeStart = (uint32_t)(r * StaticCols);
        uint32_t rangeEnd = rangeStart + (uint32_t)validCols;
        MaskReg p_end = CreatePredicate<T>(rangeEnd);
        MaskReg p_start = CreatePredicate<T>(rangeStart);
        MaskReg p_row;
        pnot(p_row, p_start, p_end);
        por(preg_valid, preg_valid, p_row, pg_all);
    }

    RegTensor<T> vreg_zero;
    vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);

    // Write-only: store zeros at padding positions without
    // reading the source. Avoids RMW on MTE2-written UB data
    // which can race on hardware.
    MaskReg preg_pad;
    pxor(preg_pad, pg_all, preg_valid, pg_all);

    uint32_t totalElems = (uint32_t)(validRows * StaticCols);
    uint16_t vlCount = CeilDivision(totalElems, (unsigned)elemPerVL);

    for (uint16_t vi = 0; vi < vlCount; ++vi) {
        vsts(vreg_zero, srcPtr, vi * elemPerVL, NORM_B16, preg_pad);
    }
}

} // namespace pto

#endif // PTO_NPU_COMMON_TQUANT_SHARED_HPP
