/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TOPK_HPP
#define TOPK_HPP

#include <pto/common/constants.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

namespace pto {

// TopK selects the k largest (largest=true) or k smallest (largest=false) elements of each row
// along the last dimension and writes the k values plus their srcIdx entries, SORTED: value
// descending for largest=true, ascending for largest=false. Ties keep the smaller element
// position first (with the default arrange srcIdx this is the smaller index, matching
// torch.topk / stable argsort of the negated values).
//
//   ITER_SELECT  : iterative top-1 select + kill (argmax style), fp32 only, O(k*n); winners
//                  leave in sorted order, parked in 64-block rings and flushed as one dense
//                  vector per 64 results; dstIdx = srcIdx[winner] via vgather2.
//   RADIX_SELECT : direct asc-devkit TopKRaidxSelect port (sorted output):
//                  1) RemapBits once (in-place for sizeof(T)==4 / isReuseSrc; else keyUb[N])
//                  2) ROUNDS of chistv2 on remapped u32 keys (FilterDataAndGivenByte-style)
//                  3) GatherGreaterAndEqual (vsqz GT then EQ) into geVal/geIdx[N]
//                  4) RADIX_SORT first K of GE (desc via ~key pad) + TwiddleOut
//
// UB tmp layout (single tile, reused across phases):
//   ITER_SELECT : [0, 16KiB) index ring, [16KiB, 32KiB) value ring
//   RADIX_SELECT: [0, 1KiB) cum hist, [1KiB) scratch, [2KiB) exclusive u16[256],
//                 [3KiB) optional keyUb[N] when sizeof(T)<4, then pair ping-pong +
//                 digit/idx/sortedIdx radix scratch
//
// Supported dtypes: ITER_SELECT fp32; RADIX_SELECT bf16/fp16/fp32/int32/uint32/int16/uint16/
// int8/uint8 (int64/uint64 not supported yet, static_assert fires).

enum class TopKAlgo { ITER_SELECT, RADIX_SELECT };

constexpr uint32_t TOPK_VL = REPEAT_BYTE / sizeof(float); // 64 lanes
constexpr uint32_t TOPK_CHUNK = TOPK_VL;
constexpr uint32_t TOPK_RING_BLOCKS = TOPK_VL;
constexpr uint32_t TOPK_RING_BYTES = TOPK_RING_BLOCKS * REPEAT_BYTE;
constexpr uint32_t TOPK_ITER_TMP_BYTES = 2 * TOPK_RING_BYTES + 32;

template <typename T>
using TopKKeyT = std::conditional_t<sizeof(T) == 4, uint32_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint8_t>>;

// RADIX tmp (asc-devkit TopKRaidxSelect-style):
//   [0, 512) u16 hist, [512, 1KiB) scratch, [1KiB, 2KiB) excl u16[256],
//   then GE val/idx[geCap], sort ping-pong key/val/pos + digit/idx/sortedIdx.
// geCap = Align(K)+VL slack so EQ early-stop cannot overrun (asc-devkit uses N; we cap at K).
constexpr uint32_t TOPK_TMP_FIXED_BYTES = 3072u;
constexpr uint32_t TOPK_RADIX_U8_VL = REPEAT_BYTE; // 256, same as asc-devkit GetVecLen() for u8
AICORE constexpr uint32_t TopKRadixAlignCount(uint32_t k)
{
    uint32_t pad = (k + TOPK_RADIX_U8_VL - 1u) / TOPK_RADIX_U8_VL * TOPK_RADIX_U8_VL;
    return pad < TOPK_RADIX_U8_VL ? TOPK_RADIX_U8_VL : pad;
}
// GE workspace: Align(K)+VL; full-N GE compact is slower on this sim than GT+PickEq.
AICORE constexpr uint32_t TopKGeCap(uint32_t kPadElems, uint32_t n = 0)
{
    (void)n;
    return TopKRadixAlignCount(kPadElems) + TOPK_CHUNK;
}
// kPadElems: padded K; n for GE buffers (and sizeof(T)<4 RemapBits keyUb).
AICORE constexpr uint32_t TopKTmpBytesFor(uint32_t kPadElems, uint32_t n = 0, uint32_t elemSize = 4)
{
    uint32_t alignCount = TopKRadixAlignCount(kPadElems);
    uint32_t geCap = TopKGeCap(kPadElems, n);
    uint32_t keyBytes = (elemSize < 4u && n > 0u) ? (n * sizeof(uint32_t)) : 0u;
    // GE val+idx (2*4*geCap) + sort key0/1+val1+pos1 (16*align) + digit/idx/sorted (7*align) + fixed
    uint32_t radixBytes = TOPK_TMP_FIXED_BYTES + keyBytes + 8u * geCap + 23u * alignCount + 64u;
    return radixBytes > TOPK_ITER_TMP_BYTES ? radixBytes : TOPK_ITER_TMP_BYTES;
}

constexpr auto TOPK_ST_B8 = std::integral_constant<::DistVST, ::DistVST::DIST_NORM_B8>();
constexpr auto TOPK_ST_B16 = std::integral_constant<::DistVST, ::DistVST::DIST_NORM_B16>();

template <typename T>
struct TopKKeyTraits {
    static constexpr bool kIsFloat =
        std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value;
    static constexpr bool kIsSigned = !kIsFloat && std::is_signed<T>::value;
    static constexpr uint32_t kSignBit = sizeof(T) == 4 ? 0x80000000u : (sizeof(T) == 2 ? 0x8000u : 0x80u);
};

// Widen one chunk of native-width elements into zero-extended u32 lanes 0..63.
template <typename T>
PTO_INTERNAL void TopKWidenU32(RegTensor<uint32_t>& key32, RegTensor<T>& raw, RegTensor<uint16_t>& zero16)
{
    if constexpr (sizeof(T) == 4) {
        key32 = (RegTensor<uint32_t>&)raw;
    } else if constexpr (sizeof(T) == 2) {
        RegTensor<uint32_t> keyHi;
        vintlv((vector_u16&)key32, (vector_u16&)keyHi, (vector_u16&)raw, (vector_u16&)zero16);
    } else {
        RegTensor<uint8_t> spreadLo, spreadHi;
        RegTensor<uint16_t> packLo, packHi;
        vintlv((vector_u8&)spreadLo, (vector_u8&)spreadHi, (vector_u8&)raw, (vector_u8&)zero16);
        vintlv((vector_u16&)packLo, (vector_u16&)packHi, (vector_u16&)spreadLo, (vector_u16&)zero16);
        key32 = (RegTensor<uint32_t>&)packLo;
    }
}

// Compare-key twiddle on zero-extended u32 keys: ascending u32 == ascending value. With
// largest=false the key is fully inverted; sub-32-bit keys keep zero high bits for largest and
// all-one high bits for smallest (the scalar threshold must match, see TopKRadixSelectRow).
//
// Important for sizeof(T)<4 floats: the negative branch uses `~bits` on a zero-extended lane,
// which paints 1s into the high padding. Mask back to the significant width BEFORE the optional
// invert, otherwise largest=false flips those padding bits and reverses value order.
template <bool largest, typename T>
PTO_INTERNAL void TopKTwiddleU32(RegTensor<uint32_t>& key32, MaskReg& pAllB32)
{
    if constexpr (TopKKeyTraits<T>::kIsFloat) {
        constexpr uint32_t kSign = TopKKeyTraits<T>::kSignBit;
        constexpr uint32_t kAll = (sizeof(T) == 4) ? 0xFFFFFFFFu : ((sizeof(T) == 2) ? 0xFFFFu : 0xFFu);
        RegTensor<uint32_t> signMask, allOnes, xorPos, xorMask, zero;
        MaskReg isNeg;
        vbr((vector_u32&)signMask, kSign);
        vbr((vector_u32&)allOnes, kAll);
        vbr((vector_u32&)xorPos, kSign);
        vbr((vector_u32&)zero, 0u);
        vand((vector_u32&)signMask, (vector_u32&)key32, (vector_u32&)signMask, pAllB32);
        vcmp_ne(isNeg, (vector_u32&)signMask, (vector_u32&)zero, pAllB32);
        vsel((vector_u32&)xorMask, (vector_u32&)allOnes, (vector_u32&)xorPos, isNeg);
        vxor((vector_u32&)key32, (vector_u32&)key32, (vector_u32&)xorMask, pAllB32);
    } else if constexpr (TopKKeyTraits<T>::kIsSigned) {
        RegTensor<uint32_t> sign32;
        vbr((vector_u32&)sign32, TopKKeyTraits<T>::kSignBit);
        vxor((vector_u32&)key32, (vector_u32&)key32, (vector_u32&)sign32, pAllB32);
    }
    if constexpr (sizeof(T) < 4) {
        constexpr uint32_t kLow = (sizeof(T) == 2) ? 0xFFFFu : 0xFFu;
        RegTensor<uint32_t> lowMask;
        vbr((vector_u32&)lowMask, kLow);
        vand((vector_u32&)key32, (vector_u32&)key32, (vector_u32&)lowMask, pAllB32);
    }
    if constexpr (!largest) {
        vnot((vector_u32&)key32, (vector_u32&)key32, pAllB32);
    }
}

template <typename T>
PTO_INTERNAL TopKKeyT<T> TopKUntwiddleKey(TopKKeyT<T> keyAsc)
{
    using K = TopKKeyT<T>;
    if constexpr (TopKKeyTraits<T>::kIsFloat) {
        return (keyAsc & (TopKKeyT<T>)TopKKeyTraits<T>::kSignBit) != 0 ?
                   static_cast<K>(keyAsc ^ (TopKKeyT<T>)TopKKeyTraits<T>::kSignBit) :
                   static_cast<K>(~keyAsc);
    } else if constexpr (TopKKeyTraits<T>::kIsSigned) {
        return static_cast<K>(keyAsc ^ (TopKKeyT<T>)TopKKeyTraits<T>::kSignBit);
    } else {
        return keyAsc;
    }
}

// Extract the byte at `shift` of each u32 key into d8 (64 valid lanes at bytes 0..63); dig16
// keeps the digits as u16 lanes for gather indices.
PTO_INTERNAL void TopKExtractDigitU8(
    RegTensor<uint8_t>& d8, RegTensor<uint16_t>& dig16, RegTensor<uint32_t>& key32, int16_t shift,
    RegTensor<uint32_t>& ff32, RegTensor<uint32_t>& zero32, RegTensor<uint16_t>& zero16, MaskReg& pAllB32,
    MaskReg& pAllB16)
{
    RegTensor<uint32_t> shifted, dig;
    RegTensor<uint16_t> hi16;
    RegTensor<uint8_t> d8h;
    vshrs((vector_u32&)shifted, (vector_u32&)key32, shift, pAllB32, MODE_ZEROING);
    vand((vector_u32&)dig, (vector_u32&)shifted, (vector_u32&)ff32, pAllB32);
    vdintlv((vector_u16&)dig16, (vector_u16&)hi16, (vector_u16&)dig, (vector_u16&)zero32);
    vdintlv((vector_u8&)d8, d8h, (vector_u8&)dig16, (vector_u8&)zero16);
}

// RemapBits once: float/int → ascending u32 compare keys (asc-devkit TwiddleIn / isReuseSrc).
// When keyPtr aliases src as u32 (sizeof(T)==4), this is in-place.
template <bool largest, typename T>
PTO_INTERNAL void TopKRemapBits(__ubuf__ T* srcPtr, __ubuf__ uint32_t* keyPtr, uint32_t n)
{
    const uint32_t nChunks = (n + TOPK_CHUNK - 1) / TOPK_CHUNK;
    __VEC_SCOPE__
    {
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint16_t> zero16;
        vbr((vector_u16&)zero16, (uint16_t)0);
        for (uint16_t c = 0; c < (uint16_t)nChunks; ++c) {
            uint32_t off = c * TOPK_CHUNK;
            uint32_t remain = n - off;
            uint32_t valid = remain < TOPK_CHUNK ? remain : TOPK_CHUNK;
            MaskReg pStore = plt_b32(valid, POST_UPDATE);
            RegTensor<T> raw;
            vlds(raw, srcPtr, off, NORM);
            RegTensor<uint32_t> key32;
            TopKWidenU32<T>(key32, raw, zero16);
            TopKTwiddleU32<largest, T>(key32, pAllB32);
            vsts(key32, keyPtr, off, NORM_B32, pStore);
        }
    }
}

// Inverse of RemapBits (asc-devkit TwiddleOut) — restore native value bits from compare key.
template <bool largest, typename T>
PTO_INTERNAL uint32_t TopKTwiddleOutScalar(uint32_t key)
{
    if constexpr (!largest) {
        key = ~key;
    }
    if constexpr (sizeof(T) < 4) {
        key &= (sizeof(T) == 2) ? 0xFFFFu : 0xFFu;
    }
    return static_cast<uint32_t>(TopKUntwiddleKey<T>(static_cast<TopKKeyT<T>>(key)));
}

template <bool largest, typename T>
PTO_INTERNAL void TopKTwiddleOutU32(RegTensor<uint32_t>& key32, MaskReg& pAllB32)
{
    if constexpr (!largest) {
        vnot((vector_u32&)key32, (vector_u32&)key32, pAllB32);
    }
    if constexpr (TopKKeyTraits<T>::kIsFloat) {
        constexpr uint32_t kSign = TopKKeyTraits<T>::kSignBit;
        RegTensor<uint32_t> signMask, zero, xorPos, notKey;
        MaskReg hasSign;
        vbr((vector_u32&)signMask, kSign);
        vbr((vector_u32&)zero, 0u);
        vand((vector_u32&)signMask, (vector_u32&)key32, (vector_u32&)signMask, pAllB32);
        vcmp_ne(hasSign, (vector_u32&)signMask, (vector_u32&)zero, pAllB32);
        vbr((vector_u32&)xorPos, kSign);
        vxor((vector_u32&)xorPos, (vector_u32&)key32, (vector_u32&)xorPos, pAllB32);
        vnot((vector_u32&)notKey, (vector_u32&)key32, pAllB32);
        vsel((vector_u32&)key32, (vector_u32&)xorPos, (vector_u32&)notKey, hasSign);
    } else if constexpr (TopKKeyTraits<T>::kIsSigned) {
        RegTensor<uint32_t> sign32;
        vbr((vector_u32&)sign32, TopKKeyTraits<T>::kSignBit);
        vxor((vector_u32&)key32, (vector_u32&)key32, (vector_u32&)sign32, pAllB32);
    }
}

// Pack 256 u32 keys' digit bytes at `shift` into one u8 VL (asc-devkit ConvertRegToWithShift).
PTO_INTERNAL void TopKRadixExtractDigitU32(
    RegTensor<uint8_t>& digitU8, __ubuf__ uint32_t* keys, uint32_t elemOff, int16_t shift)
{
    MaskReg pAllB32 = pset_b32(PAT_ALL);
    RegTensor<uint32_t> k0, k1, k2, k3, shifted0, shifted1, shifted2, shifted3;
    RegTensor<uint16_t> lo01, hi01, lo23, hi23;
    RegTensor<uint8_t> digHi;
    constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t); // 64
    vlds(k0, keys, elemOff + 0 * VL32, NORM);
    vlds(k1, keys, elemOff + 1 * VL32, NORM);
    vlds(k2, keys, elemOff + 2 * VL32, NORM);
    vlds(k3, keys, elemOff + 3 * VL32, NORM);
    vshrs((vector_u32&)shifted0, (vector_u32&)k0, shift, pAllB32, MODE_ZEROING);
    vshrs((vector_u32&)shifted1, (vector_u32&)k1, shift, pAllB32, MODE_ZEROING);
    vshrs((vector_u32&)shifted2, (vector_u32&)k2, shift, pAllB32, MODE_ZEROING);
    vshrs((vector_u32&)shifted3, (vector_u32&)k3, shift, pAllB32, MODE_ZEROING);
    // GetLowestByte: digit is already in low 8b after shift (asc-devkit ConvertRegToWithShift).
    vdintlv(lo01, hi01, (vector_u16&)shifted0, (vector_u16&)shifted1);
    vdintlv(lo23, hi23, (vector_u16&)shifted2, (vector_u16&)shifted3);
    vdintlv(digitU8, digHi, (vector_u8&)lo01, (vector_u8&)lo23);
}

// Pack 4×b32 compare masks into one b8 mask (asc-devkit TransToB8Mask).
PTO_INTERNAL void TopKTransToB8Mask(MaskReg& dst, MaskReg& m0, MaskReg& m1, MaskReg& m2, MaskReg& m3)
{
    MaskReg t16a, t16b, waste;
    pdintlv_b16(t16a, waste, m0, m1);
    pdintlv_b16(t16b, waste, m2, m3);
    pdintlv_b8(dst, waste, t16a, t16b);
}

// RemapBits + hist round-0 fused for sizeof(T)==4 (asc-devkit: TwiddleIn then MSB chist).
// Twiddle in regs, store remapped keys, pack MSB digit from the same regs — one N pass.
template <bool largest, typename T>
PTO_INTERNAL void TopKRemapAndHist0(__ubuf__ T* srcPtr, __ubuf__ uint32_t* keyPtr, uint32_t n, __ubuf__ uint32_t* cumUb)
{
    static_assert(sizeof(T) == 4, "TopKRemapAndHist0 is b32-only");
    constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
    constexpr uint32_t VL8 = TOPK_RADIX_U8_VL;
    const uint32_t nBlocks = n / VL8;
    const uint32_t nTail = n - nBlocks * VL8;
    constexpr int16_t digitShift = (int16_t)(8 * (sizeof(T) - 1));
    __VEC_SCOPE__
    {
        MaskReg pAllB8 = pset_b8(PAT_ALL);
        MaskReg pAllB16 = pset_b16(PAT_ALL);
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> zero32;
        RegTensor<uint16_t> zero16, h0, h1;
        vbr((vector_u32&)zero32, 0u);
        vbr((vector_u16&)zero16, (uint16_t)0);
        vbr((vector_u16&)h0, (uint16_t)0);
        vbr((vector_u16&)h1, (uint16_t)0);
        for (uint16_t b = 0; b < (uint16_t)nBlocks; ++b) {
            const uint32_t off = b * VL8;
            RegTensor<T> raw0, raw1, raw2, raw3;
            RegTensor<uint32_t> k0, k1, k2, k3, s0, s1, s2, s3;
            RegTensor<uint16_t> lo01, hi01, lo23, hi23;
            RegTensor<uint8_t> d8, digHi;
            vlds(raw0, srcPtr, off + 0 * VL32, NORM);
            vlds(raw1, srcPtr, off + 1 * VL32, NORM);
            vlds(raw2, srcPtr, off + 2 * VL32, NORM);
            vlds(raw3, srcPtr, off + 3 * VL32, NORM);
            TopKWidenU32<T>(k0, raw0, zero16);
            TopKWidenU32<T>(k1, raw1, zero16);
            TopKWidenU32<T>(k2, raw2, zero16);
            TopKWidenU32<T>(k3, raw3, zero16);
            TopKTwiddleU32<largest, T>(k0, pAllB32);
            TopKTwiddleU32<largest, T>(k1, pAllB32);
            TopKTwiddleU32<largest, T>(k2, pAllB32);
            TopKTwiddleU32<largest, T>(k3, pAllB32);
            vsts(k0, keyPtr, off + 0 * VL32, NORM_B32, pAllB32);
            vsts(k1, keyPtr, off + 1 * VL32, NORM_B32, pAllB32);
            vsts(k2, keyPtr, off + 2 * VL32, NORM_B32, pAllB32);
            vsts(k3, keyPtr, off + 3 * VL32, NORM_B32, pAllB32);
            // GetLowestByte-style pack (digit already in low 8b after shift; no &0xFF).
            vshrs((vector_u32&)s0, (vector_u32&)k0, digitShift, pAllB32, MODE_ZEROING);
            vshrs((vector_u32&)s1, (vector_u32&)k1, digitShift, pAllB32, MODE_ZEROING);
            vshrs((vector_u32&)s2, (vector_u32&)k2, digitShift, pAllB32, MODE_ZEROING);
            vshrs((vector_u32&)s3, (vector_u32&)k3, digitShift, pAllB32, MODE_ZEROING);
            vdintlv(lo01, hi01, (vector_u16&)s0, (vector_u16&)s1);
            vdintlv(lo23, hi23, (vector_u16&)s2, (vector_u16&)s3);
            vdintlv(d8, digHi, (vector_u8&)lo01, (vector_u8&)lo23);
            chistv2((vector_u16&)h0, (vector_u8&)d8, pAllB8, Bin_N0);
            chistv2((vector_u16&)h1, (vector_u8&)d8, pAllB8, Bin_N1);
        }
        if (nTail > 0) {
            const uint32_t tailChunks = (nTail + TOPK_CHUNK - 1) / TOPK_CHUNK;
            for (uint16_t c = 0; c < (uint16_t)tailChunks; ++c) {
                uint32_t off = nBlocks * VL8 + c * TOPK_CHUNK;
                uint32_t remain = n - off;
                uint32_t valid = remain < TOPK_CHUNK ? remain : TOPK_CHUNK;
                MaskReg pStore = plt_b32(valid, POST_UPDATE);
                MaskReg filt = plt_b8(valid, POST_UPDATE);
                RegTensor<T> raw;
                RegTensor<uint32_t> key32, s;
                RegTensor<uint16_t> dig16, hi16;
                RegTensor<uint8_t> d8, d8h;
                vlds(raw, srcPtr, off, NORM);
                TopKWidenU32<T>(key32, raw, zero16);
                TopKTwiddleU32<largest, T>(key32, pAllB32);
                vsts(key32, keyPtr, off, NORM_B32, pStore);
                vshrs((vector_u32&)s, (vector_u32&)key32, digitShift, pAllB32, MODE_ZEROING);
                vdintlv((vector_u16&)dig16, (vector_u16&)hi16, (vector_u16&)s, (vector_u16&)zero32);
                vdintlv(d8, d8h, (vector_u8&)dig16, (vector_u8&)zero16);
                chistv2((vector_u16&)h0, (vector_u8&)d8, filt, Bin_N0);
                chistv2((vector_u16&)h1, (vector_u8&)d8, filt, Bin_N1);
            }
        }
        // Store u16 cum hist directly (asc-devkit GenerateAccumulateData).
        __ubuf__ uint16_t* cum16 = (__ubuf__ uint16_t*)cumUb;
        vsts(h0, cum16, 0, TOPK_ST_B16, pAllB16);
        vsts(h1, cum16, 128, TOPK_ST_B16, pAllB16);
    }
}

// Histogram on already-remapped u32 keys (no re-twiddle).
// sizeof(T)==4: 256-wide digit pack + one high-bits prefix filter (asc-devkit
// FilterDataAndGivenByteFromOri). Else: 64-wide ExtractDigitU8 per chunk.
// Accumulates chist into h0/h1, then widens to u32 cum once.
template <typename T, uint32_t R>
PTO_INTERNAL void TopKRadixHistRoundKeys(
    __ubuf__ uint32_t* keyPtr, uint32_t n, uint32_t nChunks, uint32_t selB0, uint32_t selB1, uint32_t selB2,
    __ubuf__ uint32_t* cumUb)
{
    if constexpr (sizeof(T) == 4) {
        constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
        constexpr uint32_t VL8 = TOPK_RADIX_U8_VL;
        const uint32_t nBlocks = n / VL8;
        const uint32_t nTail = n - nBlocks * VL8;
        uint32_t selPrefix = 0;
        if constexpr (R >= 1) {
            selPrefix = selB0;
        }
        if constexpr (R >= 2) {
            selPrefix = (selPrefix << 8) | selB1;
        }
        if constexpr (R >= 3) {
            selPrefix = (selPrefix << 8) | selB2;
        }
        const int16_t digitShift = (int16_t)(8 * (sizeof(T) - 1 - R));
        const int16_t maskShift = (int16_t)(8 * (sizeof(T) - R));
        __VEC_SCOPE__
        {
            MaskReg pAllB8 = pset_b8(PAT_ALL);
            MaskReg pAllB16 = pset_b16(PAT_ALL);
            MaskReg pAllB32 = pset_b32(PAT_ALL);
            RegTensor<uint32_t> zero32, selV;
            RegTensor<uint16_t> zero16, h0, h1;
            vbr((vector_u32&)zero32, 0u);
            vbr((vector_u16&)zero16, (uint16_t)0);
            vbr((vector_u32&)selV, selPrefix);
            vbr((vector_u16&)h0, (uint16_t)0);
            vbr((vector_u16&)h1, (uint16_t)0);
            for (uint16_t b = 0; b < (uint16_t)nBlocks; ++b) {
                const uint32_t off = b * VL8;
                RegTensor<uint32_t> k0, k1, k2, k3, s0, s1, s2, s3;
                RegTensor<uint16_t> lo01, hi01, lo23, hi23;
                RegTensor<uint8_t> d8, digHi;
                vlds(k0, keyPtr, off + 0 * VL32, NORM);
                vlds(k1, keyPtr, off + 1 * VL32, NORM);
                vlds(k2, keyPtr, off + 2 * VL32, NORM);
                vlds(k3, keyPtr, off + 3 * VL32, NORM);
                vshrs((vector_u32&)s0, (vector_u32&)k0, digitShift, pAllB32, MODE_ZEROING);
                vshrs((vector_u32&)s1, (vector_u32&)k1, digitShift, pAllB32, MODE_ZEROING);
                vshrs((vector_u32&)s2, (vector_u32&)k2, digitShift, pAllB32, MODE_ZEROING);
                vshrs((vector_u32&)s3, (vector_u32&)k3, digitShift, pAllB32, MODE_ZEROING);
                vdintlv(lo01, hi01, (vector_u16&)s0, (vector_u16&)s1);
                vdintlv(lo23, hi23, (vector_u16&)s2, (vector_u16&)s3);
                vdintlv(d8, digHi, (vector_u8&)lo01, (vector_u8&)lo23);
                MaskReg filt = pAllB8;
                if constexpr (R > 0) {
                    RegTensor<uint32_t> m0, m1, m2, m3;
                    vshrs((vector_u32&)m0, (vector_u32&)k0, maskShift, pAllB32, MODE_ZEROING);
                    vshrs((vector_u32&)m1, (vector_u32&)k1, maskShift, pAllB32, MODE_ZEROING);
                    vshrs((vector_u32&)m2, (vector_u32&)k2, maskShift, pAllB32, MODE_ZEROING);
                    vshrs((vector_u32&)m3, (vector_u32&)k3, maskShift, pAllB32, MODE_ZEROING);
                    MaskReg eq0, eq1, eq2, eq3, eqm;
                    vcmp_eq(eq0, (vector_u32&)m0, (vector_u32&)selV, pAllB32);
                    vcmp_eq(eq1, (vector_u32&)m1, (vector_u32&)selV, pAllB32);
                    vcmp_eq(eq2, (vector_u32&)m2, (vector_u32&)selV, pAllB32);
                    vcmp_eq(eq3, (vector_u32&)m3, (vector_u32&)selV, pAllB32);
                    TopKTransToB8Mask(eqm, eq0, eq1, eq2, eq3);
                    pand(filt, filt, eqm, pAllB8);
                }
                chistv2((vector_u16&)h0, (vector_u8&)d8, filt, Bin_N0);
                chistv2((vector_u16&)h1, (vector_u8&)d8, filt, Bin_N1);
            }
            // Tail < 256: fall back to 64-wide ExtractDigitU8 (rare for N multiple of 256).
            if (nTail > 0) {
                RegTensor<uint32_t> ff32;
                vbr((vector_u32&)ff32, 0xFFu);
                RegTensor<uint16_t> dig16;
                RegTensor<uint8_t> d8;
                const uint32_t tailChunks = (nTail + TOPK_CHUNK - 1) / TOPK_CHUNK;
                for (uint16_t c = 0; c < (uint16_t)tailChunks; ++c) {
                    uint32_t off = nBlocks * VL8 + c * TOPK_CHUNK;
                    uint32_t remain = n - off;
                    uint32_t valid = remain < TOPK_CHUNK ? remain : TOPK_CHUNK;
                    MaskReg filt = plt_b8(valid, POST_UPDATE);
                    RegTensor<uint32_t> key32;
                    vlds(key32, keyPtr, off, NORM);
                    if constexpr (R > 0) {
                        RegTensor<uint32_t> hi;
                        vshrs((vector_u32&)hi, (vector_u32&)key32, maskShift, pAllB32, MODE_ZEROING);
                        MaskReg eqm32, eqm8, waste;
                        vcmp_eq(eqm32, (vector_u32&)hi, (vector_u32&)selV, pAllB32);
                        // Expand 64-lane b32 mask → low 64 bits of b8 (pair with false).
                        MaskReg z = pset_b32(PAT_ALLF);
                        pdintlv_b16(eqm8, waste, eqm32, z);
                        pdintlv_b8(eqm8, waste, eqm8, z);
                        pand(filt, filt, eqm8, pAllB8);
                    }
                    TopKExtractDigitU8(d8, dig16, key32, digitShift, ff32, zero32, zero16, pAllB32, pAllB16);
                    chistv2((vector_u16&)h0, (vector_u8&)d8, filt, Bin_N0);
                    chistv2((vector_u16&)h1, (vector_u8&)d8, filt, Bin_N1);
                }
            }
            __ubuf__ uint16_t* cum16 = (__ubuf__ uint16_t*)cumUb;
            vsts(h0, cum16, 0, TOPK_ST_B16, pAllB16);
            vsts(h1, cum16, 128, TOPK_ST_B16, pAllB16);
        }
    } else {
        __VEC_SCOPE__
        {
            MaskReg pAllB8 = pset_b8(PAT_ALL);
            MaskReg pAllB16 = pset_b16(PAT_ALL);
            MaskReg pAllB32 = pset_b32(PAT_ALL);
            RegTensor<uint32_t> zero32, ff32, w0, w1, w2, w3;
            RegTensor<uint16_t> zero16, h0, h1, dig16;
            RegTensor<uint8_t> d8;
            vbr((vector_u32&)zero32, 0u);
            vbr((vector_u16&)zero16, (uint16_t)0);
            vbr((vector_u32&)ff32, 0xFFu);
            vbr((vector_u16&)h0, (uint16_t)0);
            vbr((vector_u16&)h1, (uint16_t)0);
            for (uint16_t c = 0; c < (uint16_t)nChunks; ++c) {
                uint32_t off = c * TOPK_CHUNK;
                uint32_t remain = n - off;
                uint32_t valid = remain < TOPK_CHUNK ? remain : TOPK_CHUNK;
                MaskReg filt = plt_b8(valid, POST_UPDATE);
                RegTensor<uint32_t> key32;
                vlds(key32, keyPtr, off, NORM);
                MaskReg eqm;
                if constexpr (R > 0) {
                    RegTensor<uint16_t> sel16;
                    vbr((vector_u16&)sel16, (uint16_t)((selB0 << 8) | selB0));
                    TopKExtractDigitU8(
                        d8, dig16, key32, (int16_t)(8 * (sizeof(T) - 1)), ff32, zero32, zero16, pAllB32, pAllB16);
                    vcmp_eq(eqm, (vector_u8&)d8, (vector_u8&)sel16, pAllB8);
                    pand(filt, filt, eqm, pAllB8);
                }
                if constexpr (R > 1) {
                    RegTensor<uint16_t> sel16;
                    vbr((vector_u16&)sel16, (uint16_t)((selB1 << 8) | selB1));
                    TopKExtractDigitU8(
                        d8, dig16, key32, (int16_t)(8 * (sizeof(T) - 2)), ff32, zero32, zero16, pAllB32, pAllB16);
                    vcmp_eq(eqm, (vector_u8&)d8, (vector_u8&)sel16, pAllB8);
                    pand(filt, filt, eqm, pAllB8);
                }
                if constexpr (R > 2) {
                    RegTensor<uint16_t> sel16;
                    vbr((vector_u16&)sel16, (uint16_t)((selB2 << 8) | selB2));
                    TopKExtractDigitU8(
                        d8, dig16, key32, (int16_t)(8 * (sizeof(T) - 3)), ff32, zero32, zero16, pAllB32, pAllB16);
                    vcmp_eq(eqm, (vector_u8&)d8, (vector_u8&)sel16, pAllB8);
                    pand(filt, filt, eqm, pAllB8);
                }
                int16_t shift = (int16_t)(8 * (sizeof(T) - 1 - R));
                TopKExtractDigitU8(d8, dig16, key32, shift, ff32, zero32, zero16, pAllB32, pAllB16);
                chistv2((vector_u16&)h0, (vector_u8&)d8, filt, Bin_N0);
                chistv2((vector_u16&)h1, (vector_u8&)d8, filt, Bin_N1);
            }
            vintlv((vector_u16&)w0, (vector_u16&)w1, (vector_u16&)h0, (vector_u16&)zero16);
            vintlv((vector_u16&)w2, (vector_u16&)w3, (vector_u16&)h1, (vector_u16&)zero16);
            vsts(w0, cumUb, 0, NORM_B32, pAllB32);
            vsts(w1, cumUb, TOPK_CHUNK, NORM_B32, pAllB32);
            vsts(w2, cumUb, 2 * TOPK_CHUNK, NORM_B32, pAllB32);
            vsts(w3, cumUb, 3 * TOPK_CHUNK, NORM_B32, pAllB32);
        }
    }
}

// vsqz GatherMask of key > tKey (gtOnly) or key >= tKey. `cnt` from hist (not sprsts).
// Unroll 4×VL per trip; gather key into pairVal/pairKey in the same VF after mem_bar.
template <bool gtOnly>
PTO_INTERNAL void TopKRadixCompactGather(
    __ubuf__ uint32_t* keyPtr, uint32_t n, uint32_t tKey, uint32_t cnt, __ubuf__ uint32_t* pairVal,
    __ubuf__ uint32_t* pairPos, __ubuf__ uint32_t* pairKey)
{
    if (cnt == 0) {
        return;
    }
    constexpr uint32_t VL32 = TOPK_CHUNK;
    constexpr uint32_t BLK = TOPK_RADIX_U8_VL;
    const uint32_t nBlocks = n / BLK;
    const uint32_t nTail = n - nBlocks * BLK;
    __ubuf__ uint32_t* posDst = pairPos;
    __VEC_SCOPE__
    {
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> tKeyV, addOff, addBlk;
        vbr((vector_u32&)tKeyV, tKey);
        vbr((vector_u32&)addOff, VL32);
        vbr((vector_u32&)addBlk, BLK);
        RegTensor<int32_t> index;
        vci(index, (int32_t)0, INC_ORDER);
        UnalignReg uPos;
        sprclr(SPR_AR);
        for (uint16_t b = 0; b < (uint16_t)nBlocks; ++b) {
            RegTensor<uint32_t> k0, k1, k2, k3, sq0, sq1, sq2, sq3;
            RegTensor<int32_t> i0, i1, i2, i3;
            MaskReg s0, s1, s2, s3;
            i0 = index;
            vadd((vector_u32&)i1, (vector_u32&)i0, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            vadd((vector_u32&)i2, (vector_u32&)i1, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            vadd((vector_u32&)i3, (vector_u32&)i2, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            vlds(k0, keyPtr, b * BLK + 0 * VL32, NORM);
            vlds(k1, keyPtr, b * BLK + 1 * VL32, NORM);
            vlds(k2, keyPtr, b * BLK + 2 * VL32, NORM);
            vlds(k3, keyPtr, b * BLK + 3 * VL32, NORM);
            vcmp_gt(s0, (vector_u32&)k0, (vector_u32&)tKeyV, pAllB32);
            vcmp_gt(s1, (vector_u32&)k1, (vector_u32&)tKeyV, pAllB32);
            vcmp_gt(s2, (vector_u32&)k2, (vector_u32&)tKeyV, pAllB32);
            vcmp_gt(s3, (vector_u32&)k3, (vector_u32&)tKeyV, pAllB32);
            if constexpr (!gtOnly) {
                MaskReg e0, e1, e2, e3;
                vcmp_eq(e0, (vector_u32&)k0, (vector_u32&)tKeyV, pAllB32);
                vcmp_eq(e1, (vector_u32&)k1, (vector_u32&)tKeyV, pAllB32);
                vcmp_eq(e2, (vector_u32&)k2, (vector_u32&)tKeyV, pAllB32);
                vcmp_eq(e3, (vector_u32&)k3, (vector_u32&)tKeyV, pAllB32);
                por(s0, s0, e0, pAllB32);
                por(s1, s1, e1, pAllB32);
                por(s2, s2, e2, pAllB32);
                por(s3, s3, e3, pAllB32);
            }
            vsqz(sq0, (vector_u32&)i0, s0, MODE_STORED);
            vstur(uPos, sq0, posDst, POST_UPDATE);
            vsqz(sq1, (vector_u32&)i1, s1, MODE_STORED);
            vstur(uPos, sq1, posDst, POST_UPDATE);
            vsqz(sq2, (vector_u32&)i2, s2, MODE_STORED);
            vstur(uPos, sq2, posDst, POST_UPDATE);
            vsqz(sq3, (vector_u32&)i3, s3, MODE_STORED);
            vstur(uPos, sq3, posDst, POST_UPDATE);
            vadd((vector_u32&)index, (vector_u32&)index, (vector_u32&)addBlk, pAllB32, MODE_ZEROING);
        }
        const uint32_t tailChunks = (nTail + VL32 - 1) / VL32;
        for (uint16_t c = 0; c < (uint16_t)tailChunks; ++c) {
            uint32_t off = nBlocks * BLK + c * VL32;
            uint32_t remain = n - off;
            uint32_t valid = remain < VL32 ? remain : VL32;
            MaskReg pLoad = plt_b32(valid, POST_UPDATE);
            RegTensor<uint32_t> key32, sqzPos;
            MaskReg sel;
            vlds(key32, keyPtr, off, NORM);
            vcmp_gt(sel, (vector_u32&)key32, (vector_u32&)tKeyV, pLoad);
            if constexpr (!gtOnly) {
                MaskReg eq;
                vcmp_eq(eq, (vector_u32&)key32, (vector_u32&)tKeyV, pLoad);
                por(sel, sel, eq, pAllB32);
            }
            vsqz(sqzPos, (vector_u32&)index, sel, MODE_STORED);
            vstur(uPos, sqzPos, posDst, POST_UPDATE);
            vadd((vector_u32&)index, (vector_u32&)index, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
        }
        vstar(uPos, posDst);
        mem_bar(VST_VLD);
        const uint16_t rep = (uint16_t)((cnt + VL32 - 1u) / VL32);
        uint32_t remain = cnt;
        for (uint16_t i = 0; i < rep; ++i) {
            uint32_t valid = remain < VL32 ? remain : VL32;
            remain -= valid;
            MaskReg p = plt_b32(valid, POST_UPDATE);
            RegTensor<uint32_t> idx, key;
            vlds(idx, pairPos, i * VL32, NORM);
            vgather2(key, keyPtr, idx, p);
            vsts(key, pairKey, i * VL32, NORM_B32, p);
            vsts(key, pairVal, i * VL32, NORM_B32, p);
        }
    }
}

// Copy sorted GT group: TwiddleOut remapped keys → native values; dstIdx = srcIdx[pos].
template <bool largest, typename T>
PTO_INTERNAL void TopKCopySortedGT(
    __ubuf__ uint32_t* pairVal, __ubuf__ uint32_t* pairPos, uint32_t cnt, __ubuf__ T* valPtr, __ubuf__ int32_t* idxPtr,
    __ubuf__ uint32_t* srcIdxPtr)
{
    if constexpr (sizeof(T) == 4) {
        constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
        const uint16_t rep = (uint16_t)((cnt + VL32 - 1u) / VL32);
        __VEC_SCOPE__
        {
            MaskReg pAllB32 = pset_b32(PAT_ALL);
            uint32_t remain = cnt;
            for (uint16_t i = 0; i < rep; ++i) {
                uint32_t valid = remain < VL32 ? remain : VL32;
                remain -= valid;
                MaskReg p = plt_b32(valid, POST_UPDATE);
                RegTensor<uint32_t> key, pos, outIdx;
                vlds(key, pairVal, i * VL32, NORM);
                TopKTwiddleOutU32<largest, T>(key, pAllB32);
                vsts(key, (__ubuf__ uint32_t*)valPtr, i * VL32, NORM_B32, p);
                vlds(pos, pairPos, i * VL32, NORM);
                vgather2(outIdx, srcIdxPtr, pos, p);
                vsts(outIdx, (__ubuf__ uint32_t*)idxPtr, i * VL32, NORM_B32, p);
            }
        }
    } else {
        for (uint32_t j = 0; j < cnt; ++j) {
            const uint32_t bits = TopKTwiddleOutScalar<largest, T>(pairVal[j]);
            if constexpr (sizeof(T) == 2) {
                ((__ubuf__ uint16_t*)valPtr)[j] = (uint16_t)bits;
            } else {
                ((__ubuf__ uint8_t*)valPtr)[j] = (uint8_t)bits;
            }
            idxPtr[j] = (int32_t)srcIdxPtr[pairPos[j]];
        }
    }
}

// EQ fill on remapped keys: each pick is one argmin sweep; kill with tKey^1 in key domain.
// 256-wide (4×VL) when n allows — fewer loop trips than 64-wide chunks.
template <bool largest, typename T>
PTO_INTERNAL void TopKRadixPickEqSweep(
    __ubuf__ uint32_t* keyPtr, uint32_t n, uint32_t nChunks, __ubuf__ uint32_t* scratchUb)
{
    (void)nChunks;
    const uint32_t tKey = scratchUb[256];
    constexpr uint32_t VL32 = TOPK_CHUNK;
    constexpr uint32_t BLK = TOPK_RADIX_U8_VL; // 256
    const uint32_t nBlocks = n / BLK;
    const uint32_t nTail = n - nBlocks * BLK;
    __VEC_SCOPE__
    {
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> Tb, maxU, addVl;
        vbr((vector_u32&)Tb, tKey);
        vbr((vector_u32&)maxU, 0xFFFFFFFFu);
        vbr((vector_u32&)addVl, VL32);
        RegTensor<uint32_t> acc;
        vbr((vector_u32&)acc, 0xFFFFFFFFu);
        for (uint16_t b = 0; b < (uint16_t)nBlocks; ++b) {
            const uint32_t off = b * BLK;
            RegTensor<uint32_t> k0, k1, k2, k3, sel0, sel1, sel2, sel3;
            RegTensor<int32_t> c0, c1, c2, c3;
            MaskReg eq0, eq1, eq2, eq3;
            vlds(k0, keyPtr, off + 0 * VL32, NORM);
            vlds(k1, keyPtr, off + 1 * VL32, NORM);
            vlds(k2, keyPtr, off + 2 * VL32, NORM);
            vlds(k3, keyPtr, off + 3 * VL32, NORM);
            vci(c0, (int32_t)(off + 0 * VL32), INC_ORDER);
            vadd((vector_u32&)c1, (vector_u32&)c0, (vector_u32&)addVl, pAllB32, MODE_ZEROING);
            vadd((vector_u32&)c2, (vector_u32&)c1, (vector_u32&)addVl, pAllB32, MODE_ZEROING);
            vadd((vector_u32&)c3, (vector_u32&)c2, (vector_u32&)addVl, pAllB32, MODE_ZEROING);
            vcmp_eq(eq0, (vector_u32&)k0, (vector_u32&)Tb, pAllB32);
            vcmp_eq(eq1, (vector_u32&)k1, (vector_u32&)Tb, pAllB32);
            vcmp_eq(eq2, (vector_u32&)k2, (vector_u32&)Tb, pAllB32);
            vcmp_eq(eq3, (vector_u32&)k3, (vector_u32&)Tb, pAllB32);
            vsel((vector_u32&)sel0, (vector_u32&)c0, (vector_u32&)maxU, eq0);
            vsel((vector_u32&)sel1, (vector_u32&)c1, (vector_u32&)maxU, eq1);
            vsel((vector_u32&)sel2, (vector_u32&)c2, (vector_u32&)maxU, eq2);
            vsel((vector_u32&)sel3, (vector_u32&)c3, (vector_u32&)maxU, eq3);
            vmin((vector_u32&)acc, (vector_u32&)acc, (vector_u32&)sel0, pAllB32, MODE_ZEROING);
            vmin((vector_u32&)acc, (vector_u32&)acc, (vector_u32&)sel1, pAllB32, MODE_ZEROING);
            vmin((vector_u32&)acc, (vector_u32&)acc, (vector_u32&)sel2, pAllB32, MODE_ZEROING);
            vmin((vector_u32&)acc, (vector_u32&)acc, (vector_u32&)sel3, pAllB32, MODE_ZEROING);
        }
        const uint32_t tailChunks = (nTail + VL32 - 1) / VL32;
        for (uint16_t c = 0; c < (uint16_t)tailChunks; ++c) {
            uint32_t off = nBlocks * BLK + c * VL32;
            uint32_t remain = n - off;
            uint32_t valid = remain < VL32 ? remain : VL32;
            MaskReg pLoad = plt_b32(valid, POST_UPDATE);
            RegTensor<uint32_t> key32, sel;
            RegTensor<int32_t> cand;
            MaskReg eq;
            vlds(key32, keyPtr, off, NORM);
            vcmp_eq(eq, (vector_u32&)key32, (vector_u32&)Tb, pAllB32);
            pand(eq, eq, pLoad, pAllB32);
            vci(cand, (int32_t)off, INC_ORDER);
            vsel((vector_u32&)sel, (vector_u32&)cand, (vector_u32&)maxU, eq);
            vmin((vector_u32&)acc, (vector_u32&)acc, (vector_u32&)sel, pAllB32, MODE_ZEROING);
        }
        RegTensor<uint32_t> winV;
        vcmin((vector_u32&)winV, (vector_u32&)acc, pAllB32, MODE_ZEROING);
        vsts((vector_u32&)winV, scratchUb, 0, NORM_B32, pAllB32);
    }
}

template <bool largest, typename T>
PTO_INTERNAL void TopKRadixPickEq(
    __ubuf__ uint32_t* keyPtr, uint32_t n, uint32_t nChunks, uint32_t tKey, uint32_t m, uint32_t outOff,
    __ubuf__ T* valPtr, __ubuf__ int32_t* idxPtr, __ubuf__ uint32_t* scratchUb, __ubuf__ uint32_t* srcIdxPtr)
{
    scratchUb[256] = tKey;
    uint32_t prevWin = 0xFFFFFFFFu;
    for (uint32_t j = 0; j < m; ++j) {
        if (prevWin != 0xFFFFFFFFu) {
            keyPtr[prevWin] = tKey ^ 1u;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }
        TopKRadixPickEqSweep<largest, T>(keyPtr, n, nChunks, scratchUb);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        prevWin = *scratchUb;
        if (prevWin == 0xFFFFFFFFu) {
            break;
        }
        idxPtr[outOff + j] = (int32_t)srcIdxPtr[prevWin];
        const uint32_t bits = TopKTwiddleOutScalar<largest, T>(keyPtr[prevWin]);
        if constexpr (sizeof(T) == 4) {
            ((__ubuf__ uint32_t*)valPtr)[outOff + j] = bits;
        } else if constexpr (sizeof(T) == 2) {
            ((__ubuf__ uint16_t*)valPtr)[outOff + j] = (uint16_t)bits;
        } else {
            ((__ubuf__ uint8_t*)valPtr)[outOff + j] = (uint8_t)bits;
        }
    }
}

// === asc-devkit sort/sort_impl.h RADIX_SORT port (SIMD VF) =====================
// GetExclusiveSum(InvertDigit) → EightBitsSort → LocalSort → gather permute of (val,pos).

// asc-devkit GetExclusiveSum: FREQUENCY+ACCUMULATE over all digit VLs → exclusive = cum - freq.
// InvertDigit: digit^=0xFF → ascending radix on inverted digits = descending on remapped keys
// (replaces the old pad+NOT key buffer; constexpr so no scalar if-in-VF).
template <bool InvertDigit = false>
PTO_INTERNAL void TopKRadixGetExclusiveSum(
    __ubuf__ uint32_t* keys, __ubuf__ uint8_t* digitUb, __ubuf__ uint8_t* digitUbCopy, __ubuf__ uint16_t* exclUb,
    uint32_t cnt, int16_t shift)
{
    const uint16_t repeatTime = (uint16_t)((cnt + TOPK_RADIX_U8_VL - 1u) / TOPK_RADIX_U8_VL);
    __VEC_SCOPE__
    {
        MaskReg pAllB8 = pset_b8(PAT_ALL);
        MaskReg pAllB16 = pset_b16(PAT_ALL);
        RegTensor<uint16_t> dist0, dist1, cum0, cum1, excl0, excl1;
        vbr((vector_u16&)dist0, (uint16_t)0);
        vbr((vector_u16&)dist1, (uint16_t)0);
        vbr((vector_u16&)cum0, (uint16_t)0);
        vbr((vector_u16&)cum1, (uint16_t)0);
        uint32_t remain = cnt;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            uint32_t valid = remain < TOPK_RADIX_U8_VL ? remain : TOPK_RADIX_U8_VL;
            remain -= valid;
            MaskReg pDigit = plt_b8(valid, POST_UPDATE);
            RegTensor<uint8_t> digitU8;
            TopKRadixExtractDigitU32(digitU8, keys, i * TOPK_RADIX_U8_VL, shift);
            if constexpr (InvertDigit) {
                vnot((vector_u8&)digitU8, (vector_u8&)digitU8, pAllB8);
            }
            // Keep digitUbCopy pristine for EightBitsSort Gather source across all bit passes.
            vsts(digitU8, digitUb, i * TOPK_RADIX_U8_VL, TOPK_ST_B8, pDigit);
            vsts(digitU8, digitUbCopy, i * TOPK_RADIX_U8_VL, TOPK_ST_B8, pDigit);
            dhistv2((vector_u16&)dist0, (vector_u8&)digitU8, pDigit, Bin_N0);
            dhistv2((vector_u16&)dist1, (vector_u8&)digitU8, pDigit, Bin_N1);
            chistv2((vector_u16&)cum0, (vector_u8&)digitU8, pDigit, Bin_N0);
            chistv2((vector_u16&)cum1, (vector_u8&)digitU8, pDigit, Bin_N1);
        }
        vsub((vector_u16&)excl0, (vector_u16&)cum0, (vector_u16&)dist0, pAllB16);
        vsub((vector_u16&)excl1, (vector_u16&)cum1, (vector_u16&)dist1, pAllB16);
        vsts(excl0, exclUb, 0, TOPK_ST_B16, pAllB16);
        vsts(excl1, exclUb, 128, TOPK_ST_B16, pAllB16);
    }
}

// asc-devkit EightBitsSort: 4× 2-bit stable sorts per u8 VL via vsqz compress buckets.
// Indices rewritten in-place via unalign stores into idxU8 (sort_impl.h). digitsBack holds
// pristine digits from GetExclusiveSum and is Gather-only — never overwritten here.
PTO_INTERNAL void TopKRadixEightBitsSort(
    __ubuf__ uint8_t* digits, __ubuf__ uint8_t* digitsBack, __ubuf__ uint8_t* idxU8, __ubuf__ uint16_t* idxU16,
    uint32_t cnt)
{
    const uint16_t repeatTime = (uint16_t)((cnt + TOPK_RADIX_U8_VL - 1u) / TOPK_RADIX_U8_VL);
    __VEC_SCOPE__
    {
        MaskReg pAllB16 = pset_b16(PAT_ALL);
        RegTensor<int8_t> bucketIndex;
        vci(bucketIndex, (int8_t)0, INC_ORDER);
        uint32_t remainInit = cnt;
        for (uint16_t j = 0; j < repeatTime; ++j) {
            uint32_t valid = remainInit < TOPK_RADIX_U8_VL ? remainInit : TOPK_RADIX_U8_VL;
            remainInit -= valid;
            MaskReg p = plt_b8(valid, POST_UPDATE);
            vsts((RegTensor<uint8_t>&)bucketIndex, idxU8, j * TOPK_RADIX_U8_VL, TOPK_ST_B8, p);
        }
        mem_bar(VST_VLD);
        sprclr(SPR_AR);

        RegTensor<uint8_t> last2BitsMask, s0, s1, s2, s3;
        vbr((vector_u8&)last2BitsMask, (uint8_t)3);
        vbr((vector_u8&)s0, (uint8_t)0);
        vbr((vector_u8&)s1, (uint8_t)1);
        vbr((vector_u8&)s2, (uint8_t)2);
        vbr((vector_u8&)s3, (uint8_t)3);

        for (uint16_t bitPass = 0; bitPass < 4; ++bitPass) {
            UnalignReg ureg;
            sprclr(SPR_AR);
            // Reset write ptr each bit pass (asc-devkit intends in-place into inputIndexAddr).
            __ubuf__ uint8_t* compactDst = idxU8;
            uint32_t remain = cnt;
            const uint16_t bitShift = (uint16_t)(bitPass * 2);
            for (uint16_t j = 0; j < repeatTime; ++j) {
                uint32_t valid = remain < TOPK_RADIX_U8_VL ? remain : TOPK_RADIX_U8_VL;
                remain -= valid;
                MaskReg p = plt_b8(valid, POST_UPDATE);
                RegTensor<uint8_t> srcReg, indexReg, shift2, gather0, gather1, gather2, gather3;
                // Bit compare uses *current* digit order (digits).
                vlds(srcReg, digits, j * TOPK_RADIX_U8_VL, NORM);
                vlds(indexReg, idxU8, j * TOPK_RADIX_U8_VL, NORM);
                vshrs((vector_u8&)shift2, (vector_u8&)srcReg, bitShift, p, MODE_ZEROING);
                vand((vector_u8&)shift2, (vector_u8&)shift2, (vector_u8&)last2BitsMask, p);
                MaskReg m0, m1, m2, m3;
                vcmp_eq(m0, (vector_u8&)shift2, (vector_u8&)s0, p);
                vcmp_eq(m1, (vector_u8&)shift2, (vector_u8&)s1, p);
                vcmp_eq(m2, (vector_u8&)shift2, (vector_u8&)s2, p);
                vcmp_eq(m3, (vector_u8&)shift2, (vector_u8&)s3, p);
                vsqz(gather0, (vector_u8&)indexReg, m0, MODE_STORED);
                vstur(ureg, gather0, compactDst, POST_UPDATE);
                vsqz(gather1, (vector_u8&)indexReg, m1, MODE_STORED);
                vstur(ureg, gather1, compactDst, POST_UPDATE);
                vsqz(gather2, (vector_u8&)indexReg, m2, MODE_STORED);
                vstur(ureg, gather2, compactDst, POST_UPDATE);
                vsqz(gather3, (vector_u8&)indexReg, m3, MODE_STORED);
                vstur(ureg, gather3, compactDst, POST_UPDATE);
            }
            vstar(ureg, compactDst);
            mem_bar(VST_VLD);
            // Gather ALWAYS from pristine digitsBack; write reordered digits into digits.
            remain = cnt;
            for (uint16_t j = 0; j < repeatTime; ++j) {
                uint32_t valid = remain < TOPK_RADIX_U8_VL ? remain : TOPK_RADIX_U8_VL;
                remain -= valid;
                MaskReg p = plt_b8(valid, POST_UPDATE);
                RegTensor<uint8_t> srcReg, tmpIndexReg, selectData;
                vlds(srcReg, digitsBack, j * TOPK_RADIX_U8_VL, NORM);
                vlds(tmpIndexReg, idxU8, j * TOPK_RADIX_U8_VL, NORM);
                vselr(selectData, srcReg, tmpIndexReg);
                vsts(selectData, digits, j * TOPK_RADIX_U8_VL, TOPK_ST_B8, p);
            }
            mem_bar(VST_VLD);
            sprclr(SPR_AR);
        }

        // Widen per-VL u8 indices to u16 global indices (asc-devkit end of EightBitsSort).
        RegTensor<uint8_t> zeroU8;
        RegTensor<uint16_t> offset16;
        vbr((vector_u8&)zeroU8, (uint8_t)0);
        vbr((vector_u16&)offset16, (uint16_t)0);
        uint32_t remain = cnt;
        for (uint16_t j = 0; j < repeatTime; ++j) {
            uint32_t valid = remain < TOPK_RADIX_U8_VL ? remain : TOPK_RADIX_U8_VL;
            remain -= valid;
            uint32_t valid0 = valid < 128u ? valid : 128u;
            uint32_t valid1 = valid > 128u ? valid - 128u : 0u;
            MaskReg p0 = plt_b16(valid0, POST_UPDATE);
            MaskReg p1 = plt_b16(valid1, POST_UPDATE);
            RegTensor<uint8_t> sortedRes;
            RegTensor<uint16_t> sortedVal0, sortedVal1;
            vlds(sortedRes, idxU8, j * TOPK_RADIX_U8_VL, NORM);
            vintlv((vector_u8&)sortedVal0, (vector_u8&)sortedVal1, sortedRes, zeroU8);
            vadd((vector_u16&)sortedVal0, (vector_u16&)sortedVal0, (vector_u16&)offset16, pAllB16, MODE_ZEROING);
            vadd((vector_u16&)sortedVal1, (vector_u16&)sortedVal1, (vector_u16&)offset16, pAllB16, MODE_ZEROING);
            vsts(sortedVal0, idxU16, j * TOPK_RADIX_U8_VL + 0, TOPK_ST_B16, p0);
            vsts(sortedVal1, idxU16, j * TOPK_RADIX_U8_VL + 128, TOPK_ST_B16, p1);
            RegTensor<uint16_t> addVl;
            vbr((vector_u16&)addVl, (uint16_t)TOPK_RADIX_U8_VL);
            vadd((vector_u16&)offset16, (vector_u16&)offset16, (vector_u16&)addVl, pAllB16, MODE_ZEROING);
        }
    }
}

// asc-devkit LocalSort: rank = exclusive[digit] + lane - localExclusive[digit]; scatter idx.
PTO_INTERNAL void TopKRadixLocalSort(
    __ubuf__ uint8_t* digits, __ubuf__ uint16_t* idxU16, __ubuf__ uint16_t* exclUb, __ubuf__ uint32_t* sortedIdx,
    uint32_t cnt)
{
    const uint16_t repeatTime = (uint16_t)((cnt + TOPK_RADIX_U8_VL - 1u) / TOPK_RADIX_U8_VL);
    __VEC_SCOPE__
    {
        MaskReg pAllB16 = pset_b16(PAT_ALL);
        MaskReg pAllB8 = pset_b8(PAT_ALL);
        RegTensor<int16_t> lane0, lane1;
        vci(lane0, (int16_t)0, INC_ORDER);
        vci(lane1, (int16_t)128, INC_ORDER);
        RegTensor<uint16_t> bucketOff0, bucketOff1;
        vlds(bucketOff0, exclUb, 0, NORM);
        vlds(bucketOff1, exclUb, 128, NORM);
        RegTensor<uint8_t> zeroU8;
        vbr((vector_u8&)zeroU8, (uint8_t)0);

        uint32_t remain = cnt;
        uint32_t remainU8 = cnt;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            uint32_t validU8 = remainU8 < TOPK_RADIX_U8_VL ? remainU8 : TOPK_RADIX_U8_VL;
            remainU8 -= validU8;
            MaskReg pB8 = plt_b8(validU8, POST_UPDATE);
            RegTensor<uint8_t> key;
            RegTensor<uint16_t> index0, index1;
            vlds(key, digits, i * TOPK_RADIX_U8_VL, NORM);
            vlds(index0, idxU16, i * TOPK_RADIX_U8_VL + 0, NORM);
            vlds(index1, idxU16, i * TOPK_RADIX_U8_VL + 128, NORM);

            RegTensor<uint8_t> bucketOffLow, bucketOffHigh;
            vdintlv(bucketOffLow, bucketOffHigh, (vector_u8&)bucketOff0, (vector_u8&)bucketOff1);

            RegTensor<uint16_t> dist0, dist1, cum0, cum1;
            vbr((vector_u16&)dist0, (uint16_t)0);
            vbr((vector_u16&)dist1, (uint16_t)0);
            vbr((vector_u16&)cum0, (uint16_t)0);
            vbr((vector_u16&)cum1, (uint16_t)0);
            dhistv2((vector_u16&)dist0, (vector_u8&)key, pB8, Bin_N0);
            dhistv2((vector_u16&)dist1, (vector_u8&)key, pB8, Bin_N1);
            chistv2((vector_u16&)cum0, (vector_u8&)key, pB8, Bin_N0);
            chistv2((vector_u16&)cum1, (vector_u8&)key, pB8, Bin_N1);

            RegTensor<uint16_t> localOff0, localOff1;
            vsub((vector_u16&)localOff0, (vector_u16&)cum0, (vector_u16&)dist0, pAllB16);
            vsub((vector_u16&)localOff1, (vector_u16&)cum1, (vector_u16&)dist1, pAllB16);

            RegTensor<uint8_t> localOffLow, localOffHigh;
            vdintlv(localOffLow, localOffHigh, (vector_u8&)localOff0, (vector_u8&)localOff1);

            RegTensor<uint8_t> boKeyLow, boKeyHigh, loKeyLow;
            vselr(boKeyLow, bucketOffLow, key);
            vselr(boKeyHigh, bucketOffHigh, key);
            vselr(loKeyLow, localOffLow, key);

            RegTensor<uint8_t> boKey0, boKey1, loKey0, loKey1, z;
            vintlv(boKey0, boKey1, boKeyLow, boKeyHigh);
            vintlv(loKey0, loKey1, loKeyLow, zeroU8);

            RegTensor<uint16_t> rank0, rank1;
            vadd((vector_u16&)rank0, (vector_u16&)boKey0, (vector_u16&)lane0, pAllB16, MODE_ZEROING);
            vadd((vector_u16&)rank1, (vector_u16&)boKey1, (vector_u16&)lane1, pAllB16, MODE_ZEROING);
            vsub((vector_u16&)rank0, (vector_u16&)rank0, (vector_u16&)loKey0, pAllB16);
            vsub((vector_u16&)rank1, (vector_u16&)rank1, (vector_u16&)loKey1, pAllB16);

            RegTensor<uint32_t> rankU0, rankU1, rankU2, rankU3;
            RegTensor<uint32_t> idxU0, idxU1, idxU2, idxU3;
            RegTensor<uint16_t> z16;
            vbr((vector_u16&)z16, (uint16_t)0);
            vintlv((vector_u16&)rankU0, (vector_u16&)rankU1, rank0, z16);
            vintlv((vector_u16&)rankU2, (vector_u16&)rankU3, rank1, z16);
            vintlv((vector_u16&)idxU0, (vector_u16&)idxU1, index0, z16);
            vintlv((vector_u16&)idxU2, (vector_u16&)idxU3, index1, z16);

            uint32_t rem = remain;
            uint32_t n0 = rem < 64u ? rem : 64u;
            rem = rem > 64u ? rem - 64u : 0u;
            uint32_t n1 = rem < 64u ? rem : 64u;
            rem = rem > 64u ? rem - 64u : 0u;
            uint32_t n2 = rem < 64u ? rem : 64u;
            rem = rem > 64u ? rem - 64u : 0u;
            uint32_t n3 = rem < 64u ? rem : 64u;
            rem = rem > 64u ? rem - 64u : 0u;
            MaskReg p0 = plt_b32(n0, POST_UPDATE);
            MaskReg p1 = plt_b32(n1, POST_UPDATE);
            MaskReg p2 = plt_b32(n2, POST_UPDATE);
            MaskReg p3 = plt_b32(n3, POST_UPDATE);
            remain = rem;
            vscatter((vector_u32&)idxU0, sortedIdx, (vector_u32&)rankU0, p0);
            vscatter((vector_u32&)idxU1, sortedIdx, (vector_u32&)rankU1, p1);
            vscatter((vector_u32&)idxU2, sortedIdx, (vector_u32&)rankU2, p2);
            vscatter((vector_u32&)idxU3, sortedIdx, (vector_u32&)rankU3, p3);

            vadd((vector_u16&)bucketOff0, (vector_u16&)bucketOff0, (vector_u16&)dist0, pAllB16, MODE_ZEROING);
            vadd((vector_u16&)bucketOff1, (vector_u16&)bucketOff1, (vector_u16&)dist1, pAllB16, MODE_ZEROING);
        }
    }
}

// Fuse val/pos gathers (no key stream — digits inverted from val for descending).
// R7/R13: static PAT_ALL for full VLs; ≤1 dynamic plt tail.
PTO_INTERNAL void TopKRadixGatherPairU32(
    __ubuf__ uint32_t* sV, __ubuf__ uint32_t* sP, __ubuf__ uint32_t* sortedIdx, __ubuf__ uint32_t* dV,
    __ubuf__ uint32_t* dP, uint32_t cnt)
{
    constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
    const uint32_t fullRep = cnt / VL32;
    const uint32_t tailN = cnt - fullRep * VL32;
    const uint16_t tailRep = (tailN > 0u) ? (uint16_t)1 : (uint16_t)0;
    __VEC_SCOPE__
    {
        MaskReg pAll = pset_b32(PAT_ALL);
        for (uint16_t i = 0; i < (uint16_t)fullRep; ++i) {
            RegTensor<uint32_t> idx, oV, oP;
            vlds(idx, sortedIdx, i * VL32, NORM);
            vgather2(oV, sV, idx, pAll);
            vgather2(oP, sP, idx, pAll);
            vsts(oV, dV, i * VL32, NORM_B32, pAll);
            vsts(oP, dP, i * VL32, NORM_B32, pAll);
        }
        for (uint16_t t = 0; t < tailRep; ++t) {
            uint32_t sreg = tailN;
            MaskReg p = plt_b32(sreg, POST_UPDATE);
            RegTensor<uint32_t> idx, oV, oP;
            vlds(idx, sortedIdx, fullRep * VL32, NORM);
            vgather2(oV, sV, idx, p);
            vgather2(oP, sP, idx, p);
            vsts(oV, dV, fullRep * VL32, NORM_B32, p);
            vsts(oP, dP, fullRep * VL32, NORM_B32, p);
        }
    }
}

// Descending LSD on (val,pos): digit = ~byte(val) each round — no ~key buffer / pad-NOT pass.
PTO_INTERNAL void TopKLsdSortPairsDesc(
    __ubuf__ uint32_t* val0, __ubuf__ uint32_t* pos0, __ubuf__ uint32_t* val1, __ubuf__ uint32_t* pos1,
    __ubuf__ uint16_t* exclUb, __ubuf__ uint8_t* digitUb, __ubuf__ uint16_t* idxU16, __ubuf__ uint32_t* sortedIdx,
    uint32_t cnt)
{
    if (cnt <= 1u) {
        return;
    }
    const uint32_t alignCount = ((cnt + TOPK_RADIX_U8_VL - 1u) / TOPK_RADIX_U8_VL) * TOPK_RADIX_U8_VL;
    __ubuf__ uint8_t* digitUbCopy = (__ubuf__ uint8_t*)sortedIdx; // reuse (asc-devkit)
    __ubuf__ uint8_t* idxU8 = digitUbCopy + alignCount;

    __ubuf__ uint32_t* sV = val0;
    __ubuf__ uint32_t* sP = pos0;
    __ubuf__ uint32_t* dV = val1;
    __ubuf__ uint32_t* dP = pos1;
    for (int16_t round = 0; round < 4; ++round) {
        const int16_t shift = (int16_t)(round * 8);
        TopKRadixGetExclusiveSum</*InvertDigit=*/true>(sV, digitUb, digitUbCopy, exclUb, cnt, shift);
        TopKRadixEightBitsSort(digitUb, digitUbCopy, idxU8, idxU16, cnt);
        TopKRadixLocalSort(digitUb, idxU16, exclUb, sortedIdx, cnt);
        TopKRadixGatherPairU32(sV, sP, sortedIdx, dV, dP, cnt);
        __ubuf__ uint32_t* tV = sV;
        __ubuf__ uint32_t* tP = sP;
        sV = dV;
        sP = dP;
        dV = tV;
        dP = tP;
    }
    // 4 rounds → result in original *0 buffers.
}

// Compact GE positions (GT then EQ; skip EQ if cntGT>=take) + fused gather of keys/idx.
PTO_INTERNAL void TopKGatherGEByPos(
    __ubuf__ uint32_t* keyPtr, __ubuf__ uint32_t* srcIdxPtr, uint32_t n, uint32_t tKey, uint32_t take, uint32_t cntGT,
    __ubuf__ uint32_t* posBuf, __ubuf__ uint32_t* dstVal, __ubuf__ uint32_t* dstIdx)
{
    constexpr uint32_t VL32 = TOPK_CHUNK;
    const uint32_t nChunks = (n + VL32 - 1) / VL32;
    const uint16_t nPass = (cntGT < take) ? (uint16_t)2 : (uint16_t)1;
    __VEC_SCOPE__
    {
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> tKeyV, addOff;
        vbr((vector_u32&)tKeyV, tKey);
        vbr((vector_u32&)addOff, VL32);
        RegTensor<int32_t> index;
        UnalignReg ureg;
        sprclr(SPR_AR);
        __ubuf__ uint32_t* out = posBuf;
        for (uint16_t pass = 0; pass < nPass; ++pass) {
            vci(index, (int32_t)0, INC_ORDER);
            for (uint16_t c = 0; c < (uint16_t)nChunks; ++c) {
                uint32_t off = c * VL32;
                uint32_t remain = n - off;
                uint32_t valid = remain < VL32 ? remain : VL32;
                MaskReg pLoad = plt_b32(valid, POST_UPDATE);
                RegTensor<uint32_t> key32, sqz;
                MaskReg sel;
                vlds(key32, keyPtr, off, NORM);
                if (pass == 0) {
                    vcmp_gt(sel, (vector_u32&)key32, (vector_u32&)tKeyV, pLoad);
                } else {
                    vcmp_eq(sel, (vector_u32&)key32, (vector_u32&)tKeyV, pLoad);
                }
                vsqz(sqz, (vector_u32&)index, sel, MODE_STORED);
                vstur(ureg, sqz, out, POST_UPDATE);
                vadd((vector_u32&)index, (vector_u32&)index, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            }
        }
        vstar(ureg, out);
        sprclr(SPR_AR);
        mem_bar(VST_VLD);
        // Keys → dstVal; srcIdx → dstIdx (dstIdx may alias posBuf).
        const uint16_t rep = (uint16_t)((take + VL32 - 1u) / VL32);
        uint32_t remain = take;
        for (uint16_t i = 0; i < rep; ++i) {
            uint32_t valid = remain < VL32 ? remain : VL32;
            remain -= valid;
            MaskReg p = plt_b32(valid, POST_UPDATE);
            RegTensor<uint32_t> pos, key;
            vlds(pos, posBuf, i * VL32, NORM);
            vgather2(key, keyPtr, pos, p);
            vsts(key, dstVal, i * VL32, NORM_B32, p);
        }
        remain = take;
        for (uint16_t i = 0; i < rep; ++i) {
            uint32_t valid = remain < VL32 ? remain : VL32;
            remain -= valid;
            MaskReg p = plt_b32(valid, POST_UPDATE);
            RegTensor<uint32_t> pos, idx;
            vlds(pos, posBuf, i * VL32, NORM);
            vgather2(idx, srcIdxPtr, pos, p);
            vsts(idx, dstIdx, i * VL32, NORM_B32, p);
        }
    }
}

// GT-only positions + gather (K-sized workspace; EQ via PickEq).
// R12/R13: VL-aligned N → static PAT_ALL (no per-iter plt). Digits inverted from dstVal
// in LSD — no keyDup store.
template <uint32_t N>
PTO_INTERNAL void TopKGatherGTByPos(
    __ubuf__ uint32_t* keyPtr, __ubuf__ uint32_t* srcIdxPtr, uint32_t n, uint32_t tKey, uint32_t cntGT,
    __ubuf__ uint32_t* posBuf, __ubuf__ uint32_t* dstVal, __ubuf__ uint32_t* dstIdx)
{
    if (cntGT == 0) {
        return;
    }
    constexpr uint32_t VL32 = TOPK_CHUNK;
    constexpr bool kAligned = (N % VL32) == 0;
    constexpr uint32_t kFullChunks = N / VL32;
    const uint32_t nChunks = kAligned ? kFullChunks : ((n + VL32 - 1) / VL32);
    const uint32_t fullRep = cntGT / VL32;
    const uint32_t tailN = cntGT - fullRep * VL32;
    const uint16_t tailRep = (tailN > 0u) ? (uint16_t)1 : (uint16_t)0;
    __VEC_SCOPE__
    {
        MaskReg pAllB32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> tKeyV, addOff;
        vbr((vector_u32&)tKeyV, tKey);
        vbr((vector_u32&)addOff, VL32);
        RegTensor<int32_t> index;
        vci(index, (int32_t)0, INC_ORDER);
        UnalignReg ureg;
        sprclr(SPR_AR);
        __ubuf__ uint32_t* out = posBuf;
        if constexpr (kAligned) {
            for (uint16_t c = 0; c < (uint16_t)kFullChunks; ++c) {
                RegTensor<uint32_t> key32, sqz;
                MaskReg sel;
                vlds(key32, keyPtr, c * VL32, NORM);
                vcmp_gt(sel, (vector_u32&)key32, (vector_u32&)tKeyV, pAllB32);
                vsqz(sqz, (vector_u32&)index, sel, MODE_STORED);
                vstur(ureg, sqz, out, POST_UPDATE);
                vadd((vector_u32&)index, (vector_u32&)index, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            }
        } else {
            for (uint16_t c = 0; c < (uint16_t)nChunks; ++c) {
                uint32_t off = c * VL32;
                uint32_t remain = n - off;
                uint32_t valid = remain < VL32 ? remain : VL32;
                MaskReg pLoad = plt_b32(valid, POST_UPDATE);
                RegTensor<uint32_t> key32, sqz;
                MaskReg sel;
                vlds(key32, keyPtr, off, NORM);
                vcmp_gt(sel, (vector_u32&)key32, (vector_u32&)tKeyV, pLoad);
                vsqz(sqz, (vector_u32&)index, sel, MODE_STORED);
                vstur(ureg, sqz, out, POST_UPDATE);
                vadd((vector_u32&)index, (vector_u32&)index, (vector_u32&)addOff, pAllB32, MODE_ZEROING);
            }
        }
        vstar(ureg, out);
        sprclr(SPR_AR);
        mem_bar(VST_VLD);
        for (uint16_t i = 0; i < (uint16_t)fullRep; ++i) {
            RegTensor<uint32_t> pos, key, idx;
            vlds(pos, posBuf, i * VL32, NORM);
            vgather2(key, keyPtr, pos, pAllB32);
            vgather2(idx, srcIdxPtr, pos, pAllB32);
            vsts(key, dstVal, i * VL32, NORM_B32, pAllB32);
            vsts(idx, dstIdx, i * VL32, NORM_B32, pAllB32);
        }
        for (uint16_t t = 0; t < tailRep; ++t) {
            uint32_t sreg = tailN;
            MaskReg pTail = plt_b32(sreg, POST_UPDATE);
            RegTensor<uint32_t> pos, key, idx;
            vlds(pos, posBuf, fullRep * VL32, NORM);
            vgather2(key, keyPtr, pos, pTail);
            vgather2(idx, srcIdxPtr, pos, pTail);
            vsts(key, dstVal, fullRep * VL32, NORM_B32, pTail);
            vsts(idx, dstIdx, fullRep * VL32, NORM_B32, pTail);
        }
    }
}

template <bool largest, typename T, uint32_t N, uint32_t K>
PTO_INTERNAL void TopKRadixSelectRow(
    __ubuf__ T* srcPtr, uint32_t n, __ubuf__ uint32_t* tmpBase, __ubuf__ T* valPtr, __ubuf__ int32_t* idxPtr,
    __ubuf__ uint32_t* srcIdxPtr)
{
    static_assert(K > 0, "TopK RADIX_SELECT requires K > 0");
    static_assert(N > 0, "TopK RADIX_SELECT requires N > 0");
    constexpr uint32_t ROUNDS = sizeof(T);
    constexpr uint32_t kPadElems = TopKRadixAlignCount(K);
    constexpr uint32_t geCap = TopKGeCap(K, N);
    const uint32_t nChunks = (n + TOPK_CHUNK - 1) / TOPK_CHUNK;
    __ubuf__ uint32_t* cumUb = tmpBase;
    __ubuf__ uint16_t* cum16 = (__ubuf__ uint16_t*)tmpBase;
    __ubuf__ uint16_t* exclUb = (__ubuf__ uint16_t*)(tmpBase + 256); // byte 1KiB
    // sizeof(T)==4: RemapBits in-place (asc-devkit isReuseSrc). Else keyUb[N] @3KiB.
    __ubuf__ uint32_t* keyPtr;
    __ubuf__ uint32_t* geVal;
    if constexpr (sizeof(T) == 4) {
        keyPtr = (__ubuf__ uint32_t*)srcPtr;
        geVal = tmpBase + 768; // byte 3KiB
    } else {
        keyPtr = tmpBase + 768;
        geVal = keyPtr + N;
    }
    __ubuf__ uint32_t* geIdx = geVal + geCap;
    // pairKey0/1 slots kept in layout for tmp sizing; LSD now inverts digits from val (no ~key).
    __ubuf__ uint32_t* pairKey0 = geIdx + geCap;
    __ubuf__ uint32_t* pairVal1 = pairKey0 + kPadElems;
    __ubuf__ uint32_t* pairPos1 = pairVal1 + kPadElems;
    __ubuf__ uint32_t* pairKey1 = pairPos1 + kPadElems;
    (void)pairKey0;
    (void)pairKey1;
    __ubuf__ uint8_t* digitUb = (__ubuf__ uint8_t*)(pairKey1 + kPadElems);
    __ubuf__ uint16_t* idxU16 = (__ubuf__ uint16_t*)(digitUb + kPadElems);
    __ubuf__ uint32_t* sortedIdx = (__ubuf__ uint32_t*)((__ubuf__ uint8_t*)idxU16 + 2u * kPadElems);

    // 1–2) RemapBits + digit hist (b32 fuses RemapBits into hist0).
    uint32_t selByte[4] = {0, 0, 0, 0};
    uint32_t cntGT = 0;
    uint32_t r0 = 0;
    if constexpr (sizeof(T) == 4) {
        TopKRemapAndHist0<largest, T>(srcPtr, keyPtr, n, cumUb);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const uint32_t total = cum16[255];
        const uint32_t kPrime = K - cntGT;
        if (kPrime != 0 && total >= kPrime) {
            const uint32_t q = total - kPrime;
            uint32_t lo = 0;
            uint32_t hi = 255;
            while (lo < hi) {
                const uint32_t mid = (lo + hi + 1) >> 1;
                if (cum16[mid - 1] <= q) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            selByte[0] = lo;
            cntGT += total - cum16[lo];
        }
        r0 = 1;
    } else {
        TopKRemapBits<largest, T>(srcPtr, keyPtr, n);
        mem_bar(VST_VLD);
    }
    for (uint32_t r = r0; r < ROUNDS; ++r) {
        if (r == 0) {
            TopKRadixHistRoundKeys<T, 0>(keyPtr, n, nChunks, 0, 0, 0, cumUb);
        } else if (r == 1) {
            TopKRadixHistRoundKeys<T, 1>(keyPtr, n, nChunks, selByte[0], 0, 0, cumUb);
        } else if (r == 2) {
            TopKRadixHistRoundKeys<T, 2>(keyPtr, n, nChunks, selByte[0], selByte[1], 0, cumUb);
        } else {
            TopKRadixHistRoundKeys<T, 3>(keyPtr, n, nChunks, selByte[0], selByte[1], selByte[2], cumUb);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const uint32_t total = (sizeof(T) == 4) ? (uint32_t)cum16[255] : cumUb[255];
        const uint32_t kPrime = K - cntGT;
        if (kPrime == 0 || total < kPrime) {
            break;
        }
        const uint32_t q = total - kPrime;
        uint32_t lo = 0;
        uint32_t hi = 255;
        while (lo < hi) {
            const uint32_t mid = (lo + hi + 1) >> 1;
            const uint32_t midCum = (sizeof(T) == 4) ? (uint32_t)cum16[mid - 1] : cumUb[mid - 1];
            if (midCum <= q) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        selByte[r] = lo;
        const uint32_t loCum = (sizeof(T) == 4) ? (uint32_t)cum16[lo] : cumUb[lo];
        cntGT += total - loCum;
    }
    uint32_t tKey = 0;
    for (uint32_t r = 0; r < ROUNDS; ++r) {
        tKey |= selByte[r] << (8 * (ROUNDS - 1 - r));
    }
    if constexpr (!largest && sizeof(T) < 4) {
        tKey |= 0xFFFFFFFFu << (8 * sizeof(T));
    }
    if (cntGT > K) {
        cntGT = K;
    }
    constexpr bool kFullGe = (geCap >= N);
    __ubuf__ uint32_t* scratchUb = tmpBase + 128; // byte 512

    // 3–4) Pos-compact GE + O(K) gather; LSD + TwiddleOut.
    if constexpr (kFullGe) {
        TopKGatherGEByPos(keyPtr, srcIdxPtr, n, tKey, K, cntGT, /*posBuf=*/geIdx, geVal, /*dstIdx=*/geIdx);
        TopKLsdSortPairsDesc(geVal, geIdx, pairVal1, pairPos1, exclUb, digitUb, idxU16, sortedIdx, K);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        if constexpr (sizeof(T) == 4) {
            constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
            const uint16_t rep = (uint16_t)((K + VL32 - 1u) / VL32);
            __VEC_SCOPE__
            {
                MaskReg pAllB32 = pset_b32(PAT_ALL);
                uint32_t remain = K;
                for (uint16_t i = 0; i < rep; ++i) {
                    uint32_t valid = remain < VL32 ? remain : VL32;
                    remain -= valid;
                    MaskReg p = plt_b32(valid, POST_UPDATE);
                    RegTensor<uint32_t> key, idx;
                    vlds(key, geVal, i * VL32, NORM);
                    TopKTwiddleOutU32<largest, T>(key, pAllB32);
                    vsts(key, (__ubuf__ uint32_t*)valPtr, i * VL32, NORM_B32, p);
                    vlds(idx, geIdx, i * VL32, NORM);
                    vsts(idx, (__ubuf__ uint32_t*)idxPtr, i * VL32, NORM_B32, p);
                }
            }
        } else {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t bits = TopKTwiddleOutScalar<largest, T>(geVal[j]);
                if constexpr (sizeof(T) == 2) {
                    ((__ubuf__ uint16_t*)valPtr)[j] = (uint16_t)bits;
                } else {
                    ((__ubuf__ uint8_t*)valPtr)[j] = (uint8_t)bits;
                }
                idxPtr[j] = (int32_t)geIdx[j];
            }
        }
    } else {
        if (cntGT > 0) {
            TopKGatherGTByPos<N>(keyPtr, srcIdxPtr, n, tKey, cntGT, /*posBuf=*/geIdx, geVal, /*dstIdx=*/geIdx);
            TopKLsdSortPairsDesc(geVal, geIdx, pairVal1, pairPos1, exclUb, digitUb, idxU16, sortedIdx, cntGT);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            if constexpr (sizeof(T) == 4) {
                constexpr uint32_t VL32 = REPEAT_BYTE / sizeof(uint32_t);
                const uint32_t fullRep = cntGT / VL32;
                const uint32_t tailN = cntGT - fullRep * VL32;
                const uint16_t tailRep = (tailN > 0u) ? (uint16_t)1 : (uint16_t)0;
                __VEC_SCOPE__
                {
                    MaskReg pAllB32 = pset_b32(PAT_ALL);
                    for (uint16_t i = 0; i < (uint16_t)fullRep; ++i) {
                        RegTensor<uint32_t> key, idx;
                        vlds(key, geVal, i * VL32, NORM);
                        TopKTwiddleOutU32<largest, T>(key, pAllB32);
                        vsts(key, (__ubuf__ uint32_t*)valPtr, i * VL32, NORM_B32, pAllB32);
                        vlds(idx, geIdx, i * VL32, NORM);
                        vsts(idx, (__ubuf__ uint32_t*)idxPtr, i * VL32, NORM_B32, pAllB32);
                    }
                    for (uint16_t t = 0; t < tailRep; ++t) {
                        uint32_t sreg = tailN;
                        MaskReg p = plt_b32(sreg, POST_UPDATE);
                        RegTensor<uint32_t> key, idx;
                        vlds(key, geVal, fullRep * VL32, NORM);
                        TopKTwiddleOutU32<largest, T>(key, pAllB32);
                        vsts(key, (__ubuf__ uint32_t*)valPtr, fullRep * VL32, NORM_B32, p);
                        vlds(idx, geIdx, fullRep * VL32, NORM);
                        vsts(idx, (__ubuf__ uint32_t*)idxPtr, fullRep * VL32, NORM_B32, p);
                    }
                }
            } else {
                for (uint32_t j = 0; j < cntGT; ++j) {
                    const uint32_t bits = TopKTwiddleOutScalar<largest, T>(geVal[j]);
                    if constexpr (sizeof(T) == 2) {
                        ((__ubuf__ uint16_t*)valPtr)[j] = (uint16_t)bits;
                    } else {
                        ((__ubuf__ uint8_t*)valPtr)[j] = (uint8_t)bits;
                    }
                    idxPtr[j] = (int32_t)geIdx[j];
                }
            }
        }
        const uint32_t m = K - cntGT;
        if (m > 0) {
            // m is typically 1 (511 GT + 1 EQ); PickEq one-sweep beats full-N squeeze-scatter.
            TopKRadixPickEq<largest, T>(keyPtr, n, nChunks, tKey, m, cntGT, valPtr, idxPtr, scratchUb, srcIdxPtr);
        }
    }
}

// One top-1 over UB scores: park (winIdx, topVal) at ring slot j, then kill the winner.
// PTO_INTERNAL (= always_inline) so this folds into TopKIterSelectRow's __VEC_SCOPE__ (no call).
template <bool largest, uint32_t N>
PTO_INTERNAL void TopKIterParkOne(
    __ubuf__ float* sPtr, __ubuf__ uint32_t* idxRing, __ubuf__ uint32_t* valRing, uint32_t j, MaskReg pFull,
    MaskReg pTail, vector_f32 fill, vector_s32 intMax, vector_s32 arrange)
{
    constexpr uint32_t VL = TOPK_VL;
    constexpr uint32_t nf = N / VL;
    constexpr bool hasTail = (N % VL) != 0;

    vector_f32 s, acc, topS, topB;
    vector_s32 base, iCur;
    vlds(s, sPtr, 0, NORM);
    if constexpr (nf == 0) {
        vsel(s, s, fill, pTail);
    }
    acc = s;
    for (uint16_t b = 1; b < (uint16_t)nf; ++b) {
        vlds(s, sPtr, b * VL, NORM);
        if constexpr (largest) {
            vmax(acc, acc, s, pFull, MODE_ZEROING);
        } else {
            vmin(acc, acc, s, pFull, MODE_ZEROING);
        }
    }
    if constexpr (hasTail && nf > 0) {
        vlds(s, sPtr, nf * VL, NORM);
        vsel(s, s, fill, pTail);
        if constexpr (largest) {
            vmax(acc, acc, s, pFull, MODE_ZEROING);
        } else {
            vmin(acc, acc, s, pFull, MODE_ZEROING);
        }
    }
    if constexpr (largest) {
        vcmax(topS, acc, pFull, MODE_ZEROING);
    } else {
        vcmin(topS, acc, pFull, MODE_ZEROING);
    }
    vdup(topB, topS, pFull, POS_LOWEST, MODE_ZEROING);

    vector_s32 cAcc, c, winS;
    vector_bool eqB;
    vbr(base, (int32_t)0);
    vadd(iCur, base, arrange, pFull, MODE_ZEROING);
    vlds(s, sPtr, 0, NORM);
    vcmp_eq(eqB, s, topB, pFull);
    if constexpr (nf == 0) {
        pand(eqB, eqB, pTail, pFull);
    }
    vsel(cAcc, iCur, intMax, eqB);
    for (uint16_t b = 1; b < (uint16_t)nf; ++b) {
        vlds(s, sPtr, b * VL, NORM);
        vbr(base, (int32_t)(b * VL));
        vadd(iCur, base, arrange, pFull, MODE_ZEROING);
        vcmp_eq(eqB, s, topB, pFull);
        vsel(c, iCur, intMax, eqB);
        vmin((vector_u32&)cAcc, (vector_u32&)cAcc, (vector_u32&)c, pFull, MODE_ZEROING);
    }
    if constexpr (hasTail && nf > 0) {
        vlds(s, sPtr, nf * VL, NORM);
        vbr(base, (int32_t)(nf * VL));
        vadd(iCur, base, arrange, pFull, MODE_ZEROING);
        vcmp_eq(eqB, s, topB, pFull);
        pand(eqB, eqB, pTail, pFull);
        vsel(c, iCur, intMax, eqB);
        vmin((vector_u32&)cAcc, (vector_u32&)cAcc, (vector_u32&)c, pFull, MODE_ZEROING);
    }
    vcmin((vector_u32&)winS, (vector_u32&)cAcc, pFull, MODE_ZEROING);
    vsts((vector_u32&)winS, idxRing, j * VL, NORM_B32, pFull);
    vsts((vector_u32&)topB, valRing, j * VL, NORM_B32, pFull);

    vector_s32 winB;
    vector_bool ewB;
    vector_f32 kB;
    vdup(winB, winS, pFull, POS_LOWEST, MODE_ZEROING);
    vbr(base, (int32_t)0);
    vadd(iCur, base, arrange, pFull, MODE_ZEROING);
    vlds(s, sPtr, 0, NORM);
    vcmp_eq(ewB, iCur, winB, pFull);
    vsel(kB, fill, s, ewB);
    if constexpr (nf == 0) {
        vsts(kB, sPtr, 0, NORM_B32, pTail);
    } else {
        vsts(kB, sPtr, 0, NORM_B32, pFull);
    }
    for (uint16_t b = 1; b < (uint16_t)nf; ++b) {
        vlds(s, sPtr, b * VL, NORM);
        vbr(base, (int32_t)(b * VL));
        vadd(iCur, base, arrange, pFull, MODE_ZEROING);
        vcmp_eq(ewB, iCur, winB, pFull);
        vsel(kB, fill, s, ewB);
        vsts(kB, sPtr, b * VL, NORM_B32, pFull);
    }
    if constexpr (hasTail && nf > 0) {
        vlds(s, sPtr, nf * VL, NORM);
        vbr(base, (int32_t)(nf * VL));
        vadd(iCur, base, arrange, pFull, MODE_ZEROING);
        vcmp_eq(ewB, iCur, winB, pFull);
        vsel(kB, fill, s, ewB);
        vsts(kB, sPtr, nf * VL, NORM_B32, pTail);
    }
}

// Iterative top-1 select (argmax style). Scores are killed in UB after each pick; winners leave
// in sorted order, parked in 64-block rings and flushed as one dense vector per 64 results.
// N/K are compile-time; park is always_inline TopKIterParkOne (if constexpr rem).
template <bool largest, uint32_t N, uint32_t K>
PTO_INTERNAL void TopKIterSelectRow(
    __ubuf__ float* sPtr, __ubuf__ uint32_t* idxRing, __ubuf__ uint32_t* valRing, __ubuf__ float* valPtr,
    __ubuf__ int32_t* idxPtr, __ubuf__ uint32_t* srcIdxPtr)
{
    constexpr uint32_t VL = TOPK_VL;
    constexpr bool hasTail = (N % VL) != 0;
    constexpr uint32_t tailCount = N - (N / VL) * VL;
    constexpr uint32_t nGroup = K / VL;
    constexpr uint32_t rem = K % VL;
    static_assert(N > 0, "TopK ITER_SELECT requires N > 0");
    static_assert(K > 0, "TopK ITER_SELECT requires K > 0");

    __VEC_SCOPE__
    {
        MaskReg pFull = pset_b32(PAT_ALL);
        MaskReg pTail;
        if constexpr (hasTail) {
            uint32_t sregT = tailCount;
            pTail = plt_b32(sregT, POST_UPDATE);
        }
        vector_f32 fill;
        if constexpr (largest) {
            vbr(fill, -3.40282347e38f);
        } else {
            vbr(fill, 3.40282346e38f);
        }
        vector_s32 intMax, arrange;
        vbr(intMax, (int32_t)0x7FFFFFFF);
        vci(arrange, 0, INC_ORDER);

        vector_s32 cIdx;
        vector_u32 idxVec, srcIdxVec, valVec;
        for (uint16_t g = 0; g < (uint16_t)nGroup; ++g) {
            for (uint16_t j = 0; j < (uint16_t)VL; ++j) {
                TopKIterParkOne<largest, N>(sPtr, idxRing, valRing, j, pFull, pTail, fill, intMax, arrange);
            }
            vci(cIdx, 0, INC_ORDER);
            vshls((vector_u32&)cIdx, (vector_u32&)cIdx, (int16_t)6, pFull, MODE_ZEROING);
            vgather2(idxVec, idxRing, (vector_u32&)cIdx, pFull);
            vgather2(srcIdxVec, srcIdxPtr, (vector_u32&)idxVec, pFull);
            vgather2(valVec, valRing, (vector_u32&)cIdx, pFull);
            vsts(srcIdxVec, (__ubuf__ uint32_t*)idxPtr, g * VL, NORM_B32, pFull);
            vsts(valVec, (__ubuf__ uint32_t*)valPtr, g * VL, NORM_B32, pFull);
        }
        if constexpr (rem > 0) {
            for (uint16_t j = 0; j < (uint16_t)rem; ++j) {
                TopKIterParkOne<largest, N>(sPtr, idxRing, valRing, j, pFull, pTail, fill, intMax, arrange);
            }
            vci(cIdx, 0, INC_ORDER);
            vshls((vector_u32&)cIdx, (vector_u32&)cIdx, (int16_t)6, pFull, MODE_ZEROING);
            uint32_t sregR = rem;
            MaskReg pRem = plt_b32(sregR, POST_UPDATE);
            vgather2(idxVec, idxRing, (vector_u32&)cIdx, pRem);
            vgather2(srcIdxVec, srcIdxPtr, (vector_u32&)idxVec, pRem);
            vgather2(valVec, valRing, (vector_u32&)cIdx, pRem);
            vsts(srcIdxVec, (__ubuf__ uint32_t*)idxPtr, nGroup * VL, NORM_B32, pRem);
            vsts(valVec, (__ubuf__ uint32_t*)valPtr, nGroup * VL, NORM_B32, pRem);
        }
    }
}

template <
    TopKAlgo algo, typename DstValueTile, typename DstIdxTile, typename SrcTile, typename SrcIdxTile, typename TmpTile,
    int dim = -1>
PTO_INTERNAL void TopKCheckStatic()
{
    static_assert(dim == -1, "TopK only supports sorting along the last dimension (dim = -1).");
    static_assert(
        (SrcTile::Loc == TileType::Vec) && (DstValueTile::Loc == TileType::Vec) && (DstIdxTile::Loc == TileType::Vec) &&
            (SrcIdxTile::Loc == TileType::Vec) && (TmpTile::Loc == TileType::Vec),
        "TileType must be Vec!");
    static_assert(
        (SrcTile::isRowMajor && DstValueTile::isRowMajor && DstIdxTile::isRowMajor && SrcIdxTile::isRowMajor &&
         TmpTile::isRowMajor),
        "Expect row major tiles.");
    static_assert(
        (std::is_same<typename DstValueTile::DType, typename SrcTile::DType>::value),
        "Value tile dtype must match the source tile dtype.");
    static_assert(
        (std::is_same<typename SrcIdxTile::DType, int32_t>::value) &&
            (std::is_same<typename DstIdxTile::DType, int32_t>::value),
        "Index tiles dtype must be int32_t.");
    static_assert(
        TmpTile::Cols * sizeof(typename TmpTile::DType) >= TOPK_TMP_FIXED_BYTES,
        "Tmp tile must hold the TopK fixed tables; callers additionally need"
        " TopKTmpBytesFor(align8(k)) bytes for the RADIX_SELECT pair buffers.");
    if constexpr (algo == TopKAlgo::ITER_SELECT) {
        static_assert((std::is_same<typename SrcTile::DType, float>::value), "TopK ITER_SELECT only supports DT_FP32.");
        static_assert(SrcTile::Cols > 0, "TopK ITER_SELECT requires a compile-time SrcTile::Cols > 0.");
    } else {
        static_assert(
            sizeof(typename SrcTile::DType) <= 4,
            "TopK RADIX_SELECT does not support DT_INT64/DT_UINT64 yet; use keys of at most 32 bits.");
    }
}

// K is compile-time for both ITER_SELECT (unrolls the pick loop) and RADIX_SELECT.
template <
    TopKAlgo algo, uint32_t K, typename DstValueTile, typename DstIdxTile, typename SrcTile, typename SrcIdxTile,
    typename TmpTile, int dim = -1>
AICORE inline void TOPK_IMPL(
    DstValueTile& value, DstIdxTile& index, SrcTile& src, SrcIdxTile& srcIdx, TmpTile& tmp, bool largest)
{
    static_assert(K > 0, "TopK requires a non-zero template K.");
    TopKCheckStatic<algo, DstValueTile, DstIdxTile, SrcTile, SrcIdxTile, TmpTile, dim>();
    using T = typename SrcTile::DType;
    constexpr uint32_t kSrcRowStride = SrcTile::RowStride;
    constexpr uint32_t kValRowStride = DstValueTile::RowStride;
    constexpr uint32_t kIdxRowStride = DstIdxTile::RowStride;
    constexpr uint32_t kSrcIdxRowStride = SrcIdxTile::RowStride;
    __ubuf__ T* srcBase = (__ubuf__ T*)(src.data());
    __ubuf__ T* valBase = (__ubuf__ T*)(value.data());
    __ubuf__ int32_t* idxBase = (__ubuf__ int32_t*)(index.data());
    __ubuf__ uint32_t* srcIdxBase = (__ubuf__ uint32_t*)(srcIdx.data());
    __ubuf__ uint32_t* tmpBase = (__ubuf__ uint32_t*)(tmp.data());
    const uint32_t rows = src.GetValidRow();
    const uint32_t n = src.GetValidCol();
    for (uint32_t row = 0; row < rows; ++row) {
        __ubuf__ T* srcPtr = srcBase + row * kSrcRowStride;
        __ubuf__ T* valPtr = valBase + row * kValRowStride;
        __ubuf__ int32_t* idxPtr = idxBase + row * kIdxRowStride;
        __ubuf__ uint32_t* srcIdxPtr = srcIdxBase + row * kSrcIdxRowStride;
        if constexpr (algo == TopKAlgo::ITER_SELECT) {
            constexpr uint32_t kN = static_cast<uint32_t>(SrcTile::Cols);
            if (largest) {
                TopKIterSelectRow<true, kN, K>(
                    srcPtr, tmpBase, tmpBase + TOPK_RING_BYTES / sizeof(uint32_t), valPtr, idxPtr, srcIdxPtr);
            } else {
                TopKIterSelectRow<false, kN, K>(
                    srcPtr, tmpBase, tmpBase + TOPK_RING_BYTES / sizeof(uint32_t), valPtr, idxPtr, srcIdxPtr);
            }
        } else {
            constexpr uint32_t kN = static_cast<uint32_t>(SrcTile::Cols);
            if (largest) {
                TopKRadixSelectRow<true, T, kN, K>(srcPtr, n, tmpBase, valPtr, idxPtr, srcIdxPtr);
            } else {
                TopKRadixSelectRow<false, T, kN, K>(srcPtr, n, tmpBase, valPtr, idxPtr, srcIdxPtr);
            }
        }
    }
}

// Public TopK entry (standalone exp header; not registered in pto_instr.hpp).
// e.g. TOPK<ITER_SELECT, 3>(...) / TOPK<RADIX_SELECT, 512>(...); no runtime k.
template <
    TopKAlgo algo, uint32_t K, typename DstValueTile, typename DstIdxTile, typename SrcTile, typename SrcIdxTile,
    typename TmpTile>
AICORE inline void TOPK(
    DstValueTile& value, DstIdxTile& index, SrcTile& src, SrcIdxTile& srcIdx, TmpTile& tmp, bool largest)
{
    TOPK_IMPL<algo, K>(value, index, src, srcIdx, tmp, largest);
}

} // namespace pto
#endif
