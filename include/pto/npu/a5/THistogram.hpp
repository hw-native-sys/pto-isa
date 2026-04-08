/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef THISTOGRAM_HPP
#define THISTOGRAM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

// HistByte enum is defined in <pto/common/type.hpp>

PTO_INTERNAL void histogram_b8i_b32o(vector_u8 &vb8_src, vector_u16 &vb16_bin_n0, vector_u16 &vb16_bin_n1,
                                     vector_u32 &vb32_bin_n0_even_inc, vector_u32 &vb32_bin_n0_odd_inc,
                                     vector_u32 &vb32_bin_n1_even_inc, vector_u32 &vb32_bin_n1_odd_inc,
                                     vector_bool &preg_b8_0, vector_bool &preg_b8_1, vector_bool &preg_b16,
                                     vector_bool &preg_b32)
{
    vector_u32 vb32_bin_n0_even, vb32_bin_n0_odd, vb32_bin_n1_even, vb32_bin_n1_odd;
    chistv2(vb16_bin_n0, vb8_src, preg_b8_0, Bin_N0);
    chistv2(vb16_bin_n1, vb8_src, preg_b8_1, Bin_N1);
    vcvt(vb32_bin_n0_even, vb16_bin_n0, preg_b16, PART_EVEN);
    vcvt(vb32_bin_n0_odd, vb16_bin_n0, preg_b16, PART_ODD);
    vcvt(vb32_bin_n1_even, vb16_bin_n1, preg_b16, PART_EVEN);
    vcvt(vb32_bin_n1_odd, vb16_bin_n1, preg_b16, PART_ODD);
    vadd(vb32_bin_n0_even_inc, vb32_bin_n0_even_inc, vb32_bin_n0_even, preg_b32, MODE_ZEROING);
    vadd(vb32_bin_n0_odd_inc, vb32_bin_n0_odd_inc, vb32_bin_n0_odd, preg_b32, MODE_ZEROING);
    vadd(vb32_bin_n1_even_inc, vb32_bin_n1_even_inc, vb32_bin_n1_even, preg_b32, MODE_ZEROING);
    vadd(vb32_bin_n1_odd_inc, vb32_bin_n1_odd_inc, vb32_bin_n1_odd, preg_b32, MODE_ZEROING);
    vbr(vb16_bin_n0, 0);
    vbr(vb16_bin_n1, 0);
}

template <typename TileDst, typename TileSrc, typename TileIdx, bool isMSB = true>
__tf__ PTO_INTERNAL void THistogram(typename TileDst::TileDType __out__ bin_count,
                                    typename TileSrc::TileDType __in__ scores, typename TileIdx::TileDType __in__ idx,
                                    unsigned validRows, unsigned validCols)
{
    __ubuf__ typename TileDst::DType *dstPtr = (__ubuf__ typename TileDst::DType *)__cce_get_tile_ptr(bin_count);
    __ubuf__ typename TileSrc::DType *srcPtr = (__ubuf__ typename TileSrc::DType *)__cce_get_tile_ptr(scores);
    __ubuf__ typename TileIdx::DType *idxPtr = (__ubuf__ typename TileIdx::DType *)__cce_get_tile_ptr(idx);
    constexpr unsigned Type_coeff = sizeof(uint16_t) / sizeof(uint8_t); // input is of type b16, load of type b8
    __VEC_SCOPE__
    {
        vector_u16 vb16_BIN_N0, vb16_BIN_N1;
        vector_u32 vb32_BIN_N0_even, vb32_BIN_N0_odd, vb32_BIN_N1_even, vb32_BIN_N1_odd;
        vector_u8 vb8_src_MSB, vb8_src_LSB, vb8_idx;
        vector_bool preg_idx;
        constexpr unsigned ElemPerRepeatB8 = REPEAT_BYTE / sizeof(uint8_t);
        unsigned repeatTimesPerRow = CeilDivision(validCols, ElemPerRepeatB8);
        vbr(vb16_BIN_N0, 0);
        vbr(vb16_BIN_N1, 0);
        vector_bool preg_all_b16 = pset_b16(PAT_ALL);
        vector_bool preg_all_b32 = pset_b32(PAT_ALL);
        __ubuf__ uint32_t *dstPtr128ElemShift = (__ubuf__ uint32_t *)dstPtr + 128;
        for (uint16_t r = 0; r < (uint16_t)validRows; ++r) {
            vlds(vb8_idx, idxPtr, 1, BRC_B8, POST_UPDATE);
            uint32_t sreg_even = validCols, sreg_odd = validCols;
            vbr(vb32_BIN_N0_even, 0);
            vbr(vb32_BIN_N0_odd, 0);
            vbr(vb32_BIN_N1_even, 0);
            vbr(vb32_BIN_N1_odd, 0);
            for (uint16_t c = 0; c < (uint16_t)repeatTimesPerRow; ++c) {
                vector_bool preg_b8_0 = CreatePredicate<uint8_t>(sreg_even);
                vector_bool preg_b8_1 = CreatePredicate<uint8_t>(sreg_odd);
                vlds((vector_u8 &)vb8_src_LSB, (vector_u8 &)vb8_src_MSB, (__ubuf__ uint8_t *&)srcPtr,
                     Type_coeff * (r * TileSrc::Cols + c * ElemPerRepeatB8), DINTLV_B8);
                if constexpr (isMSB) {
                    histogram_b8i_b32o(vb8_src_MSB, vb16_BIN_N0, vb16_BIN_N1, (vector_u32 &)vb32_BIN_N0_even,
                                       (vector_u32 &)vb32_BIN_N0_odd, (vector_u32 &)vb32_BIN_N1_even,
                                       (vector_u32 &)vb32_BIN_N1_odd, preg_b8_0, preg_b8_1, preg_all_b16, preg_all_b32);
                } else {
                    vcmp_eq(preg_idx, (vector_u8 &)vb8_src_MSB, vb8_idx, preg_b8_0);
                    histogram_b8i_b32o(vb8_src_LSB, vb16_BIN_N0, vb16_BIN_N1, (vector_u32 &)vb32_BIN_N0_even,
                                       (vector_u32 &)vb32_BIN_N0_odd, (vector_u32 &)vb32_BIN_N1_even,
                                       (vector_u32 &)vb32_BIN_N1_odd, preg_idx, preg_idx, preg_all_b16, preg_all_b32);
                }
            }
            vsts((vector_u32 &)vb32_BIN_N0_even, (vector_u32 &)vb32_BIN_N0_odd, (__ubuf__ uint32_t *&)dstPtr, 256 * r,
                 INTLV_B32, preg_all_b32);
            vsts((vector_u32 &)vb32_BIN_N1_even, (vector_u32 &)vb32_BIN_N1_odd,
                 (__ubuf__ uint32_t *&)dstPtr128ElemShift, 256 * r, INTLV_B32, preg_all_b32);
        }
    }
}

// ---------------------------------------------------------------------------
// uint32 helper functions
// ---------------------------------------------------------------------------

// Deinterleave 256 packed uint32 elements into 4 individual byte vectors.
PTO_INTERNAL void deintlv_u32_bytes(__ubuf__ uint32_t *srcPtr, unsigned elemOffset, vector_u8 &byte0, vector_u8 &byte1,
                                    vector_u8 &byte2, vector_u8 &byte3)
{
    constexpr unsigned TC = sizeof(uint32_t) / sizeof(uint16_t);
    __ubuf__ uint16_t *src16 = (__ubuf__ uint16_t *)srcPtr;
    vector_u16 lo0, hi0, lo1, hi1;
    vlds(lo0, hi0, src16, TC * elemOffset, DINTLV_B16);
    vlds(lo1, hi1, src16, TC * elemOffset + 128 * TC, DINTLV_B16);
    vdintlv(byte0, byte1, (vector_u8 &)lo0, (vector_u8 &)lo1);
    vdintlv(byte2, byte3, (vector_u8 &)hi0, (vector_u8 &)hi1);
}

// Load filter index values from the idx tile based on which byte is being processed.
template <HistByte byte>
PTO_INTERNAL void load_filter_indices(__ubuf__ uint8_t *idxPtr, unsigned idxCols, vector_u8 &idx0, vector_u8 &idx1,
                                      vector_u8 &idx2)
{
    if constexpr (byte <= HistByte::BYTE_2) {
        vlds(idx0, idxPtr, 0 * idxCols, BRC_B8);
    }
    if constexpr (byte <= HistByte::BYTE_1) {
        vlds(idx1, idxPtr, 1 * idxCols, BRC_B8);
    }
    if constexpr (byte <= HistByte::BYTE_0) {
        vlds(idx2, idxPtr, 2 * idxCols, BRC_B8);
    }
}

// Apply cascaded byte filters and compute histogram for the selected byte.
template <HistByte byte>
PTO_INTERNAL void filter_and_histogram_u32(vector_u8 &byte0, vector_u8 &byte1, vector_u8 &byte2, vector_u8 &byte3,
                                           vector_u8 &idx0, vector_u8 &idx1, vector_u8 &idx2, vector_u16 &vb16_N0,
                                           vector_u16 &vb16_N1, vector_u32 &vb32_N0_even, vector_u32 &vb32_N0_odd,
                                           vector_u32 &vb32_N1_even, vector_u32 &vb32_N1_odd, vector_bool &preg0,
                                           vector_bool &preg1, vector_bool &preg_b16, vector_bool &preg_b32)
{
    if constexpr (byte == HistByte::BYTE_3) {
        histogram_b8i_b32o(byte3, vb16_N0, vb16_N1, vb32_N0_even, vb32_N0_odd, vb32_N1_even, vb32_N1_odd, preg0, preg1,
                           preg_b16, preg_b32);
    } else {
        vector_bool filt;
        vcmp_eq(filt, byte3, idx0, preg0);
        if constexpr (byte <= HistByte::BYTE_1) {
            vcmp_eq(filt, byte2, idx1, filt);
        }
        if constexpr (byte <= HistByte::BYTE_0) {
            vcmp_eq(filt, byte1, idx2, filt);
        }
        if constexpr (byte == HistByte::BYTE_2) {
            histogram_b8i_b32o(byte2, vb16_N0, vb16_N1, vb32_N0_even, vb32_N0_odd, vb32_N1_even, vb32_N1_odd, filt,
                               filt, preg_b16, preg_b32);
        } else if constexpr (byte == HistByte::BYTE_1) {
            histogram_b8i_b32o(byte1, vb16_N0, vb16_N1, vb32_N0_even, vb32_N0_odd, vb32_N1_even, vb32_N1_odd, filt,
                               filt, preg_b16, preg_b32);
        } else {
            histogram_b8i_b32o(byte0, vb16_N0, vb16_N1, vb32_N0_even, vb32_N0_odd, vb32_N1_even, vb32_N1_odd, filt,
                               filt, preg_b16, preg_b32);
        }
    }
}

// ---------------------------------------------------------------------------
// uint32 input support: THistogramU32
// ---------------------------------------------------------------------------
// For uint32 data, the four bytes are:
//   byte0 (bits 7-0, LSB), byte1 (bits 15-8), byte2 (bits 23-16), byte3 (bits 31-24, MSB)
//
// Radix sort processes MSB-first:
// HistByte::BYTE_3 → histogram of byte3 (MSB, first pass, no filtering)
// HistByte::BYTE_2 → histogram of byte2, filtered by byte3 == idx row 0
// HistByte::BYTE_1 → histogram of byte1, filtered by byte3 == idx row 0 AND byte2 == idx row 1
// HistByte::BYTE_0 → histogram of byte0 (LSB), filtered by all three upper bytes
//
// The idx tile has shape (3 - byteVal, validCols) with RowMajor layout and uint8_t type.
// Each idx row stores one filter byte value broadcast across all columns.
// Byte extraction: DINTLV_B16 + vdintlv on the uint32 source data.

template <HistByte byte, typename TileDst, typename TileSrc, typename TileIdx>
__tf__ PTO_INTERNAL void THistogramU32(typename TileDst::TileDType __out__ bin_count,
                                       typename TileSrc::TileDType __in__ scores,
                                       typename TileIdx::TileDType __in__ idx, unsigned validRows, unsigned validCols)
{
    __ubuf__ typename TileDst::DType *dstPtr = (__ubuf__ typename TileDst::DType *)__cce_get_tile_ptr(bin_count);
    __ubuf__ typename TileSrc::DType *srcPtr = (__ubuf__ typename TileSrc::DType *)__cce_get_tile_ptr(scores);
    __ubuf__ typename TileIdx::DType *idxPtr = (__ubuf__ typename TileIdx::DType *)__cce_get_tile_ptr(idx);
    __VEC_SCOPE__
    {
        vector_u16 vb16_BIN_N0, vb16_BIN_N1;
        vector_u32 vb32_BIN_N0_even, vb32_BIN_N0_odd, vb32_BIN_N1_even, vb32_BIN_N1_odd;
        vector_u8 vb8_byte0, vb8_byte1, vb8_byte2, vb8_byte3;
        vector_u8 vb8_idx0, vb8_idx1, vb8_idx2;
        constexpr unsigned ElemPerRepeatB8 = REPEAT_BYTE / sizeof(uint8_t);
        unsigned repeatTimesPerRow = CeilDivision(validCols, ElemPerRepeatB8);
        vbr(vb16_BIN_N0, 0);
        vbr(vb16_BIN_N1, 0);
        vector_bool preg_all_b16 = pset_b16(PAT_ALL);
        vector_bool preg_all_b32 = pset_b32(PAT_ALL);
        __ubuf__ uint32_t *dstPtr128 = (__ubuf__ uint32_t *)dstPtr + 128;
        for (uint16_t r = 0; r < (uint16_t)validRows; ++r) {
            load_filter_indices<byte>(idxPtr, TileIdx::Cols, vb8_idx0, vb8_idx1, vb8_idx2);
            uint32_t sreg_even = validCols, sreg_odd = validCols;
            vbr(vb32_BIN_N0_even, 0);
            vbr(vb32_BIN_N0_odd, 0);
            vbr(vb32_BIN_N1_even, 0);
            vbr(vb32_BIN_N1_odd, 0);
            for (uint16_t c = 0; c < (uint16_t)repeatTimesPerRow; ++c) {
                vector_bool preg_b8_0 = CreatePredicate<uint8_t>(sreg_even);
                vector_bool preg_b8_1 = CreatePredicate<uint8_t>(sreg_odd);
                deintlv_u32_bytes((__ubuf__ uint32_t *)srcPtr, r * TileSrc::Cols + c * ElemPerRepeatB8, vb8_byte0,
                                  vb8_byte1, vb8_byte2, vb8_byte3);
                filter_and_histogram_u32<byte>(vb8_byte0, vb8_byte1, vb8_byte2, vb8_byte3, vb8_idx0, vb8_idx1, vb8_idx2,
                                               vb16_BIN_N0, vb16_BIN_N1, vb32_BIN_N0_even, vb32_BIN_N0_odd,
                                               vb32_BIN_N1_even, vb32_BIN_N1_odd, preg_b8_0, preg_b8_1, preg_all_b16,
                                               preg_all_b32);
            }
            vsts((vector_u32 &)vb32_BIN_N0_even, (vector_u32 &)vb32_BIN_N0_odd, (__ubuf__ uint32_t *&)dstPtr, 256 * r,
                 INTLV_B32, preg_all_b32);
            vsts((vector_u32 &)vb32_BIN_N1_even, (vector_u32 &)vb32_BIN_N1_odd, (__ubuf__ uint32_t *&)dstPtr128,
                 256 * r, INTLV_B32, preg_all_b32);
        }
    }
}

template <HistByte byte, typename TileDst, typename TileSrc, typename TileIdx>
PTO_INTERNAL void THISTOGRAM_IMPL(TileDst &dst, TileSrc &src, TileIdx &idx)
{
    using SrcT = typename TileSrc::DType;
    using DstT = typename TileDst::DType;
    using IdxT = typename TileIdx::DType;
    static_assert(std::is_same<DstT, uint32_t>::value, "Fix: THISTOGRAM destination must be uint32_t.");
    static_assert(std::is_same<IdxT, uint8_t>::value, "Fix: THISTOGRAM index must be uint8_t.");
    static_assert(std::is_same<SrcT, uint16_t>::value || std::is_same<SrcT, uint32_t>::value,
                  "Fix: THISTOGRAM source must be uint16_t or uint32_t.");
    static_assert(TileSrc::isRowMajor, "Fix: THISTOGRAM source should only follow row major layout.");
    static_assert(TileDst::isRowMajor, "Fix: THISTOGRAM destination should only follow row major layout.");

    if constexpr (std::is_same<SrcT, uint16_t>::value) {
        // uint16 mode: only BYTE_0 (LSB, bits 7-0) and BYTE_1 (MSB, bits 15-8) are valid.
        static_assert(byte == HistByte::BYTE_0 || byte == HistByte::BYTE_1,
                      "Fix: THISTOGRAM with uint16 source only supports BYTE_0 (LSB) and BYTE_1 (MSB).");
        static_assert((!TileIdx::isBoxedLayout && !TileIdx::isRowMajor && TileIdx::Cols == 1),
                      "Fix: THISTOGRAM (uint16) index should use DN layout with exactly one column: "
                      "BLayout::ColMajor + SLayout::NoneBox + Cols=1.");
        constexpr bool isMSB = (byte == HistByte::BYTE_1);
        THistogram<TileDst, TileSrc, TileIdx, isMSB>(dst.data(), src.data(), idx.data(), src.GetValidRow(),
                                                     src.GetValidCol());
    } else {
        // uint32 mode: all four bytes valid.
        // Validate idx tile shape based on byte being processed.
        // For BYTE_3 (MSB, first pass): no index input expected (idx tile is unused).
        // For BYTE_2: idx shape must be (1, N) where N == source cols.
        // For BYTE_1: idx shape must be (2, N).
        // For BYTE_0 (LSB, last pass): idx shape must be (3, N).
        if constexpr (byte == HistByte::BYTE_3) {
            // No index requirement for the first pass (MSB).
        } else if constexpr (byte == HistByte::BYTE_2) {
            static_assert(TileIdx::isRowMajor, "Fix: THISTOGRAM (uint32) index should follow row major layout.");
            static_assert(TileIdx::Rows == 1, "Fix: THISTOGRAM BYTE_2 index must have exactly 1 row.");
            static_assert(TileIdx::Cols == TileSrc::Cols, "Fix: THISTOGRAM BYTE_2 index cols must match source cols.");
        } else if constexpr (byte == HistByte::BYTE_1) {
            static_assert(TileIdx::isRowMajor, "Fix: THISTOGRAM (uint32) index should follow row major layout.");
            static_assert(TileIdx::Rows == 2, "Fix: THISTOGRAM BYTE_1 index must have exactly 2 rows.");
            static_assert(TileIdx::Cols == TileSrc::Cols, "Fix: THISTOGRAM BYTE_1 index cols must match source cols.");
        } else if constexpr (byte == HistByte::BYTE_0) {
            static_assert(TileIdx::isRowMajor, "Fix: THISTOGRAM (uint32) index should follow row major layout.");
            static_assert(TileIdx::Rows == 3, "Fix: THISTOGRAM BYTE_0 index must have exactly 3 rows.");
            static_assert(TileIdx::Cols == TileSrc::Cols, "Fix: THISTOGRAM BYTE_0 index cols must match source cols.");
        }
        THistogramU32<byte, TileDst, TileSrc, TileIdx>(dst.data(), src.data(), idx.data(), src.GetValidRow(),
                                                       src.GetValidCol());
    }
}

} // namespace pto
#endif
