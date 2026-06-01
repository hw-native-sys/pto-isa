/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MSCATTER_HPP
#define MSCATTER_HPP

#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include "common.hpp"

namespace pto {

enum class ScatterAtomicOp : uint8_t
{
    None = 0,
    Add = 1,
    Max = 2,
    Min = 3
};

enum class ScatterOOB : uint8_t
{
    Undefined = 0,
    Skip = 1,
    Clamp = 2,
    Wrap = 3
};

#ifndef PTO_COALESCE_ENUM_DEFINED
#define PTO_COALESCE_ENUM_DEFINED
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1
};
#endif

template <typename T>
struct IsValidMScatterDType {
    static constexpr bool value = std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                                  std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                  std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float>;
};

template <typename T, ScatterAtomicOp Atomic, Coalesce Mode>
struct IsValidMScatterAtomic {
    static constexpr bool value =
        (Atomic == ScatterAtomicOp::None) ||
        ((Atomic == ScatterAtomicOp::Add) &&
         (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, half> ||
          std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t>));
};

template <typename Tile>
struct IsMScatterNDTile {
    static constexpr bool value = Tile::isRowMajor && (Tile::SFractal == SLayout::NoneBox);
};

template <typename Tile>
struct IsMScatterNZTile {
    static constexpr bool value =
        !Tile::isRowMajor && (Tile::SFractal == SLayout::RowMajor) && (Tile::SFractalSize == TileConfig::fractalABSize);
};

template <ScatterOOB Oob>
AICORE PTO_INLINE uint32_t mscatter_remap(uint32_t idx, uint32_t cap, uint32_t &doWrite)
{
    if constexpr (Oob == ScatterOOB::Undefined) {
        doWrite = 1u;
        return idx;
    } else if constexpr (Oob == ScatterOOB::Skip) {
        doWrite = (idx < cap) ? 1u : 0u;
        return idx;
    } else if constexpr (Oob == ScatterOOB::Clamp) {
        doWrite = 1u;
        return (idx >= cap) ? (cap - 1u) : idx;
    } else {
        doWrite = 1u;
        return idx % cap;
    }
}

template <typename T>
AICORE PTO_INLINE void MScatterRowDma(__gm__ T *dst, __ubuf__ T *src, uint32_t lenBytes)
{
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 4) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    }
}

template <typename T>
AICORE PTO_INLINE void MScatterRowMultiDma(__gm__ T *dst, __ubuf__ T *src, uint16_t nBurst, uint32_t lenBytes,
                                           uint32_t ubGapBlocks, uint32_t gmGapBytes)
{
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, nBurst, lenBytes, 0, 0, ubGapBlocks, gmGapBytes);
    } else if constexpr (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, nBurst, lenBytes, 0, 0, ubGapBlocks, gmGapBytes);
    } else if constexpr (sizeof(T) == 4) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBytes, 0, 0, ubGapBlocks, gmGapBytes);
    }
}

template <typename T>
AICORE PTO_INLINE void MScatterAtomicAddSet()
{
    if constexpr (std::is_same_v<T, float>) {
        set_atomic_f32();
    } else if constexpr (std::is_same_v<T, half>) {
        set_atomic_f16();
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        set_atomic_bf16();
    } else if constexpr (std::is_same_v<T, int32_t>) {
        set_atomic_s32();
    } else if constexpr (std::is_same_v<T, int16_t>) {
        set_atomic_s16();
    } else if constexpr (std::is_same_v<T, int8_t>) {
        set_atomic_s8();
    }
    set_atomic_add();
}

AICORE PTO_INLINE void MScatterAtomicNone()
{
    set_atomic_none();
}

template <typename T>
AICORE PTO_INLINE uint64_t MScatterNZGmOffset(uint32_t logicalRow, uint32_t logicalCol, int gShape0, int gShape1,
                                              int gStride0, int gStride1, int gStride2, int gStride3, int gStride4)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kFRow = FRACTAL_NZ_ROW;
    const uint32_t blockColCombined = logicalCol / kC0;
    const uint32_t colInBlock = logicalCol - blockColCombined * kC0;
    const uint32_t blockRow = logicalRow / kFRow;
    const uint32_t rowInBlock = logicalRow - blockRow * kFRow;
    const uint32_t blockColOuter0 = (gShape0 == 1) ? 0u : (blockColCombined / (uint32_t)gShape1);
    const uint32_t blockColOuter1 = (gShape0 == 1) ? blockColCombined : (blockColCombined - blockColOuter0 * gShape1);
    return (uint64_t)blockColOuter0 * (uint64_t)gStride0 + (uint64_t)blockColOuter1 * (uint64_t)gStride1 +
           (uint64_t)blockRow * (uint64_t)gStride2 + (uint64_t)rowInBlock * (uint64_t)gStride3 +
           (uint64_t)colInBlock * (uint64_t)gStride4;
}

AICORE PTO_INLINE float MScatterBf16BitsToFloat(uint16_t bits)
{
    constexpr uint8_t shiftOffset = 16;
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = static_cast<uint32_t>(bits) << shiftOffset;
    return conv.f;
}

AICORE PTO_INLINE uint16_t MScatterFloatToBf16Bits(float value)
{
    constexpr uint8_t shiftOffset = 16;
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = value;
    uint32_t lsb = (conv.u >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((conv.u + bias) >> shiftOffset);
}

AICORE PTO_INLINE uint16_t MScatterBf16ToBits(bfloat16_t value)
{
    union {
        bfloat16_t bf;
        uint16_t u;
    } conv;
    conv.bf = value;
    return conv.u;
}

AICORE PTO_INLINE bfloat16_t MScatterBitsToBf16(uint16_t bits)
{
    union {
        uint16_t u;
        bfloat16_t bf;
    } conv;
    conv.u = bits;
    return conv.bf;
}

template <ScatterAtomicOp Atomic, typename T>
AICORE PTO_INLINE void MScatterScalarStore(__gm__ T *dst, T value)
{
    if constexpr (Atomic == ScatterAtomicOp::Add) {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            uint16_t prevBits = MScatterBf16ToBits(*dst);
            uint16_t valBits = MScatterBf16ToBits(value);
            float prev = MScatterBf16BitsToFloat(prevBits);
            float vsrc = MScatterBf16BitsToFloat(valBits);
            float sum = prev + vsrc;
            *dst = MScatterBitsToBf16(MScatterFloatToBf16Bits(sum));
        } else if constexpr (std::is_same_v<T, half>) {
            float prev = static_cast<float>(*dst);
            float vsrc = static_cast<float>(value);
            *dst = static_cast<T>(prev + vsrc);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            int32_t prev = static_cast<int32_t>(*dst);
            int32_t vsrc = static_cast<int32_t>(value);
            *dst = static_cast<T>(prev + vsrc);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            int32_t prev = static_cast<int32_t>(*dst);
            int32_t vsrc = static_cast<int32_t>(value);
            *dst = static_cast<T>(prev + vsrc);
        } else {
            *dst = static_cast<T>(*dst + value);
        }
    } else {
        *dst = value;
    }
}

template <ScatterAtomicOp Atomic, ScatterOOB Oob, typename T, typename TIdx, typename SrcTile, typename IdxTile>
__tf__ AICORE void MScatterRowImpl(__gm__ T *tablePtr, typename SrcTile::TileDType __in__ src,
                                   typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                   uint32_t tableRows, uint32_t tableRowStride)
{
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();

    if constexpr (Atomic == ScatterAtomicOp::Add) {
        MScatterAtomicAddSet<T>();
    }

    PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();

    const uint32_t lenBytes = validCol * sizeof(T);
    constexpr uint32_t kRowStride = SrcTile::RowStride;

    for (uint32_t r = 0; r < validRow; r++) {
        uint32_t rawIdx = static_cast<uint32_t>(idxPtr[r]);
        uint32_t doWrite;
        uint32_t safeIdx = mscatter_remap<Oob>(rawIdx, tableRows, doWrite);
        if (doWrite) {
            __gm__ T *dstRow = tablePtr + static_cast<uint64_t>(safeIdx) * tableRowStride;
            __ubuf__ T *srcRow = srcPtr + r * kRowStride;
            MScatterRowDma<T>(dstRow, srcRow, lenBytes);
            if constexpr (Atomic == ScatterAtomicOp::None) {
                PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            }
        }
    }

    if constexpr (Atomic == ScatterAtomicOp::Add) {
        PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        MScatterAtomicNone();
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        PtoSetWaitFlag<PIPE_S, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
    PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
}

template <ScatterAtomicOp Atomic, ScatterOOB Oob, typename T, typename TIdx, typename SrcTile, typename IdxTile>
__tf__ AICORE void MScatterRowNzImpl(__gm__ T *tablePtr, typename SrcTile::TileDType __in__ src,
                                     typename IdxTile::TileDType __in__ indices, uint32_t validRow, int gShape0,
                                     int gShape1, int gShape2, int gStride0, int gStride1, int gStride2, int gStride3)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kFRow = FRACTAL_NZ_ROW;
    constexpr uint32_t kFractalRowBytes = kC0 * sizeof(T);

    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();

    if constexpr (Atomic == ScatterAtomicOp::Add) {
        MScatterAtomicAddSet<T>();
    }

    PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();

    const uint32_t tableLogicalRows = (uint32_t)gShape2 * kFRow;
    const uint32_t gmGapBytes = ((uint32_t)gStride1 - kC0) * (uint32_t)sizeof(T);
    constexpr uint32_t ubGapBlocks = (uint32_t)SrcTile::Rows - 1u;
    const int64_t tileOuterStrideElem = (int64_t)gShape1 * (int64_t)SrcTile::Rows * (int64_t)kC0;

    for (uint32_t r = 0; r < validRow; r++) {
        uint32_t rawIdx = static_cast<uint32_t>(idxPtr[r]);
        uint32_t doWrite;
        uint32_t safeIdx = mscatter_remap<Oob>(rawIdx, tableLogicalRows, doWrite);
        if (doWrite) {
            const uint32_t dstBlockRow = safeIdx / kFRow;
            const uint32_t dstRowInBlock = safeIdx - dstBlockRow * kFRow;
            const uint32_t srcBlockRow = r / kFRow;
            const uint32_t srcRowInBlock = r - srcBlockRow * kFRow;

            for (uint32_t i = 0; i < (uint32_t)gShape0; i++) {
                __gm__ T *dstAddr = tablePtr + (int64_t)i * (int64_t)gStride0 +
                                    (int64_t)dstBlockRow * (int64_t)gStride2 +
                                    (int64_t)dstRowInBlock * (int64_t)gStride3;
                __ubuf__ T *srcAddr = srcPtr + (int64_t)i * tileOuterStrideElem +
                                      (int64_t)srcBlockRow * (int64_t)kFRow * (int64_t)kC0 +
                                      (int64_t)srcRowInBlock * (int64_t)kC0;
                MScatterRowMultiDma<T>(dstAddr, srcAddr, (uint16_t)gShape1, kFractalRowBytes, ubGapBlocks, gmGapBytes);
            }
            if constexpr (Atomic == ScatterAtomicOp::None) {
                PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            }
        }
    }

    if constexpr (Atomic == ScatterAtomicOp::Add) {
        PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        MScatterAtomicNone();
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        PtoSetWaitFlag<PIPE_S, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
    PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
}

template <ScatterAtomicOp Atomic, ScatterOOB Oob, typename T, typename TIdx, typename SrcTile, typename IdxTile>
__tf__ AICORE void MScatterElemImpl(__gm__ T *tablePtr, typename SrcTile::TileDType __in__ src,
                                    typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                    uint32_t tableSize)
{
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();

    constexpr uint32_t kSrcRowStride = SrcTile::RowStride;
    constexpr uint32_t kIdxRowStride = IdxTile::RowStride;

    for (uint32_t r = 0; r < validRow; r++) {
        const uint32_t srcRowOff = r * kSrcRowStride;
        const uint32_t idxRowOff = r * kIdxRowStride;
        for (uint32_t c = 0; c < validCol; c++) {
            const uint32_t idxOff = idxRowOff + c;
            const uint32_t srcOff = srcRowOff + c;
            uint32_t rawIdx = static_cast<uint32_t>(idxPtr[idxOff]);
            uint32_t doWrite;
            uint32_t safeIdx = mscatter_remap<Oob>(rawIdx, tableSize, doWrite);
            if (doWrite) {
                MScatterScalarStore<Atomic, T>(tablePtr + safeIdx, srcPtr[srcOff]);
            }
        }
    }

    PtoSetWaitFlag<PIPE_S, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_MTE2>();
    PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
}

template <ScatterAtomicOp Atomic, ScatterOOB Oob, typename T, typename TIdx, typename SrcTile, typename IdxTile>
__tf__ AICORE void MScatterElemNzImpl(__gm__ T *tablePtr, typename SrcTile::TileDType __in__ src,
                                      typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                      uint32_t tableSize, int gShape0, int gShape1, int gStride0, int gStride1,
                                      int gStride2, int gStride3, int gStride4, uint32_t nLogicalCols)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();

    constexpr uint32_t kIdxRowStride = IdxTile::RowStride;
    const uint32_t nColBlocks = (validCol + kC0 - 1u) / kC0;
    const uint32_t kSrcColBlockStride = (uint32_t)SrcTile::Rows * kC0;

    for (uint32_t bcol = 0; bcol < nColBlocks; bcol++) {
        const uint32_t cBase = bcol * kC0;
        const uint32_t cLimit = (cBase + kC0 < validCol) ? (cBase + kC0) : validCol;
        const uint32_t kInBlock = cLimit - cBase;
        __ubuf__ T *srcBlockBase = srcPtr + (uint64_t)bcol * (uint64_t)kSrcColBlockStride;
        for (uint32_t r = 0; r < validRow; r++) {
            const uint32_t idxRowOff = r * kIdxRowStride;
            __ubuf__ T *srcRowBase = srcBlockBase + (uint64_t)r * (uint64_t)kC0;
            for (uint32_t cInner = 0; cInner < kInBlock; cInner++) {
                const uint32_t c = cBase + cInner;
                const uint32_t idxOff = idxRowOff + c;
                uint32_t rawIdx = static_cast<uint32_t>(idxPtr[idxOff]);
                uint32_t doWrite;
                uint32_t safeIdx = mscatter_remap<Oob>(rawIdx, tableSize, doWrite);
                if (doWrite) {
                    const uint32_t logicalRow = safeIdx / nLogicalCols;
                    const uint32_t logicalCol = safeIdx - logicalRow * nLogicalCols;
                    const uint64_t dstOff = MScatterNZGmOffset<T>(logicalRow, logicalCol, gShape0, gShape1, gStride0,
                                                                  gStride1, gStride2, gStride3, gStride4);
                    MScatterScalarStore<Atomic, T>(tablePtr + dstOff, srcRowBase[cInner]);
                }
            }
        }
    }

    PtoSetWaitFlag<PIPE_S, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_MTE2>();
    PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
}

template <Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalTable, typename SrcTile, typename IdxTile>
PTO_INTERNAL void MScatterCheck()
{
    using T = typename SrcTile::DType;
    using TIdx = typename IdxTile::DType;

    static_assert(IsValidMScatterDType<T>::value,
                  "MSCATTER A2/A3 data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float.");
    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MSCATTER A2/A3 index type must be int32_t or uint32_t.");
    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MSCATTER A2/A3 destination table must be a GM GlobalTensor with element type matching the source.");
    static_assert(SrcTile::Loc == TileType::Vec, "MSCATTER A2/A3 source must be a Vec tile (UB).");
    static_assert(IdxTile::Loc == TileType::Vec, "MSCATTER A2/A3 indices must be a Vec tile (UB).");

    static_assert(IdxTile::isRowMajor, "MSCATTER A2/A3 index tile must be BLayout::RowMajor.");
    static_assert(IdxTile::SFractal == SLayout::NoneBox, "MSCATTER A2/A3 index tile must be ND (SLayout::NoneBox).");

    constexpr bool kIsTableND = (GlobalTable::layout == Layout::ND);
    constexpr bool kIsTableNZ = (GlobalTable::layout == Layout::NZ);
    constexpr bool kIsSrcND = IsMScatterNDTile<SrcTile>::value;
    constexpr bool kIsSrcNZ = IsMScatterNZTile<SrcTile>::value;

    static_assert(kIsTableND || kIsTableNZ, "MSCATTER A2/A3 table must use Layout::ND or Layout::NZ.");
    static_assert((kIsTableND && kIsSrcND) || (kIsTableNZ && kIsSrcNZ),
                  "MSCATTER A2/A3 layout pairing must be either:\n"
                  "  (a) GM Layout::ND + UB tile (BLayout::RowMajor + SLayout::NoneBox), or\n"
                  "  (b) GM Layout::NZ + UB tile (BLayout::ColMajor + SLayout::RowMajor + SFractalSize=512).");

    static_assert(SrcTile::Cols * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "MSCATTER A2/A3 source tile padded Cols*sizeof(T) must be 32-byte aligned.");

    if constexpr (kIsTableNZ) {
        static_assert(GlobalTable::staticShape[3] == FRACTAL_NZ_ROW,
                      "MSCATTER A2/A3 NZ table requires staticShape[3] == FRACTAL_NZ_ROW (16).");
        static_assert(GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T),
                      "MSCATTER A2/A3 NZ table requires staticShape[4] == 32 / sizeof(T).");
        static_assert(SrcTile::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0,
                      "MSCATTER A2/A3 NZ source tile Cols must be a multiple of C0 (= 32 / sizeof(T)).");
        static_assert(SrcTile::Rows % FRACTAL_NZ_ROW == 0,
                      "MSCATTER A2/A3 NZ source tile Rows must be a multiple of FRACTAL_NZ_ROW (16).");
    }

    static_assert(IsValidMScatterAtomic<T, Atomic, Mode>::value,
                  "MSCATTER A2/A3 atomic operation: Add only valid for int8/int16/int32/half/bfloat16/float; "
                  "Max/Min not supported.");

    constexpr int kSrcValidR = SrcTile::ValidRow;
    constexpr int kSrcValidC = SrcTile::ValidCol;
    constexpr int kIdxValidR = IdxTile::ValidRow;
    constexpr int kIdxValidC = IdxTile::ValidCol;

    if constexpr (Mode == Coalesce::Row) {
        if constexpr (kSrcValidR > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidR == 1 && kIdxValidC == kSrcValidR,
                          "MSCATTER A2/A3 Coalesce::Row requires index tile valid shape [1, R].");
        }
    } else {
        if constexpr (kSrcValidR > 0 && kIdxValidR > 0) {
            static_assert(kIdxValidR == kSrcValidR,
                          "MSCATTER A2/A3 Coalesce::Elem requires index tile ValidRow == source ValidRow.");
        }
        if constexpr (kSrcValidC > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidC == kSrcValidC,
                          "MSCATTER A2/A3 Coalesce::Elem requires index tile ValidCol == source ValidCol.");
        }
    }
}

template <Coalesce Mode = Coalesce::Row, ScatterAtomicOp Atomic = ScatterAtomicOp::None,
          ScatterOOB Oob = ScatterOOB::Undefined, typename GlobalTable, typename SrcTile, typename IdxTile>
PTO_INTERNAL void MSCATTER_IMPL(GlobalTable &table, SrcTile &src, IdxTile &indices)
{
    using T = typename SrcTile::DType;
    using TIdx = typename IdxTile::DType;

    MScatterCheck<Mode, Atomic, GlobalTable, SrcTile, IdxTile>();

    __gm__ T *tablePtr = reinterpret_cast<__gm__ T *>(table.data());

    const uint32_t validRow = src.GetValidRow();
    const uint32_t validCol = src.GetValidCol();

    constexpr bool kIsTableNZ = (GlobalTable::layout == Layout::NZ);

    if constexpr (kIsTableNZ) {
        const int gShape0 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_0));
        const int gShape1 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_1));
        const int gShape2 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_2));
        const int gStride0 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_0));
        const int gStride1 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_1));
        const int gStride2 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_2));
        const int gStride3 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_3));
        const int gStride4 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_4));

        if constexpr (Mode == Coalesce::Row) {
            MScatterRowNzImpl<Atomic, Oob, T, TIdx, SrcTile, IdxTile>(tablePtr, src.data(), indices.data(), validRow,
                                                                      gShape0, gShape1, gShape2, gStride0, gStride1,
                                                                      gStride2, gStride3);
        } else {
            constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
            const uint32_t nLogicalCols = static_cast<uint32_t>(gShape0 * gShape1) * kC0;
            const uint32_t tableSize = static_cast<uint32_t>(gShape2 * FRACTAL_NZ_ROW) * nLogicalCols;
            MScatterElemNzImpl<Atomic, Oob, T, TIdx, SrcTile, IdxTile>(
                tablePtr, src.data(), indices.data(), validRow, validCol, tableSize, gShape0, gShape1, gStride0,
                gStride1, gStride2, gStride3, gStride4, nLogicalCols);
        }
    } else {
        if constexpr (Mode == Coalesce::Row) {
            const uint32_t tableRows =
                static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                      table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3));
            const uint32_t tableRowStride = static_cast<uint32_t>(table.GetStride(GlobalTensorDim::DIM_3));
            MScatterRowImpl<Atomic, Oob, T, TIdx, SrcTile, IdxTile>(tablePtr, src.data(), indices.data(), validRow,
                                                                    validCol, tableRows, tableRowStride);
        } else {
            const uint32_t tableSize =
                static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                      table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3) *
                                      table.GetShape(GlobalTensorDim::DIM_4));
            MScatterElemImpl<Atomic, Oob, T, TIdx, SrcTile, IdxTile>(tablePtr, src.data(), indices.data(), validRow,
                                                                     validCol, tableSize);
        }
    }
}

} // namespace pto

#endif // MSCATTER_HPP
