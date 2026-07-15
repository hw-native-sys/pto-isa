/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __PTO_TPRINT_A2A3__
#define __PTO_TPRINT_A2A3__

#include "pto/common/pto_tile.hpp"
#include <type_traits>

namespace pto {

template <typename T>
PTO_INTERNAL constexpr const __gm__ char* GetDTypeName()
{
    return "unknown";
}

#define DEFINE_TYPE_NAME_GROUP(name, ...)                                 \
    template <>                                                           \
    PTO_INTERNAL constexpr const __gm__ char* GetDTypeName<__VA_ARGS__>() \
    {                                                                     \
        return name;                                                      \
    }

DEFINE_TYPE_NAME_GROUP("uint32", std::uint32_t)
DEFINE_TYPE_NAME_GROUP("int32", std::int32_t)
DEFINE_TYPE_NAME_GROUP("uint16", std::uint16_t)
DEFINE_TYPE_NAME_GROUP("int16", std::int16_t)
DEFINE_TYPE_NAME_GROUP("uint8", std::uint8_t)
DEFINE_TYPE_NAME_GROUP("int8", std::int8_t)
DEFINE_TYPE_NAME_GROUP("float32", float)
DEFINE_TYPE_NAME_GROUP("float16", half)

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintValue(T& val, int col)
{
    if (col > 0) {
        cce::printf(" ");
    }

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if constexpr (Format == PrintFormat::Width8_Precision4) {
            cce::printf("%8.4f", static_cast<float>(val));
        } else if constexpr (Format == PrintFormat::Width8_Precision2) {
            cce::printf("%8.2f", static_cast<float>(val));
        } else if constexpr (Format == PrintFormat::Width10_Precision6) {
            cce::printf("%10.6f", static_cast<float>(val));
        }
    } else if constexpr (std::is_signed_v<T>) {
        if constexpr (Format == PrintFormat::Width10_Precision6) {
            cce::printf("%10d", static_cast<int>(val));
        } else {
            cce::printf("%8d", static_cast<int>(val));
        }
    } else if constexpr (std::is_unsigned_v<T>) {
        if constexpr (Format == PrintFormat::Width10_Precision6) {
            cce::printf("%10u", static_cast<unsigned int>(val));
        } else {
            cce::printf("%8u", static_cast<unsigned int>(val));
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupported data type for Print.");
    }
}

template <PrintFormat Format, typename TileDataIn>
PTO_INTERNAL void PrintTileRow(__ubuf__ typename TileDataIn::DType* src, int row, int validCols)
{
    using DType = typename TileDataIn::DType;
    for (int j = 0; j < TileDataIn::Cols; ++j) {
        DType val = *(src + GetTileOffset<TileDataIn>(row, j));
        PrintValue<Format>(val, j);
        if (j == validCols - 1 && validCols > 0 && validCols < TileDataIn::Cols) {
            cce::printf("|");
        }
    }
}

template <PrintFormat Format>
PTO_INTERNAL void PrintHorizontalSeparator(int totalCols, int validCols)
{
    for (int j = 0; j < totalCols; ++j) {
        if (j > 0) {
            cce::printf(" ");
        }
        if constexpr (Format == PrintFormat::Width10_Precision6) {
            cce::printf("----------"); // 10 dashes to match %10 width
        } else {
            cce::printf("--------"); // 8 dashes to match %8 width
        }

        if (j == validCols - 1 && validCols > 0 && validCols < totalCols) {
            cce::printf("|");
        }
    }
}

template <PrintFormat Format, typename TileDataIn>
__tf__ PTO_INTERNAL void TPrintTileImpl(typename TileDataIn::TileDType __in__ srcData, int validRows, int validCols)
{
    using DType = typename TileDataIn::DType;
    __ubuf__ DType* src = (__ubuf__ DType*)__cce_get_tile_ptr(srcData);

    cce::printf(
        "=== [TPRINT Tile] Data Type: %s, Layout: %s, TileType: %s ===\n", GetDTypeName<DType>(),
        GetLayoutName(TileDataIn::BFractal, TileDataIn::SFractal), "Vec");
    cce::printf("  Shape: [%d, %d], Valid Shape: [%d, %d]\n", TileDataIn::Rows, TileDataIn::Cols, validRows, validCols);
    for (int i = 0; i < TileDataIn::Rows; ++i) {
        PrintTileRow<Format, TileDataIn>(src, i, validCols);
        cce::printf("\n");
        if (i == validRows - 1 && validRows > 0 && validRows < TileDataIn::Rows) {
            PrintHorizontalSeparator<Format>(TileDataIn::Cols, validCols);
            cce::printf("\n");
        }
    }
}

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintRow(T* dataPtr, int i0, int i1, int i2, int r, int n4, int s0, int s1, int s2, int s3, int s4)
{
    for (int c = 0; c < n4; ++c) {
        size_t offset = i0 * s0 + i1 * s1 + i2 * s2 + r * s3 + c * s4;
        auto val = dataPtr[offset];
        PrintValue<Format>(val, c);
    }
    cce::printf("\n");
}

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintGlobalTensorNDOrDN(
    T* dataPtr, int n0, int n1, int n2, int n3, int n4, int s0, int s1, int s2, int s3, int s4)
{
    cce::printf("  Shape: [%d, %d, %d, %d, %d]\n", n0, n1, n2, n3, n4);
    // traverse the batch according to [n0, n1, n2]
    for (int i0 = 0; i0 < n0; ++i0) {
        for (int i1 = 0; i1 < n1; ++i1) {
            for (int i2 = 0; i2 < n2; ++i2) {
                cce::printf("  Batch [%d, %d, %d]:\n", i0, i1, i2);
                // print 2D matrix (Row: n3, Col: n4)
                for (int r = 0; r < n3; ++r) {
                    PrintRow<Format>(dataPtr, i0, i1, i2, r, n4, s0, s1, s2, s3, s4);
                }
            }
        }
    }
}

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintGlobalTensorNZ(
    T* dataPtr, int n0, int n1, int n2, int n3, int n4, int s0, int s1, int s2, int s3, int s4)
{
    // Shape<1, Cols/(C0Size/sizeof(T)), Rows/FractalRow, FractalRow, C0Size/sizeof(T)>
    // Stride<C*R, R*C0Size/sizeof(T), FractalRow*C0Size/sizeof(T), C0Size/sizeof(T), 1>
    int logical_rows = n2 * n3;
    int logical_cols = n1 * n4;
    cce::printf("  Logical Shape: [%d, %d]\n", logical_rows, logical_cols);
    for (int r = 0; r < logical_rows; ++r) {
        for (int c = 0; c < logical_cols; ++c) {
            int block_row = r / n3;
            int in_block_row = r % n3;
            int block_col = c / n4;
            int in_block_col = c % n4;
            size_t offset = block_row * s2 + block_col * s1 + in_block_row * s3 + in_block_col * s4;
            auto val = dataPtr[offset];
            PrintValue<Format>(val, c);
        }
        cce::printf("\n");
    }
}

template <PrintFormat Format, typename GlobalData>
PTO_INTERNAL void TPrintGlobalTensorImpl(GlobalData& src)
{
    using DType = typename GlobalData::DType;
    using ElemType = typename GlobalData::RawDType;

    int n0 = src.GetShape(GlobalTensorDim::DIM_0);
    int n1 = src.GetShape(GlobalTensorDim::DIM_1);
    int n2 = src.GetShape(GlobalTensorDim::DIM_2);
    int n3 = src.GetShape(GlobalTensorDim::DIM_3);
    int n4 = src.GetShape(GlobalTensorDim::DIM_4);

    int s0 = src.GetStride(GlobalTensorDim::DIM_0);
    int s1 = src.GetStride(GlobalTensorDim::DIM_1);
    int s2 = src.GetStride(GlobalTensorDim::DIM_2);
    int s3 = src.GetStride(GlobalTensorDim::DIM_3);
    int s4 = src.GetStride(GlobalTensorDim::DIM_4);

    typename GlobalData::DType* dataPtr = src.data();

    if constexpr (GlobalData::layout == Layout::ND || GlobalData::layout == Layout::DN) {
        cce::printf(
            "=== [TPRINT GlobalTensor] Data Type: %s, Layout: %s ===\n", GetDTypeName<ElemType>(),
            GlobalData::layout == Layout::ND ? "ND" : "DN");
        PrintGlobalTensorNDOrDN<Format>(dataPtr, n0, n1, n2, n3, n4, s0, s1, s2, s3, s4);
    } else if constexpr (GlobalData::layout == Layout::NZ) {
        cce::printf("=== [TPRINT GlobalTensor] Data Type: %s, Layout: %s ===\n", GetDTypeName<ElemType>(), "NZ");
        PrintGlobalTensorNZ<Format>(dataPtr, n0, n1, n2, n3, n4, s0, s1, s2, s3, s4);
    } else {
        static_assert(sizeof(GlobalData) == 0, "Unsupported GlobalTensor layout.");
    }
}

template <PrintFormat Format, typename T>
PTO_INTERNAL void TPRINT_IMPL(T& src)
{
    pipe_barrier(PIPE_ALL);
    if constexpr (is_tile_data_v<T>) {
        static_assert(T::Loc == TileType::Vec, "TileType of source tile must be Vec.");

        int validRows = src.GetValidRow();
        int validCols = src.GetValidCol();
        TPrintTileImpl<Format, T>(src.data(), validRows, validCols);
        return;
    } else if constexpr (is_global_data_v<T>) {
        TPrintGlobalTensorImpl<Format, T>(src);
        return;
    } else {
        static_assert(sizeof(T) == 0, "TPRINT: Only Vec Tile and GlobalTensor are supported without tmp buffer.");
    }
}

template <PrintFormat Format, typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TPrintCopyAcc2GM(
    typename GlobalData::DType __out__* tmp, typename TileData::TileDType __in__ src)
{
    using T = typename TileData::DType;
    __cc__ T* srcAddr = (__cc__ T*)__cce_get_tile_ptr(src);
    __gm__ T* tmpAddr = reinterpret_cast<__gm__ T*>(tmp);

    constexpr uint16_t c0 = 16;
    constexpr uint8_t nz2ndEn = 1;
    constexpr uint16_t ndNum = 1;
    uint16_t srcStride = TileData::Rows;
    uint16_t dstNdStride = (uint16_t)(TileData::Numel);
    uint16_t srcNdStride = (uint16_t)(TileData::Numel * c0);
    if constexpr (TileData::Compact == CompactMode::Normal) {
        srcStride = (TileData::Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        srcNdStride = srcStride * TileData::Cols * c0;
    }

    uint64_t xmReg = ((TileData::Cols & 0xfff) << 4) | (static_cast<uint64_t>(TileData::Rows & 0xffff) << 16) |
                     (static_cast<uint64_t>(TileData::Cols & 0xffffffff) << 32);
    uint64_t xtReg = srcStride | ((static_cast<uint64_t>(QuantMode_t::NoQuant >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     ((static_cast<uint64_t>(STPhase::Unspecified) & 0x3) << 32) |
                     ((static_cast<uint64_t>(QuantMode_t::NoQuant) & 0x1f) << 34) |
                     ((static_cast<uint64_t>(ReluPreMode::NoRelu) & 0x7) << 39) |
                     ((static_cast<uint64_t>(nz2ndEn) & 0x1) << 43);
    uint64_t config = ndNum | (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) |
                      (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);
    set_nd_para(config);
    copy_matrix_cc_to_gm(tmpAddr, srcAddr, xmReg, xtReg);
}

#ifdef PTO_NPU_ARCH_A2A3
template <PrintFormat Format, typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TPrintCopyMat2GM(
    typename GlobalData::DType __out__* tmp, typename TileData::TileDType __in__ src)
{
    using T = typename TileData::DType;
    __cbuf__ T* srcAddr = (__cbuf__ T*)__cce_get_tile_ptr(src);
    __gm__ T* tmpAddr = reinterpret_cast<__gm__ T*>(tmp);

    uint16_t lenBurst = (TileData::Numel * sizeof(T)) >> SHIFT_BLOCK_BYTE;
    copy_cbuf_to_gm(tmpAddr, srcAddr, (uint8_t)0, 1, lenBurst, 0, 0);
}
#endif

template <PrintFormat Format, typename TileData>
PTO_INTERNAL void TPrintMatOrAccTileByTmp(__gm__ typename TileData::DType* tmp, int validRows, int validCols)
{
    using T = typename TileData::DType;
    cce::printf(
        "=== [TPRINT Tile] Data Type: %s, Layout: %s, TileType: %s ===\n", GetDTypeName<T>(),
        GetLayoutName(TileData::BFractal, TileData::SFractal), GetTileTypeName<TileData::Loc>());
    cce::printf("  Shape: [%d, %d], Valid Shape: [%d, %d]\n", TileData::Rows, TileData::Cols, validRows, validCols);
    for (int i = 0; i < TileData::Rows; ++i) {
        for (int j = 0; j < TileData::Cols; ++j) {
            T val = *(tmp + i * TileData::Cols + j);
            PrintValue<Format>(val, j);
            if (j == validCols - 1 && validCols > 0 && validCols < TileData::Cols) {
                cce::printf("|");
            }
        }
        cce::printf("\n");
        if (i == validRows - 1 && validRows > 0 && validRows < TileData::Rows) {
            PrintHorizontalSeparator<Format>(TileData::Cols, validCols);
            cce::printf("\n");
        }
    }
}

template <PrintFormat Format, typename TileData, typename GlobalData>
PTO_INTERNAL void TPRINT_IMPL(TileData& src, GlobalData& tmp)
{
    pipe_barrier(PIPE_ALL);
    static_assert(is_tile_data_v<TileData>, "Fix: TPRINT First parameter must be Tile type.");
    static_assert(is_global_data_v<GlobalData>, "Fix: TPRINT Second parameter must be GlobalTensor type.");

    using T = typename TileData::DType;
    __gm__ T* tmpData = reinterpret_cast<__gm__ T*>(tmp.data());

    if constexpr (TileData::Loc == TileType::Mat) {
#ifdef PTO_NPU_ARCH_A2A3
        TPrintCopyMat2GM<Format, GlobalData, TileData>(tmp.data(), src.data());
#else
        static_assert(sizeof(TileData) == 0, "Fix: TPRINT Mat Tile is not supported in A5.");
#endif
    } else if constexpr (TileData::Loc == TileType::Acc) {
        TPrintCopyAcc2GM<Format, GlobalData, TileData>(tmp.data(), src.data());
    } else if constexpr (TileData::Loc == TileType::Vec) {
        TPRINT_IMPL<Format>(src);
    } else {
        static_assert(sizeof(TileData) == 0, "Fix: TPRINT TileType must be Mat / Vec / Acc.");
    }
    pipe_barrier(PIPE_ALL);
    int validRows = src.GetValidRow();
    int validCols = src.GetValidCol();
    TPrintMatOrAccTileByTmp<Format, TileData>(tmpData, validRows, validCols);
    pipe_barrier(PIPE_ALL);
}
} // namespace pto

#endif
