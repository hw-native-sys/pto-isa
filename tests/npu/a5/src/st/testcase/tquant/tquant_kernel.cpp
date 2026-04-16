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
#include <pto/common/constants.hpp>
#include <pto/npu/a5/TQuant.hpp>
#include "acl/acl.h"
#include <type_traits>

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))
#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))

namespace TQuantTest {

constexpr int TQUANT_A5_UB_ALIGN_BYTES = 32;
constexpr int TQUANT_A5_UB_SIZE_BYTES = 256 * 1024;

// FP32 --> MXFP8
// Quantize fp32 tile to fp8 (e4m3) and exponent-only (e8m0).
// Pad columns to multiples of 32 using min fill to avoid reading garbage.
template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
__global__ AICORE void runTQuant(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                 __gm__ float __in__ *src)
{
    constexpr int paddedCols = PTO_CEIL(validCols, 32);
    constexpr int groupedCols_flattened = validRows * (paddedCols / 32);
    constexpr int groupedCols_valid = paddedCols / 32;
    // Pad 1D tile cols for 32-byte row alignment (required by Vec RowMajor NoneBox tiles)
    constexpr int groupedCols_flat_f32 = PTO_CEIL(groupedCols_flattened, 8); // float:  8 elems = 32 B
    constexpr int groupedCols_flat_u8 = PTO_CEIL(groupedCols_flattened, 32); // uint8: 32 elems = 32 B
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, groupedCols_flattened>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstFP8Global = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flat_u8, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                           512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, validRows, paddedCols, BLayout::RowMajor, validRows, paddedCols,
                            SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, float, 1, groupedCols_flat_f32, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, float, 1, groupedCols_flat_f32, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    ScalingTile scalingTile(1, groupedCols_flattened);
    DstFP8Tile fp8Tile;
    DstE8Tile e8Tile(1, groupedCols_flattened);
    MaxTile maxPerGpTile(1, groupedCols_flattened);

    SrcGlobal srcGlobal(src);
    DstE8Global e8Global(out_e8m0);
    DstFP8Global fp8Global((__gm__ int8_t *)out_fp8);

    TASSIGN(srcTile, 0x0);          // 128 KB = 0x20000
    TASSIGN(maxPerGpTile, 0x20100); // 4 KB   = 0x1000 (Max and Scaling can overlap)
    TASSIGN(scalingTile, 0x21820);  // 8 KB   = 0x2000
    TASSIGN(e8Tile, 0x24100);       // 1 KB   = 0x400
    TASSIGN(fp8Tile, 0x25100);      // 32  KB = 0x8000
    TLOAD(srcTile, srcGlobal);

    if constexpr (mode == 0) {
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

        TSTORE(e8Global, e8Tile);
        TSTORE(fp8Global, fp8Tile);
    } else {
        // NZ mode: plain TQUANT (ND output), then TMOV ND→ZZ for e8m0, TMOV ND→NZ for fp8.
        constexpr int groupedCols_valid = paddedCols / 32;

        // E8M0 ZZ destination: 2D with fractalMxSize=32 for [16,2] inner box
        using E8ZzTile = Tile<TileType::Vec, uint8_t, validRows, groupedCols_valid, BLayout::RowMajor, -1, -1,
                              SLayout::RowMajor, 32, PadValue::Zero>;
        // 1D flat alias at same address for TSTORE (no dispatch path for isRowMajor + SLayout::RowMajor)
        using E8StoreTile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1,
                                 SLayout::NoneBox, 512, PadValue::Zero>;
        // Scratch tile for TMOV ZZ gather-index generation
        constexpr int tmpBufSize = (16 + (validRows / 16) * (groupedCols_valid / 2) + 16) * sizeof(uint16_t);
        constexpr int tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);
        using TmpTile = Tile<TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                             512, PadValue::Zero>;

        // TMOV fp8 ND→NZ
        constexpr int virtualRow = PTO_CEIL(validRows, FRACTAL_NZ_ROW) + 1; // NZ + 1 for reducing bank conflict
        using DstNZ_int8 = Tile<TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, validRows, paddedCols,
                                SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
        DstNZ_int8 fp8TileNZ;
        E8ZzTile e8ZzTile(validRows, groupedCols_valid);
        E8StoreTile e8StoreTile(1, groupedCols_flattened);
        TmpTile tmpTile(1, tmpBufSizeAligned);

        TASSIGN(fp8TileNZ, 0x0);       // reuse src tile address since it's consumed before TMOV
        TASSIGN(e8ZzTile, 0x20100);    // reuse maxPerGpTile address (consumed after TQUANT)
        TASSIGN(e8StoreTile, 0x20100); // alias for flat TSTORE at same address
        TASSIGN(tmpTile, 0x30100);     // scratch for ZZ indices

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Step 1: Plain TQUANT (ND output), same as mode==0
        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);
        // Step 2: Convert E8M0 exponents ND → ZZ layout
        TMOV(e8ZzTile, e8Tile, tmpTile);
        // Step 3: Convert FP8 data ND → NZ layout
        TMOV(fp8TileNZ, fp8Tile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Store ZZ-reordered e8m0 exponents
        TSTORE(e8Global, e8StoreTile);

        using DstFp8GlobalNZ = GlobalTensor<int8_t, TileShape2D<int8_t, validRows, paddedCols, Layout::NZ>,
                                            BaseShape2D<int8_t, validRows, paddedCols, Layout::NZ>, Layout::NZ>;
        DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t *)out_fp8);
        TSTORE(fp8GlobalNZ, fp8TileNZ);
    }
}

// FP32 --> INT8 SYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Sym(__gm__ int8_t __out__ *out_s8, __gm__ float __in__ *src,
                                        __gm__ float __in__ *scale)
{
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(int8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, int8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;
    // tmp is unused on A5 (native broadcast); declared only to keep the interface identical to A2/A3.
    using TmpTile = Tile<TileType::Vec, float, validRows, BLOCK_BYTE_SIZE / sizeof(float), BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstS8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);
    TmpTile tmpTile(validRows, BLOCK_BYTE_SIZE / sizeof(float));

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_s8);
    ParaGlobal scaleGlobal(scale);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstS8Tile, 0x0);
    TASSIGN(scaleTile, 0x20100);
    TASSIGN(tmpTile, 0x28000);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::INT8_SYM, DstTile, SrcTile, ParaTile>(dstS8Tile, srcTile, scaleTile, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstS8Tile);
}

// FP32 --> INT8 ASYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Asym(__gm__ uint8_t __out__ *out_u8, __gm__ float __in__ *src,
                                         __gm__ float __in__ *scale, __gm__ float __in__ *offset)
{
    // pad each row to multiple of 32 elements
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(uint8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, uint8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;
    // tmp is unused on A5 (native broadcast); declared only to keep the interface identical to A2/A3.
    using TmpTile = Tile<TileType::Vec, float, validRows, BLOCK_BYTE_SIZE / sizeof(float), BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstU8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);
    ParaTile offsetTile(validRows, 1);
    TmpTile tmpTile(validRows, BLOCK_BYTE_SIZE / sizeof(float));

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_u8);
    ParaGlobal scaleGlobal(scale);
    ParaGlobal offsetGlobal(offset);

    TASSIGN(srcTile, 0x0);       // 128 KB
    TASSIGN(dstU8Tile, 0x20100); // 32 KB
    TASSIGN(scaleTile, 0x30100);
    TASSIGN(offsetTile, 0x32500);
    TASSIGN(tmpTile, 0x34000);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);
    TLOAD(offsetTile, offsetGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::INT8_ASYM, DstTile, SrcTile, ParaTile>(dstU8Tile, srcTile, scaleTile, tmpTile, &offsetTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstU8Tile);
}

template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream)
{
    runTQuant<validRows, validCols, mode, scaleAlg><<<1, nullptr, stream>>>(dst_exp, dst, src);
}

template <int validRows, int validCols, int mode, pto::QuantType quantType>
void LaunchTQuantInt8(std::conditional_t<quantType == pto::QuantType::INT8_SYM, int8_t, uint8_t> *dst, float *src,
                      float *scale, void *stream, float *offset = nullptr)
{
    if constexpr (quantType == pto::QuantType::INT8_SYM) {
        runTQuantInt8Sym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale);
    } else {
        runTQuantInt8Asym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale, offset);
    }
}

// BF16 --> MXFP8
template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
__global__ AICORE void runTQuantBF16(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                     __gm__ bfloat16_t __in__ *src)
{
    constexpr int paddedCols = PTO_CEIL(validCols, 32);
    constexpr int groupedCols_flattened = validRows * (paddedCols / 32);
    constexpr int groupedCols_valid = paddedCols / 32;
    // Static col counts padded to 32-byte row alignment (required by Vec RowMajor NoneBox tiles)
    constexpr int groupedCols_e8_static = PTO_CEIL(groupedCols_valid, 32);        // uint8:  32 elems = 32 B
    constexpr int groupedCols_b16_static = PTO_CEIL(groupedCols_valid, 16);       // bf16:   16 elems = 32 B
    constexpr int groupedCols_flat_aligned = PTO_CEIL(groupedCols_flattened, 32); // for 1D store alias
    using SrcGlobal =
        GlobalTensor<bfloat16_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, groupedCols_flattened>, pto::Stride<1, 1, 1, groupedCols_flattened, 1>>;
    using DstFP8Global = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, bfloat16_t, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                         512, PadValue::Zero>;
    // 2D tiles for E8M0 exponents, group-max, and scaling (rows x groups_per_row);
    // data is stored contiguously — TQuant flattens via TRESHAPE internally.
    using DstE8Tile = Tile<TileType::Vec, uint8_t, validRows, groupedCols_e8_static, BLayout::RowMajor, -1, -1,
                           SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, validRows, paddedCols, BLayout::RowMajor, validRows, paddedCols,
                            SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, bfloat16_t, validRows, groupedCols_b16_static, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, bfloat16_t, validRows, groupedCols_b16_static, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    ScalingTile scalingTile(validRows, groupedCols_valid);
    DstFP8Tile fp8Tile;
    DstE8Tile e8Tile(validRows, groupedCols_valid);
    MaxTile maxPerGpTile(validRows, groupedCols_valid);

    SrcGlobal srcGlobal(src);
    DstE8Global e8Global(out_e8m0);
    DstFP8Global fp8Global((__gm__ int8_t *)out_fp8);

    // UB layout for bf16: worst-case 128x128 needs ~60 KB total
    TASSIGN(srcTile, 0x0);         // bf16 src: up to 128*128*2 = 32 KB
    TASSIGN(maxPerGpTile, 0x8100); // bf16 group max (2D: up to 128*16*2 = 4 KB)
    TASSIGN(scalingTile, 0x9200);  // bf16 scaling   (2D: up to 128*16*2 = 4 KB)
    TASSIGN(e8Tile, 0xA300);       // uint8 e8m0     (2D: up to 128*32*1 = 4 KB)
    TASSIGN(fp8Tile, 0xB400);      // int8 fp8 output (up to 128*128*1 = 16 KB)

    TLOAD(srcTile, srcGlobal);

    if constexpr (mode == 0) {
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

        // E8M0: TQUANT stores data contiguously; use 1D alias for TSTORE
        using E8StoreND = Tile<TileType::Vec, uint8_t, 1, groupedCols_flat_aligned, BLayout::RowMajor, -1, -1,
                               SLayout::NoneBox, 512, PadValue::Zero>;
        E8StoreND e8StoreND(1, groupedCols_flattened);
        TASSIGN(e8StoreND, 0xA300);
        TSTORE(e8Global, e8StoreND);
        TSTORE(fp8Global, fp8Tile);
    } else {
        // NZ mode: TQUANT (ND output), then TMOV ND->ZZ for e8m0, TMOV ND->NZ for fp8.
        constexpr int groupedCols_valid = paddedCols / 32;

        // E8M0 ZZ destination: 2D with fractalMxSize=32 for [16,2] inner box
        using E8ZzTile = Tile<TileType::Vec, uint8_t, validRows, groupedCols_valid, BLayout::RowMajor, -1, -1,
                              SLayout::RowMajor, 32, PadValue::Zero>;
        // 1D flat alias at same address for TSTORE (no dispatch path for isRowMajor + SLayout::RowMajor)
        using E8StoreTile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1,
                                 SLayout::NoneBox, 512, PadValue::Zero>;
        // Scratch tile for TMOV ZZ gather-index generation
        constexpr int tmpBufSize = (16 + (validRows / 16) * (groupedCols_valid / 2) + 16) * sizeof(uint16_t);
        constexpr int tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);
        using TmpTile = Tile<TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                             512, PadValue::Zero>;

        // TMOV fp8 ND->NZ
        constexpr int virtualRow = PTO_CEIL(validRows, FRACTAL_NZ_ROW) + 1;
        using DstNZ_int8 = Tile<TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, validRows, paddedCols,
                                SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

        DstNZ_int8 fp8TileNZ;
        E8ZzTile e8ZzTile(validRows, groupedCols_valid);
        E8StoreTile e8StoreTile(1, groupedCols_flattened);
        TmpTile tmpTile(1, tmpBufSizeAligned);

        TASSIGN(fp8TileNZ, 0x0);      // reuse src tile address (consumed by TQUANT)
        TASSIGN(e8ZzTile, 0x8100);    // reuse maxPerGpTile address (consumed after TQUANT)
        TASSIGN(e8StoreTile, 0x8100); // alias for flat TSTORE at same address
        TASSIGN(tmpTile, 0x9200);     // reuse scalingTile address

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Step 1: Plain TQUANT (ND output)
        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);
        // Step 2: Convert E8M0 exponents ND -> ZZ layout
        TMOV(e8ZzTile, e8Tile, tmpTile);
        // Step 3: Convert FP8 data ND -> NZ layout
        TMOV(fp8TileNZ, fp8Tile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Store ZZ-reordered e8m0 exponents
        TSTORE(e8Global, e8StoreTile);

        using DstFp8GlobalNZ = GlobalTensor<int8_t, TileShape2D<int8_t, validRows, paddedCols, Layout::NZ>,
                                            BaseShape2D<int8_t, validRows, paddedCols, Layout::NZ>, Layout::NZ>;
        DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t *)out_fp8);
        TSTORE(fp8GlobalNZ, fp8TileNZ);
    }
}

template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8_BF16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    runTQuantBF16<validRows, validCols, mode, scaleAlg><<<1, nullptr, stream>>>(dst_exp, dst, (bfloat16_t *)src);
}

// FP16 --> MXFP8
template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
__global__ AICORE void runTQuantFP16(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                     __gm__ half __in__ *src)
{
    constexpr int paddedCols = PTO_CEIL(validCols, 32);
    constexpr int groupedCols_flattened = validRows * (paddedCols / 32);
    constexpr int groupedCols_valid = paddedCols / 32;
    // 1D-flat exp/max/scaling: TQUANT writes E8M0 packed-contiguous (1D fast path).
    constexpr int groupedCols_flat_u8 = PTO_CEIL(groupedCols_flattened, 32);  // uint8:  32 elems = 32 B
    constexpr int groupedCols_flat_b16 = PTO_CEIL(groupedCols_flattened, 16); // fp16:   16 elems = 32 B
    using SrcGlobal = GlobalTensor<half, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, groupedCols_flattened>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstFP8Global = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, half, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flat_u8, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                           512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, validRows, paddedCols, BLayout::RowMajor, validRows, paddedCols,
                            SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, half, 1, groupedCols_flat_b16, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, half, 1, groupedCols_flat_b16, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    ScalingTile scalingTile(1, groupedCols_flattened);
    DstFP8Tile fp8Tile;
    DstE8Tile e8Tile(1, groupedCols_flattened);
    MaxTile maxPerGpTile(1, groupedCols_flattened);

    SrcGlobal srcGlobal(src);
    DstE8Global e8Global(out_e8m0);
    DstFP8Global fp8Global((__gm__ int8_t *)out_fp8);

    // UB layout for fp16 (2 bytes per element, same footprint as bf16)
    TASSIGN(srcTile, 0x0);          // fp16 src: up to 128*128*2 = 32 KB
    TASSIGN(maxPerGpTile, 0x10100); // fp16 group max
    TASSIGN(scalingTile, 0x10600);  // fp16 scaling factors
    TASSIGN(e8Tile, 0x20700);       // uint8 e8m0 exponents
    TASSIGN(fp8Tile, 0x20C00);      // int8 fp8 output

    TLOAD(srcTile, srcGlobal);

    if constexpr (mode == 0) {
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

        TSTORE(e8Global, e8Tile);
        TSTORE(fp8Global, fp8Tile);
    } else {
        // NZ mode: TQUANT (ND output), then TMOV ND->ZZ for e8m0, TMOV ND->NZ for fp8.
        constexpr int groupedCols_valid = paddedCols / 32;

        using E8ZzTile = Tile<TileType::Vec, uint8_t, validRows, groupedCols_valid, BLayout::RowMajor, -1, -1,
                              SLayout::RowMajor, 32, PadValue::Zero>;
        using E8StoreTile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1,
                                 SLayout::NoneBox, 512, PadValue::Zero>;
        constexpr int tmpBufSize = (16 + (validRows / 16) * (groupedCols_valid / 2) + 16) * sizeof(uint16_t);
        constexpr int tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);
        using TmpTile = Tile<TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                             512, PadValue::Zero>;

        constexpr int virtualRow = PTO_CEIL(validRows, FRACTAL_NZ_ROW) + 1;
        using DstNZ_int8 = Tile<TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, validRows, paddedCols,
                                SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

        DstNZ_int8 fp8TileNZ;
        E8ZzTile e8ZzTile(validRows, groupedCols_valid);
        E8StoreTile e8StoreTile(1, groupedCols_flattened);
        TmpTile tmpTile(1, tmpBufSizeAligned);

        TASSIGN(fp8TileNZ, 0x0);
        TASSIGN(e8ZzTile, 0x10100);
        TASSIGN(e8StoreTile, 0x10100);
        TASSIGN(tmpTile, 0x10600);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile, ScalingTile, scaleAlg>(
            fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);
        TMOV(e8ZzTile, e8Tile, tmpTile);
        TMOV(fp8TileNZ, fp8Tile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(e8Global, e8StoreTile);

        using DstFp8GlobalNZ = GlobalTensor<int8_t, TileShape2D<int8_t, validRows, paddedCols, Layout::NZ>,
                                            BaseShape2D<int8_t, validRows, paddedCols, Layout::NZ>, Layout::NZ>;
        DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t *)out_fp8);
        TSTORE(fp8GlobalNZ, fp8TileNZ);
    }
}

template <int validRows, int validCols, int mode, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8_FP16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    runTQuantFP16<validRows, validCols, mode, scaleAlg><<<1, nullptr, stream>>>(dst_exp, dst, (half *)src);
}

template <typename SrcT, int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
struct MxFp8Exp2DSpec {
    static_assert(staticCols % 64 == 0, "static tail axis must be 64-aligned");
    static_assert(validCols % 64 == 0, "valid tail axis must be 64-aligned");
    static_assert(validRows <= staticRows && validCols <= staticCols, "valid shape must fit in static tile shape");
    static constexpr int rows = validRows;
    static constexpr int cols = validCols;
    static constexpr pto::QuantScaleAlg scaleAlgValue = scaleAlg;
    static constexpr int validGroups = validCols / 32;
    static constexpr int expCols = PTO_CEIL(staticCols / 32, 32);
    static constexpr int packedGroups = validRows * validGroups;
    static constexpr int packedGroupsAligned = PTO_CEIL(packedGroups, 16);
    static constexpr int ubAlign = TQUANT_A5_UB_ALIGN_BYTES;
    static constexpr int srcBytes = staticRows * staticCols * sizeof(SrcT);
    static constexpr int maxBytes = packedGroupsAligned * sizeof(SrcT);
    static constexpr int scalingBytes = maxBytes;
    static constexpr int e8Bytes = staticRows * expCols * sizeof(uint8_t);
    static constexpr int fp8Bytes = staticRows * staticCols * sizeof(uint8_t);
    static constexpr int maxOffset = PTO_CEIL(srcBytes, ubAlign);
    static constexpr int scalingOffset = PTO_CEIL(maxOffset + maxBytes, ubAlign);
    static constexpr int e8Offset = PTO_CEIL(scalingOffset + scalingBytes, ubAlign);
    static constexpr int fp8Offset = PTO_CEIL(e8Offset + e8Bytes, ubAlign);
    static_assert(fp8Offset + fp8Bytes < TQUANT_A5_UB_SIZE_BYTES, "TQUANT MXFP8 2D test UB layout exceeds UB size");

    using SrcGlobal = GlobalTensor<SrcT, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validGroups>, pto::Stride<1, 1, 1, validGroups, 1>>;
    using DstFP8Global = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, SrcT, staticRows, staticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, staticRows, expCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                           512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, staticRows, staticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, SrcT, 1, packedGroupsAligned, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, SrcT, 1, packedGroupsAligned, BLayout::RowMajor, -1, -1>;
};

template <typename Spec>
__global__ AICORE void runTQuantExp2D(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                      __gm__ typename Spec::SrcGlobal::RawDType __in__ *src)
{
    typename Spec::SrcTile srcTile(Spec::rows, Spec::cols);
    typename Spec::ScalingTile scalingTile(1, Spec::packedGroups);
    typename Spec::DstFP8Tile fp8Tile(Spec::rows, Spec::cols);
    typename Spec::DstE8Tile e8Tile(Spec::rows, Spec::validGroups);
    typename Spec::MaxTile maxPerGpTile(1, Spec::packedGroups);

    typename Spec::SrcGlobal srcGlobal(src);
    typename Spec::DstE8Global e8Global(out_e8m0);
    typename Spec::DstFP8Global fp8Global((__gm__ int8_t *)out_fp8);

    TASSIGN(srcTile, 0x0);
    TASSIGN(maxPerGpTile, Spec::maxOffset);
    TASSIGN(scalingTile, Spec::scalingOffset);
    TASSIGN(e8Tile, Spec::e8Offset);
    TASSIGN(fp8Tile, Spec::fp8Offset);
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::MXFP8, typename Spec::DstFP8Tile, typename Spec::SrcTile, typename Spec::DstE8Tile,
           typename Spec::MaxTile, typename Spec::ScalingTile, Spec::scaleAlgValue>(fp8Tile, srcTile, &e8Tile,
                                                                                    &maxPerGpTile, &scalingTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(e8Global, e8Tile);
    TSTORE(fp8Global, fp8Tile);
}

template <int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8_FP32_Exp2D(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream)
{
    using Spec = MxFp8Exp2DSpec<float, staticRows, staticCols, validRows, validCols, scaleAlg>;
    runTQuantExp2D<Spec><<<1, nullptr, stream>>>(dst_exp, dst, src);
}

template <int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8_BF16_Exp2D(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    using Spec = MxFp8Exp2DSpec<bfloat16_t, staticRows, staticCols, validRows, validCols, scaleAlg>;
    runTQuantExp2D<Spec><<<1, nullptr, stream>>>(dst_exp, dst, (bfloat16_t *)src);
}

template <int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP8_FP16_Exp2D(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    using Spec = MxFp8Exp2DSpec<half, staticRows, staticCols, validRows, validCols, scaleAlg>;
    runTQuantExp2D<Spec><<<1, nullptr, stream>>>(dst_exp, dst, (half *)src);
}

PTO_INTERNAL void CompactFp4PackedRows(__ubuf__ uint8_t *dstPtr, __ubuf__ uint8_t *srcPtr, uint32_t rows,
                                       uint32_t validPackedCols, uint32_t srcStride, uint32_t dstStride)
{
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(uint8_t);
    RegTensor<uint8_t> vreg;
    UnalignReg ureg;
    uint16_t repeatTimes = CeilDivision(validPackedCols, elementsPerRepeat);
    for (uint16_t row = 0; row < (uint16_t)rows; ++row) {
        uint32_t remaining = validPackedCols;
        __ubuf__ uint8_t *srcRow = srcPtr + row * srcStride;
        __ubuf__ uint8_t *dstRow = dstPtr + row * dstStride;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            uint32_t cols = remaining > elementsPerRepeat ? elementsPerRepeat : remaining;
            MaskReg preg = CreatePredicate<uint8_t>(cols);
            uint32_t offset = repeat * elementsPerRepeat;
            vldas(ureg, srcRow + offset);
            vldus(vreg, ureg, srcRow + offset);
            vsts(vreg, dstRow, offset, NORM_B8, preg);
            remaining -= cols;
        }
    }
}

template <typename SrcT, int validRows, int validCols>
struct MxFp4B16Spec {
    static constexpr int rows = validRows;
    static constexpr int paddedCols = PTO_CEIL(validCols, 32);
    static constexpr int packedCols = paddedCols / 2;
    static constexpr int validPackedCols = PTO_DIV_ROUNDUP(validCols, 2);
    static constexpr int groupedColsFlattened = validRows * (paddedCols / 32);
    static constexpr int groupedColsValid = paddedCols / 32;
    static constexpr int groupedColsE8Static = PTO_CEIL(groupedColsValid, 32);
    static constexpr int groupedColsB16Static = PTO_CEIL(groupedColsValid, 16);
    static constexpr int groupedColsFlatAligned = PTO_CEIL(groupedColsFlattened, 32);
    static constexpr int groupedColsB16FlatAligned = PTO_CEIL(groupedColsFlattened, 16);
    static constexpr int fp4FlatAligned = PTO_CEIL(validRows * packedCols, 32);
    static constexpr int validPackedColsAligned = PTO_CEIL(validPackedCols, 32);
    static constexpr int ubAlign = TQUANT_A5_UB_ALIGN_BYTES;
    static constexpr int srcBytes = validRows * paddedCols * sizeof(SrcT);
    static constexpr int maxBytes = groupedColsB16FlatAligned * sizeof(SrcT);
    static constexpr int scalingBytes = maxBytes;
    static constexpr int e8Bytes = validRows * groupedColsE8Static * sizeof(uint8_t);
    static constexpr int fp4Bytes = validRows * packedCols * sizeof(uint8_t);
    static constexpr int fp4StoreBytes = validRows * validPackedColsAligned * sizeof(uint8_t);
    static constexpr int maxOffset = PTO_CEIL(srcBytes, ubAlign);
    static constexpr int scalingOffset = PTO_CEIL(maxOffset + maxBytes, ubAlign);
    static constexpr int e8Offset = PTO_CEIL(scalingOffset + scalingBytes, ubAlign);
    static constexpr int fp4Offset = PTO_CEIL(e8Offset + e8Bytes, ubAlign);
    static constexpr int fp4StoreOffset = PTO_CEIL(fp4Offset + fp4Bytes, ubAlign);

    using SrcGlobal = GlobalTensor<SrcT, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, groupedColsValid>, pto::Stride<1, 1, 1, groupedColsValid, 1>>;
    using DstFP4Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validPackedCols>, pto::Stride<1, 1, 1, validPackedCols, 1>>;
    using SrcTile = Tile<TileType::Vec, SrcT, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, validRows, groupedColsE8Static, BLayout::RowMajor, -1, -1,
                           SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP4Tile = Tile<TileType::Vec, float4_e2m1x2_t, 1, fp4FlatAligned, BLayout::RowMajor, -1, -1,
                            SLayout::NoneBox, 512, PadValue::Zero>;
    using DstBytesTile = Tile<TileType::Vec, uint8_t, validRows, validPackedColsAligned, BLayout::RowMajor, -1, -1,
                              SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, SrcT, 1, groupedColsB16FlatAligned, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, SrcT, 1, groupedColsB16FlatAligned, BLayout::RowMajor, -1, -1>;
};

template <typename Spec>
PTO_INTERNAL void AssignMxFp4B16Tiles(typename Spec::SrcTile &srcTile, typename Spec::MaxTile &maxPerGpTile,
                                      typename Spec::ScalingTile &scalingTile, typename Spec::DstE8Tile &e8Tile,
                                      typename Spec::DstFP4Tile &fp4Tile, typename Spec::DstBytesTile &fp4BytesTile)
{
    TASSIGN(srcTile, 0x0);
    TASSIGN(maxPerGpTile, Spec::maxOffset);
    TASSIGN(scalingTile, Spec::scalingOffset);
    TASSIGN(e8Tile, Spec::e8Offset);
    TASSIGN(fp4Tile, Spec::fp4Offset);
    TASSIGN(fp4BytesTile, Spec::fp4StoreOffset);
}

template <typename Spec>
PTO_INTERNAL void StoreMxFp4B16Result(typename Spec::DstFP4Tile &fp4Tile, typename Spec::DstBytesTile &fp4BytesTile,
                                      typename Spec::DstE8Tile &e8Tile, typename Spec::DstE8Global &e8Global,
                                      typename Spec::DstFP4Global &fp4Global)
{
    __VEC_SCOPE__
    {
        mem_bar(VST_VLD);
        CompactFp4PackedRows((__ubuf__ uint8_t *)fp4BytesTile.data(), (__ubuf__ uint8_t *)fp4Tile.data(), Spec::rows,
                             Spec::validPackedCols, Spec::packedCols, Spec::validPackedColsAligned);
        mem_bar(VST_VST);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(e8Global, e8Tile);
    TSTORE(fp4Global, fp4BytesTile);
}

template <typename SrcT, int validRows, int validCols, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
__global__ AICORE void runTQuantMXFP4E2M1B16(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp4,
                                             __gm__ SrcT __in__ *src)
{
    using Spec = MxFp4B16Spec<SrcT, validRows, validCols>;
    static_assert(Spec::fp4StoreOffset + Spec::fp4StoreBytes <= TQUANT_A5_UB_SIZE_BYTES,
                  "MXFP4 E2M1 test kernel UB layout exceeds UB capacity.");
    typename Spec::SrcTile srcTile(validRows, validCols);
    typename Spec::DstFP4Tile fp4Tile(1, Spec::fp4Bytes);
    typename Spec::DstBytesTile fp4BytesTile(validRows, Spec::validPackedCols);
    typename Spec::DstE8Tile e8Tile(validRows, Spec::groupedColsValid);
    typename Spec::MaxTile maxPerGpTile(1, Spec::groupedColsFlattened);
    typename Spec::ScalingTile scalingTile(1, Spec::groupedColsFlattened);

    typename Spec::SrcGlobal srcGlobal(src);
    typename Spec::DstE8Global e8Global(out_e8m0);
    typename Spec::DstFP4Global fp4Global(out_fp4);

    AssignMxFp4B16Tiles<Spec>(srcTile, maxPerGpTile, scalingTile, e8Tile, fp4Tile, fp4BytesTile);
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    if constexpr (scaleAlg == pto::QuantScaleAlg::OCP) {
        TQUANT<pto::QuantType::MXFP4_E2M1, typename Spec::DstFP4Tile, typename Spec::SrcTile, typename Spec::DstE8Tile,
               typename Spec::MaxTile>(fp4Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);
    } else {
        TQUANT<pto::QuantType::MXFP4_E2M1, typename Spec::DstFP4Tile, typename Spec::SrcTile, typename Spec::DstE8Tile,
               typename Spec::MaxTile, typename Spec::ScalingTile, scaleAlg>(fp4Tile, srcTile, &e8Tile, &maxPerGpTile,
                                                                             &scalingTile);
    }
    StoreMxFp4B16Result<Spec>(fp4Tile, fp4BytesTile, e8Tile, e8Global, fp4Global);
}

template <int validRows, int validCols, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP4_E2M1_FP16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    runTQuantMXFP4E2M1B16<half, validRows, validCols, scaleAlg><<<1, nullptr, stream>>>(dst_exp, dst, (half *)src);
}

template <int validRows, int validCols, pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP4_E2M1_BF16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    runTQuantMXFP4E2M1B16<bfloat16_t, validRows, validCols, scaleAlg>
        <<<1, nullptr, stream>>>(dst_exp, dst, (bfloat16_t *)src);
}

template <typename SrcT, int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
struct MxFp4B16Exp2DSpec {
    static_assert(staticCols % 32 == 0, "static tail axis must be 32-aligned");
    static_assert(validCols % 32 == 0, "valid tail axis must be 32-aligned");
    static_assert(validRows <= staticRows && validCols <= staticCols, "valid shape must fit in static tile shape");
    static constexpr int rows = validRows;
    static constexpr int cols = validCols;
    static constexpr pto::QuantScaleAlg scaleAlgValue = scaleAlg;
    static constexpr int validGroups = validCols / 32;
    static constexpr int expCols = PTO_CEIL(staticCols / 32, 32);
    static constexpr int packedGroups = validRows * validGroups;
    static constexpr int packedGroupsAligned = PTO_CEIL(packedGroups, 16);
    static constexpr int staticPackedCols = staticCols / 2;
    static constexpr int packedCols = staticPackedCols;
    static constexpr int validPackedCols = PTO_DIV_ROUNDUP(validCols, 2);
    static constexpr int validPackedColsAligned = PTO_CEIL(validPackedCols, 32);
    static constexpr int ubAlign = TQUANT_A5_UB_ALIGN_BYTES;
    static constexpr int srcBytes = staticRows * staticCols * sizeof(SrcT);
    static constexpr int maxBytes = packedGroupsAligned * sizeof(SrcT);
    static constexpr int scalingBytes = maxBytes;
    static constexpr int e8Bytes = staticRows * expCols * sizeof(uint8_t);
    static constexpr int fp4Bytes = staticRows * staticPackedCols * sizeof(uint8_t);
    static constexpr int fp4StoreBytes = validRows * validPackedColsAligned * sizeof(uint8_t);
    static constexpr int maxOffset = PTO_CEIL(srcBytes, ubAlign);
    static constexpr int scalingOffset = PTO_CEIL(maxOffset + maxBytes, ubAlign);
    static constexpr int e8Offset = PTO_CEIL(scalingOffset + scalingBytes, ubAlign);
    static constexpr int fp4Offset = PTO_CEIL(e8Offset + e8Bytes, ubAlign);
    static constexpr int fp4StoreOffset = PTO_CEIL(fp4Offset + fp4Bytes, ubAlign);

    using SrcGlobal = GlobalTensor<SrcT, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validGroups>, pto::Stride<1, 1, 1, validGroups, 1>>;
    using DstFP4Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validPackedCols>, pto::Stride<1, 1, 1, validPackedCols, 1>>;
    using SrcTile = Tile<TileType::Vec, SrcT, staticRows, staticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, staticRows, expCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                           512, PadValue::Zero>;
    using DstFP4Tile = Tile<TileType::Vec, float4_e2m1x2_t, 1, fp4Bytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using DstBytesTile = Tile<TileType::Vec, uint8_t, validRows, validPackedColsAligned, BLayout::RowMajor, -1, -1,
                              SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, SrcT, 1, packedGroupsAligned, BLayout::RowMajor, -1, -1>;
    using ScalingTile = Tile<TileType::Vec, SrcT, 1, packedGroupsAligned, BLayout::RowMajor, -1, -1>;
};

template <typename Spec>
__global__ AICORE void runTQuantMXFP4E2M1B16Exp2D(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp4,
                                                  __gm__ typename Spec::SrcGlobal::RawDType __in__ *src)
{
    static_assert(Spec::fp4StoreOffset + Spec::fp4StoreBytes <= TQUANT_A5_UB_SIZE_BYTES,
                  "MXFP4 E2M1 2D test kernel UB layout exceeds UB capacity.");
    typename Spec::SrcTile srcTile(Spec::rows, Spec::cols);
    typename Spec::DstFP4Tile fp4Tile(1, Spec::fp4Bytes);
    typename Spec::DstBytesTile fp4BytesTile(Spec::rows, Spec::validPackedCols);
    typename Spec::DstE8Tile e8Tile(Spec::rows, Spec::validGroups);
    typename Spec::MaxTile maxPerGpTile(1, Spec::packedGroups);
    typename Spec::ScalingTile scalingTile(1, Spec::packedGroups);

    typename Spec::SrcGlobal srcGlobal(src);
    typename Spec::DstE8Global e8Global(out_e8m0);
    typename Spec::DstFP4Global fp4Global(out_fp4);

    TASSIGN(srcTile, 0x0);
    TASSIGN(maxPerGpTile, Spec::maxOffset);
    TASSIGN(scalingTile, Spec::scalingOffset);
    TASSIGN(e8Tile, Spec::e8Offset);
    TASSIGN(fp4Tile, Spec::fp4Offset);
    TASSIGN(fp4BytesTile, Spec::fp4StoreOffset);
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::MXFP4_E2M1, typename Spec::DstFP4Tile, typename Spec::SrcTile, typename Spec::DstE8Tile,
           typename Spec::MaxTile, typename Spec::ScalingTile, Spec::scaleAlgValue>(fp4Tile, srcTile, &e8Tile,
                                                                                    &maxPerGpTile, &scalingTile);

    StoreMxFp4B16Result<Spec>(fp4Tile, fp4BytesTile, e8Tile, e8Global, fp4Global);
}

template <int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP4_E2M1_BF16_Exp2D(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    using Spec = MxFp4B16Exp2DSpec<bfloat16_t, staticRows, staticCols, validRows, validCols, scaleAlg>;
    runTQuantMXFP4E2M1B16Exp2D<Spec><<<1, nullptr, stream>>>(dst_exp, dst, (bfloat16_t *)src);
}

template <int staticRows, int staticCols, int validRows, int validCols,
          pto::QuantScaleAlg scaleAlg = pto::QuantScaleAlg::OCP>
void LaunchTQuantMXFP4_E2M1_FP16_Exp2D(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream)
{
    using Spec = MxFp4B16Exp2DSpec<half, staticRows, staticCols, validRows, validCols, scaleAlg>;
    runTQuantMXFP4E2M1B16Exp2D<Spec><<<1, nullptr, stream>>>(dst_exp, dst, (half *)src);
}

} // namespace TQuantTest

// MXFP8 cases
template void TQuantTest::LaunchTQuantMXFP8<32, 32, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<32, 64, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<64, 128, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<128, 128, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<32, 64, 1>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<64, 128, 1>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<64, 256, 1>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<64, 512, 1>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<128, 128, 1>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<15, 32, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<7, 64, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<33, 64, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
// INT8 SYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                 void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
// INT8 ASYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                  float *scale, void *stream,
                                                                                  float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);
template void TQuantTest::LaunchTQuantInt8<32, 72, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src, float *scale,
                                                                                 void *stream, float *offset);
// MXFP8 BF16 cases
template void TQuantTest::LaunchTQuantMXFP8_BF16<32, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<64, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<128, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<14, 16, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                            void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<7, 48, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<32, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<64, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<128, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<32, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                     uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<64, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                     uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<128, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                      uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<7, 48, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                   uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16<2, 256, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                    uint8_t *dst_exp, void *stream);
// MXFP8 FP16 cases
template void TQuantTest::LaunchTQuantMXFP8_FP16<32, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<64, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<128, 128, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<4, 256, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                            void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<11, 640, 0>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<32, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                     uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<64, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                     uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<128, 128, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                      uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<2, 256, 0, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                    uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8_BF16_Exp2D<3, 128, 1, 64>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                                      void *stream);

template void TQuantTest::LaunchTQuantMXFP4_E2M1_FP16<2, 128>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_FP16<32, 1024>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                                void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_FP16<2, 256, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                      uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_BF16<2, 128>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_BF16<32, 1024>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                                void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_BF16<2, 256, pto::QuantScaleAlg::NV>(uint8_t *dst, uint16_t *src,
                                                                                      uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP4_E2M1_BF16_Exp2D<4, 128, 2, 128, pto::QuantScaleAlg::NV>(uint8_t *dst,
                                                                                                    uint16_t *src,
                                                                                                    uint8_t *dst_exp,
                                                                                                    void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<32, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<64, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                             void *stream);
template void TQuantTest::LaunchTQuantMXFP8_FP16<128, 128, 1>(uint8_t *dst, uint16_t *src, uint8_t *dst_exp,
                                                              void *stream);
