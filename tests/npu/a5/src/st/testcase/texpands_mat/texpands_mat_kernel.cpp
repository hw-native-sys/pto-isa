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
#include <limits>
#include <algorithm>

using namespace std;
using namespace pto;

template <typename T, typename TileData, typename TileUBData, typename GlobalData>
AICORE inline void runTexpandsAndTstore(__gm__ T *&out, TileData &MatTile, TileUBData &srcTile, GlobalData &dstGlobal,
                                        T value, int elementSize)
{
    TASSIGN(MatTile, 0x0);
    TASSIGN(srcTile, 0x0);

    __cbuf__ T *srcMatAddr = MatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    uint8_t syncID = 0;

#if defined(__DAV_CUBE__)
    TEXPANDS<TileData>(MatTile, value); // MTE2
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // L1 -> UB : AIC
    uint16_t blockCount = 1;
    uint16_t blockLen = elementSize * sizeof(T) / 32;
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);

    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int Rows, int Cols>
AICORE inline void runTexpands_Tile(__gm__ T *out, T value)
{
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                    pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>>;
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::NoneBox>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    using TileUBData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    TileUBData srcTile;
    constexpr int elementSize = Rows * Cols;
    runTexpandsAndTstore(out, MatTile, srcTile, dstGlobal, value, elementSize);
}

template <typename T, int N, int C1, int H, int W, int C0>
AICORE inline void runTexpands_ConvTile(__gm__ T *out, T value)
{
    constexpr int elementSize = N * C1 * H * W * C0;
    constexpr int bufferSizeA = elementSize * sizeof(T);
    constexpr int reshapeRow = N;
    constexpr int reshapeCol = C1 * H * W * C0;
    using GlobalData = GlobalTensor<
        T, pto::Shape<1, 1, 1, reshapeRow, reshapeCol>,
        pto::Stride<reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeCol, 1>>;
    GlobalData dstGlobal(out);

    using TileData = ConvTile<TileType::Mat, T, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<N, C1, H, W, C0>>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    using TileUBData = Tile<TileType::Vec, T, reshapeRow, reshapeCol, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(reshapeRow, reshapeCol);

    runTexpandsAndTstore(out, MatTile, srcTile, dstGlobal, value, elementSize);
}

template <typename T, int N, int D, int C1, int H, int W, int C0>
AICORE inline void runTexpands_ConvTile_3d(__gm__ T *out, T value)
{
    constexpr int elementSize = N * D * C1 * H * W * C0;
    constexpr int bufferSizeA = elementSize * sizeof(T);
    constexpr int reshapeRow = N * D;
    constexpr int reshapeCol = C1 * H * W * C0;
    using GlobalData = GlobalTensor<
        T, pto::Shape<1, 1, 1, reshapeRow, reshapeCol>,
        pto::Stride<reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeCol, 1>>;
    GlobalData dstGlobal(out);

    using TileData = ConvTile<TileType::Mat, T, bufferSizeA, Layout::NDC1HWC0, pto::ConvTileShape<N, D, C1, H, W, C0>>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    using TileUBData = Tile<TileType::Vec, T, reshapeRow, reshapeCol, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(reshapeRow, reshapeCol);

    runTexpandsAndTstore(out, MatTile, srcTile, dstGlobal, value, elementSize);
}

template <typename T, int format, int shape0, int shape1, int shape2, int shape3, int shape4, int shape5 = 0>
__global__ AICORE void TEXPANDS_KERNEL(__gm__ uint8_t *out, T value)
{
    if constexpr (format == 0) { // Tile
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            runTexpands_Tile<bfloat16_t, shape0, shape1>(reinterpret_cast<__gm__ bfloat16_t *>(out), bfloat16_t(0));
        } else {
            runTexpands_Tile<T, shape0, shape1>(reinterpret_cast<__gm__ T *>(out), value);
        }
    } else if constexpr (format == 1) { // convTile
        runTexpands_ConvTile<T, shape0, shape1, shape2, shape3, shape4>(reinterpret_cast<__gm__ T *>(out), value);
    } else if constexpr (format == 2) { // convTile 3D
        runTexpands_ConvTile_3d<T, shape0, shape1, shape2, shape3, shape4, shape5>(reinterpret_cast<__gm__ T *>(out),
                                                                                   value);
    }
}

template <int32_t testKey>
void launchTEXPANDS_MAT(uint8_t *out, void *stream)
{
    if constexpr (testKey == 1) {
        TEXPANDS_KERNEL<half, 0, 128, 128, 0, 0, 0, 0><<<1, nullptr, stream>>>(out, half(2));
    } else if constexpr (testKey == 2) {
        TEXPANDS_KERNEL<int16_t, 0, 32, 64, 0, 0, 0, 0><<<1, nullptr, stream>>>(out, int16_t(5));
    } else if constexpr (testKey == 3) {
        TEXPANDS_KERNEL<float, 0, 32, 32, 0, 0, 0, 0><<<1, nullptr, stream>>>(out, float(3));
    } else if constexpr (testKey == 4) {
        TEXPANDS_KERNEL<int8_t, 0, 32, 32, 0, 0, 0, 0><<<1, nullptr, stream>>>(out, int8_t(1));
    } else if constexpr (testKey == 5) {
        // uint16_t represent bfloat16
        TEXPANDS_KERNEL<uint16_t, 0, 256, 256, 0, 0, 0, 0><<<1, nullptr, stream>>>(out, 0);
    } else if constexpr (testKey == 6) {
        TEXPANDS_KERNEL<half, 1, 1, 16, 7, 7, 16, 0><<<1, nullptr, stream>>>(out, half(3));
    } else if constexpr (testKey == 7) {
        TEXPANDS_KERNEL<int16_t, 1, 2, 5, 2, 3, 8, 0><<<1, nullptr, stream>>>(out, int16_t(8));
    } else if constexpr (testKey == 8) {
        TEXPANDS_KERNEL<int32_t, 2, 2, 2, 3, 2, 1, 8><<<1, nullptr, stream>>>(out, int32_t(5));
    } else if constexpr (testKey == 9) {
        TEXPANDS_KERNEL<uint32_t, 2, 2, 3, 4, 1, 2, 8><<<1, nullptr, stream>>>(out, uint32_t(11));
    }
}

template void launchTEXPANDS_MAT<1>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<2>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<3>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<4>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<5>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<6>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<7>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<8>(uint8_t *out, void *stream);
template void launchTEXPANDS_MAT<9>(uint8_t *out, void *stream);