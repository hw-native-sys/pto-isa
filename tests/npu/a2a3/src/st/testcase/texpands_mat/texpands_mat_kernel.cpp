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

template <typename GlobalData, typename TileData, int reshapeRow, int reshapeCol>
AICORE inline void TSTORE_MAT2GM_CONVTILE(GlobalData &dst, TileData &src)
{
    __cbuf__ typename TileData::DType *srcAddr = (__cbuf__ typename TileData::DType *)src.data();
    typename GlobalData::DType *dstAddr = dst.data();

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);

    uint32_t validRow = reshapeRow;
    uint32_t validCol = reshapeCol;

    uint16_t nBurst = (validCol + blockSizeElem - 1) / blockSizeElem;
    uint16_t lenBurst = validRow;
    uint16_t l1Gap = 0;
    uint16_t gmGap = 0;
    copy_cbuf_to_gm(dstAddr, srcAddr, (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
}

template <typename T, int Rows, int Cols>
AICORE inline void runTExpandS_Tile(__gm__ T *out, T value)
{
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                    pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>>;
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    Event<Op::TEXPANDS_MAT, Op::TSTORE_MAT> evtExpands_Store;
    evtExpands_Store = TEXPANDS<TileData>(MatTile, value);
    TSTORE(dstGlobal, MatTile, evtExpands_Store);
    out = dstGlobal.data();
}

template <typename T, int N, int C1, int H, int W, int C0>
AICORE inline void runTSetValue_ConvTile(__gm__ T *out, T value)
{
    constexpr int elementSize = N * C1 * H * W * C0;
    constexpr int bufferSizeA = elementSize * sizeof(T);
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, 1, elementSize>,
                                    pto::Stride<elementSize, elementSize, elementSize, elementSize, 1>>;
    GlobalData dstGlobal(out);

    using TileData = ConvTile<TileType::Mat, T, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<N, C1, H, W, C0>>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    TEXPANDS<TileData>(MatTile, value);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    constexpr int reshapeRow = N;
    constexpr int reshapeCol = C1 * H * W * C0;
    TSTORE_MAT2GM_CONVTILE<GlobalData, TileData, reshapeRow, reshapeCol>(dstGlobal, MatTile);
    out = dstGlobal.data();
}

template <typename T, int format, int shape0, int shape1, int shape2, int shape3, int shape4>
__global__ AICORE void TEXPANDS_KERNEL(__gm__ uint8_t *out, T value)
{
    if constexpr (format == 0) { // Tile
        if constexpr (std::is_same_v<T, uint16_t>) {
            runTExpandS_Tile<bfloat16_t, shape0, shape1>(reinterpret_cast<__gm__ bfloat16_t *>(out), 7);
        } else {
            runTExpandS_Tile<T, shape0, shape1>(reinterpret_cast<__gm__ T *>(out), value);
        }
    } else if constexpr (format == 1) { // convTile
        runTSetValue_ConvTile<T, shape0, shape1, shape2, shape3, shape4>(reinterpret_cast<__gm__ T *>(out), value);
    }
}

template <int32_t testKey>
void launchTEXPANDS_Mat(uint8_t *out, void *stream)
{
    if constexpr (testKey == 1) {
        TEXPANDS_KERNEL<half, 0, 128, 128, 0, 0, 0><<<1, nullptr, stream>>>(out, half(2));
    } else if constexpr (testKey == 2) {
        TEXPANDS_KERNEL<int16_t, 0, 32, 64, 0, 0, 0><<<1, nullptr, stream>>>(out, int16_t(5));
    } else if constexpr (testKey == 3) {
        TEXPANDS_KERNEL<float, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, float(3));
    } else if constexpr (testKey == 4) {
        TEXPANDS_KERNEL<int8_t, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, int8_t(1));
    } else if constexpr (testKey == 5) {
        TEXPANDS_KERNEL<uint16_t, 0, 256, 256, 0, 0, 0><<<1, nullptr, stream>>>(out, uint16_t(7));
    } else if constexpr (testKey == 6) {
        TEXPANDS_KERNEL<half, 1, 2, 32, 14, 14, 8><<<1, nullptr, stream>>>(out, half(3));
    } else if constexpr (testKey == 7) {
        TEXPANDS_KERNEL<int16_t, 1, 2, 5, 2, 3, 8><<<1, nullptr, stream>>>(out, int16_t(8));
    } else if constexpr (testKey == 8) {
        TEXPANDS_KERNEL<int32_t, 1, 2, 5, 5, 1, 8><<<1, nullptr, stream>>>(out, int32_t(5));
    } else if constexpr (testKey == 9) {
        TEXPANDS_KERNEL<int32_t, 1, 3, 4, 5, 1, 8><<<1, nullptr, stream>>>(out, uint32_t(11));
    }
}

template void launchTEXPANDS_Mat<1>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<2>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<3>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<4>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<5>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<6>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<7>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<8>(uint8_t *out, void *stream);
template void launchTEXPANDS_Mat<9>(uint8_t *out, void *stream);