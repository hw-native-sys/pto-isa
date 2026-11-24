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
using namespace pto;
template <typename T, int R, int C>
using Mem = global_tensor<T, RowMajor<R, C>>;
constexpr TIMG2COLParams imageParams(unsigned h, unsigned w, unsigned channels)
{
    return {
        uint64_t(h) | (uint64_t(w) << 16) | (uint64_t(channels) << 32) | (uint64_t(1) << 48) | (uint64_t(1) << 56),
        (uint64_t(1) << 32) | (uint64_t(1) << 37) | (uint64_t(1) << 42) | (uint64_t(1) << 48), 0};
}
void imageND(float* input, float* output)
{
    CubeTileM16<float, 16, 8> d;
    Mem<float, 16, 3> src(input);
    Mem<float, 16, 8> out(output);
    TIMG2COL(d, src, imageParams(4, 4, 3));
    if (get_thread_idx() == 0)
        TSTORE(out, d);
}
void imageDN(float* input, float* output)
{
    using S = Tile<Location::Left, float, 16, 8>;
    SharedTile<S> shared;
    S local;
    Mem<float, 3, 16> src(input);
    Mem<float, 16, 8> out(output);
    TIMG2COL_SPART<DN2ND>(shared, src, imageParams(4, 4, 3), 8);
    TMOV_S2L_EXTRACT(local, shared);
    TSTORE(out, local);
}
void weights(float* input, float* output)
{
    using S = SharedMatrixRight<float, 8, 8>;
    SharedTile<S> shared;
    S local;
    Mem<float, 8, 3> src(input);
    Mem<float, 8, 8> out(output);
    TLOAD<OHWI2NK, 8>(shared, src, make_weight_tload_params(3, 8, 1, 1));
    TMOV_S2L_EXTRACT(local, shared);
    TSTORE(out, local);
}
void slots(float* image, float* weight, float* output)
{
    using A = SharedMatrixLeft<float, 64, 8>;
    using B = SharedMatrixRight<float, 8, 8>;
    using SA = SharedTileSlot<0, A>;
    using SB = SharedTileSlot<1, B>;
    using D = CubeAccumulatorM16<float, 16, 8>;
    Mem<float, 64, 3> src(image);
    Mem<float, 8, 3> w(weight);
    D dst;
    if (get_thread_idx() == 0) {
        TIMG2COL_SPART_SLOT<NORM, 0, 8, A>(src, imageParams(8, 8, 3));
        TLOAD_SLOT<OIHW2NK, 1, 8, B>(w, make_weight_tload_params(3, 8, 1, 1));
    }
    TMATMUL_SLOT<D, 0, SA, 1, SB>(dst);
    Mem<float, 16, 8> out(output + get_thread_idx() * 128);
    TSTORE(out, dst);
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <array>
#include <iostream>
#include <stdexcept>
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
int main()
{
    std::array<float, 192> image{}, nchw{};
    std::array<float, 24> weight{};
    std::array<float, 512> out{};
    for (unsigned i = 0; i < image.size(); ++i)
        image[i] = float(i % 13) - 6;
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned c = 0; c < 3; ++c)
            nchw[c * 16 + i] = image[i * 3 + c];
    for (unsigned i = 0; i < weight.size(); ++i)
        weight[i] = float(i % 7) - 3;
    linx_cpu::runCore([&] { imageND(image.data(), out.data()); });
    for (unsigned r = 0; r < 16; ++r)
        for (unsigned c = 0; c < 8; ++c)
            check(out[r * 8 + c] == (c < 3 ? image[r * 3 + c] : 0), "TIMG2COL NHWC and C0 padding");
    imageDN(nchw.data(), out.data());
    for (unsigned r = 0; r < 16; ++r)
        for (unsigned c = 0; c < 8; ++c)
            check(out[r * 8 + c] == (c < 3 ? image[r * 3 + c] : 0), "TIMG2COL NCHW Shared SPART");
    weights(weight.data(), out.data());
    for (unsigned r = 0; r < 8; ++r)
        for (unsigned c = 0; c < 8; ++c)
            check(out[r * 8 + c] == (c < 3 ? weight[r * 3 + c] : 0), "weight TLOAD NK packing");
    linx_cpu::runCore([&] { slots(image.data(), weight.data(), out.data()); });
    for (unsigned r = 0; r < 64; ++r)
        for (unsigned n = 0; n < 8; ++n) {
            float sum = 0;
            for (unsigned k = 0; k < 3; ++k)
                sum += image[r * 3 + k] * weight[n * 3 + k];
            check(out[r * 8 + n] == sum, "Shared slot image/weight to matrix pipeline");
        }
    std::cout << "Linx image, weight, and Shared slot execution checks passed\n";
}
#endif
