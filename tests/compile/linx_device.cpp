/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;
using T = Tile<TileType::Vec, float, 4, 8>;
using M = Tile<TileType::Vec, uint32_t, 4, 8>;
using G = GlobalTensor<float, Shape<1, 1, 1, 4, 8>, Stride<32, 32, 32, 8, 1>>;

extern "C" void linxKernel(float* output, float* input0, float* input1)
{
    G out(output), in0(input0), in1(input1);
    T a, b, dst, tmp;
    M mask;
    TLOAD(a, in0);
    TLOAD(b, in1);
    TADD(dst, a, b);
    TMUL(dst, dst, b);
    TSUB(dst, dst, a);
    TCMP(mask, a, b, CmpMode::GT);
    TSEL(dst, mask, dst, b, tmp);
    TSTORE(out, dst);
}

extern "C" void linxMath(float* output, float* input0, float* input1)
{
    G out(output), in0(input0), in1(input1);
    T a, b, dst;
    TLOAD(a, in0);
    TLOAD(b, in1);
    TABS(dst, a);
    TDIV(dst, dst, b);
    TMAX(dst, dst, a);
    TMIN(dst, dst, b);
    TSQRT(dst, dst);
    TEXP(dst, dst);
    TSTORE(out, dst);
}

extern "C" void linxLogical(int32_t* output, int32_t* input0, int32_t* input1)
{
    using IG = GlobalTensor<int32_t, Shape<1, 1, 1, 4, 8>, Stride<32, 32, 32, 8, 1>>;
    IG out(output), in0(input0), in1(input1);
    Tile<TileType::Vec, int32_t, 4, 8> a, b, dst;
    TLOAD(a, in0);
    TLOAD(b, in1);
    TAND(dst, a, b);
    TOR(dst, dst, a);
    TSTORE(out, dst);
}

extern "C" void linxMatrix(float* output, float* input0, float* input1)
{
    using GA = GlobalTensor<float, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;
    using GB = GlobalTensor<float, Shape<1, 1, 1, 16, 8>, Stride<128, 128, 128, 8, 1>>;
    GA in0(input0);
    GB in1(input1), out(output);
    CubeTileM16<float, 16, 16> a;
    CubeTileN8<float, 16, 8> b;
    CubeAccumulatorM16<float, 16, 8> c;
    TLOAD(a, in0);
    TLOAD(b, in1);
    TMATMUL(c, a, b);
    TSTORE(out, c);
}

#ifdef PTO_LINX_TEST_REJECT_TRANSPOSE
extern "C" void linxTranspose(float* output, float* input)
{
    using GT = GlobalTensor<float, Shape<1, 1, 1, 8, 4>, Stride<32, 32, 32, 4, 1>>;
    G in(input);
    GT out(output);
    T src, tmp;
    Tile<TileType::Vec, float, 8, 4> dst;
    TLOAD(src, in);
    TTRANS(dst, src, tmp);
    TSTORE(out, dst);
}

#endif

// Unit columns with a runtime row pitch remain supported by native TLSU.
extern "C" void linxDynamicRowStride(float* output, float* input, int inputPitch, int outputPitch)
{
    using S = Stride<1, 1, 1, DYNAMIC, 1>;
    using StridedG = GlobalTensor<float, Shape<1, 1, 1, 4, 8>, S>;
    StridedG in(input, {}, S(inputPitch)), out(output, {}, S(outputPitch));
    T tile;
    TLOAD(tile, in);
    TSTORE(out, tile);
}
