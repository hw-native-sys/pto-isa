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
#ifndef LINX_NEGATIVE
#define LINX_NEGATIVE 0
#endif
void invalidKernel(float* pointer)
{
#if LINX_NEGATIVE == 1
    Tile<Location::Vec, float, 4, 32> a, b;
    TTRANS(a, b);
#elif LINX_NEGATIVE == 2
    Tile<Location::Vec, float, 1, 131072> a;
    global_tensor<float, RowMajor<1, 131072>> m(pointer);
    TLOAD(a, m);
#elif LINX_NEGATIVE == 3
    ElementTile<__half, 32> bad;
#elif LINX_NEGATIVE == 4
    VecTileM32<uint32_t, 32, 4> a;
    global_tensor<float, RowMajor<32, 4>> m(pointer);
    TLOAD(a, m);
#elif LINX_NEGATIVE == 5
    CubeTileM32<float, 32, 8> a;
    CubeTileN8<float, 8, 8> b;
    CubeAccumulatorM32<float, 32, 8> d;
    constexpr auto attr = [] {
        FixpAttr a;
        a.TransA = true;
        return a;
    }();
    TMATMUL<attr>(d, a, b);
#elif LINX_NEGATIVE == 6
    VecTileM32<float, 32, 4> a;
    auto p = predicate::compare_scalar<CmpMode::GE>(a, 0.f);
    (void)p.Carrier;
#elif LINX_NEGATIVE == 7
    using P = VecTileM32<float, 32, 8>;
    using W = VecTileM32<float, 32, 4>;
    LocalCubeAssembly<P, W, W> a;
    auto first = a.writer<0>();
    global_tensor<float, RowMajor<32, 4>> m(pointer);
    TLOAD_ASS(first, m);
#elif LINX_NEGATIVE == 8
    VecTileM32<uint32_t, 32, 4> a;
    global_tensor<uint32_t, RowMajor<32, 4>> m(reinterpret_cast<uint32_t*>(pointer));
    TSTORE_ATOMIC_ADD<3>(m, a);
#endif
}
#ifdef __CPU_SIM
#include <cstdlib>
int main(int argc, char** argv)
{
    if (argc == 1)
        return 0;
    using P = VecTileM32<float, 32, 8>;
    using W = VecTileM32<float, 32, 4>;
    LocalCubeAssembly<P, W, W> assembly;
    if (std::atoi(argv[1]) == 4) {
        linx_cpu::runCore([] {
            using A = SharedTileSlot<0, SharedMatrixLeft<float, 64, 8>>;
            using B = SharedTileSlot<1, SharedMatrixRight<float, 8, 8>>;
            CubeAccumulatorM16<float, 16, 8> dst;
            TMATMUL_SLOT<decltype(dst), 0, A, 1, B>(dst);
        });
    } else if (std::atoi(argv[1]) == 3) {
        VecTileM32<uint32_t, 32, 4> cube;
        TEXPANDS(cube, 0u);
        TUNPACK(cube, cube, 0x0500);
    } else if (std::atoi(argv[1]) == 1) {
        auto incomplete = assembly.finish();
        (void)incomplete;
    } else {
        float input[128]{};
        global_tensor<float, RowMajor<32, 4>> m(input);
        auto second = assembly.writer<1>();
        TLOAD_ASS(second, m);
    }
    return 0;
}
#endif
