/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSORT32_SOFT_HPP
#define TSORT32_SOFT_HPP

#include <pto/common/constants.hpp>

namespace pto {

constexpr const uint32_t VBS32_BLOCK_SIZE = 32;
constexpr const uint32_t VBS32_HALF_DST_STRIDE = 128;

static AICORE inline void Merge4SortedHalf(
    __ubuf__ half* dst, __ubuf__ half* list0, __ubuf__ half* list1, __ubuf__ half* list2, __ubuf__ half* list3,
    uint32_t listLen, __ubuf__ half* scratch)
{
    __ubuf__ half* addrArray[4] = {list0, list1, list2, list3};
    uint64_t count = listLen | (static_cast<uint64_t>(listLen) << 16) | (static_cast<uint64_t>(listLen) << 32) |
                     (static_cast<uint64_t>(listLen) << 48);
    uint64_t config = 1 | (0b1111ULL << 8);
    vmrgsort4(scratch, addrArray, count, config);
    pipe_barrier(PIPE_ALL);
    uint16_t lenBurst = (listLen * 4 * 4 * 2 + 31) / 32;
    pto_copy_ubuf_to_ubuf(dst, scratch, 1, lenBurst, 0, 0);
    pipe_barrier(PIPE_ALL);
}

static AICORE inline void Merge2SortedHalf(
    __ubuf__ half* dst, __ubuf__ half* list0, __ubuf__ half* list1, uint32_t listLen, __ubuf__ half* scratch)
{
    __ubuf__ half* addrArray[4] = {list0, list1, list0, list1};
    uint64_t count = listLen | (static_cast<uint64_t>(listLen) << 16);
    uint64_t config = 1 | (0b0011ULL << 8);
    vmrgsort4(scratch, addrArray, count, config);
    pipe_barrier(PIPE_ALL);
    uint16_t lenBurst = (listLen * 2 * 4 * 2 + 31) / 32;
    pto_copy_ubuf_to_ubuf(dst, scratch, 1, lenBurst, 0, 0);
    pipe_barrier(PIPE_ALL);
}

static AICORE inline void SoftVbsort32Half(
    __ubuf__ half* dst, __ubuf__ half* src, __ubuf__ uint32_t* idx, uint32_t repeatNum, __ubuf__ half* scratch)
{
    __ubuf__ uint16_t* srcU16 = reinterpret_cast<__ubuf__ uint16_t*>(src);
    __ubuf__ uint16_t* idxU16 = reinterpret_cast<__ubuf__ uint16_t*>(idx);

    for (uint32_t r = 0; r < repeatNum; r++) {
        __ubuf__ uint16_t* srcBlock = srcU16 + r * VBS32_BLOCK_SIZE;
        __ubuf__ uint16_t* idxBlock = idxU16 + r * VBS32_BLOCK_SIZE * 2;
        __ubuf__ half* dstBlock = dst + r * VBS32_HALF_DST_STRIDE;

        // Step 1: Interleave (values, zeros, idx_lo, idx_hi) -> tuples in dstBlock
        // On kirinDev0000, CCE_VL=64 bytes, so vector_u16 holds 32 half elements.
        // each vintlv cascade produces 16 tuples (8 in result + 8 in result_hi)
        // so process 2 chunks of 16 values each.
        for (uint32_t chunk = 0; chunk < 2; chunk++) {
            __ubuf__ half* dstChunk0 = dstBlock + chunk * 64;
            __ubuf__ half* dstChunk1 = dstBlock + chunk * 64 + 32;

            __VEC_SCOPE__
            {
                vector_u16 valReg;
                vlds(valReg, srcBlock + chunk * 16, 0, NORM);

                vector_u16 idxReg;
                vlds(idxReg, idxBlock + chunk * 32, 0, NORM);

                vector_u16 idxLo, idxHi;
                vdintlv(idxLo, idxHi, idxReg, idxReg);

                vector_u16 zeroReg;
                vdup(zeroReg, static_cast<uint16_t>(0), pset_b16(PAT_ALL), MODE_ZEROING);

                vector_u16 tAC, tAC_hi;
                vintlv(tAC, tAC_hi, valReg, idxLo);

                vector_u16 tBD, tBD_hi;
                vintlv(tBD, tBD_hi, zeroReg, idxHi);

                vector_u16 result, result_hi;
                vintlv(result, result_hi, tAC, tBD);

                vsts((vector_f16&)result, dstChunk0, 0, NORM_B16, pset_b16(PAT_ALL));
                vsts((vector_f16&)result_hi, dstChunk1, 0, NORM_B16, pset_b16(PAT_ALL));
            }
            pipe_barrier(PIPE_ALL);
        }

        // Step 2: vmrgsort4 cascade to sort 32 tuples in descending order
        // Phase 1: merge 4 runs of 1 -> 8 runs of 4
        for (uint32_t g = 0; g < 8; g++) {
            __ubuf__ half* base = dstBlock + g * 16;
            Merge4SortedHalf(base, base, base + 4, base + 8, base + 12, 1, scratch);
        }

        // Phase 2: merge 4 runs of 4 -> 2 runs of 16
        for (uint32_t g = 0; g < 2; g++) {
            __ubuf__ half* base = dstBlock + g * 64;
            Merge4SortedHalf(base, base, base + 16, base + 32, base + 48, 4, scratch);
        }

        // Phase 3: merge 2 runs of 16 -> 1 run of 32
        Merge2SortedHalf(dstBlock, dstBlock, dstBlock + 64, 16, scratch);
    }
}

} // namespace pto
#endif // TSORT32_SOFT_HPP
