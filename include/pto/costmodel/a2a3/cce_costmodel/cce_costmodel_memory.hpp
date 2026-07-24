/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>

inline void set_l3d_rpt(auto rptConfig) { ::pto::mocker::RecordCceCall("set_l3d_rpt", 0, rptConfig); }

inline void copy_cbuf_to_bt(
    auto dst, auto src, auto convControl, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
    const uint64_t bytes = nBurst * lenBurst * 64;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_BT);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_BT, "copy_cbuf_to_bt", cycles, dst, src, convControl, nBurst, lenBurst,
        srcStride, dstStride);
}
inline void copy_cbuf_to_fbuf(auto dst, auto src, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
    const uint64_t bytes = nBurst * lenBurst * 128;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_FB);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_FB, "copy_cbuf_to_fbuf", cycles, dst, src, nBurst, lenBurst, srcStride,
        dstStride);
}
inline void copy_cbuf_to_gm(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
    const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_GM);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_GM, "copy_cbuf_to_gm", cycles, dst, src, sid, nBurst, lenBurst,
        srcStride, dstStride);
}
inline void copy_gm_to_cbuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto gmGap, auto l1Gap, auto pad)
{
    const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf", cycles, dst, src, sid, nBurst, lenBurst, gmGap,
        l1Gap, pad);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b16(
    auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue,
    auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
    const uint64_t bytes = ndNum * nValue * dValue * 2;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b16", cycles, dst, src, sid, ndNum,
        nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b32s(
    auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue,
    auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
    const uint64_t bytes = ndNum * nValue * dValue * 4;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b32s", cycles, dst, src, sid, ndNum,
        nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b8(
    auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue,
    auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
    const uint64_t bytes = ndNum * nValue * dValue;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b8", cycles, dst, src, sid, ndNum,
        nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_ubuf_align_b16(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap,
    auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b16", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_gm_to_ubuf_align_b32(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap,
    auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b32", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_gm_to_ubuf_align_b8(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap,
    auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b8", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_matrix_cc_to_cbuf(
    auto dst, auto src, auto sid, auto nSize, auto mSize, auto dstStrideD, auto srcStride, auto reserved, auto quantPre,
    auto reluMode, auto flag0, auto flag1)
{
    const uint64_t bytes = nSize * mSize * 2;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L0C_TO_L1);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L0C_TO_L1, "copy_matrix_cc_to_cbuf", cycles, dst, src, sid, nSize, mSize,
        dstStrideD, srcStride, reserved, quantPre, reluMode, flag0, flag1);
}
inline void copy_matrix_cc_to_gm(auto dst, auto src, auto xmReg, auto xtReg)
{
    const uint64_t rows = ExtractBits(xmReg, 16, 0xffffULL);
    const uint64_t cols = ExtractBits(xmReg, 4, 0xfffULL);
    const uint64_t bytes = rows * cols * 2;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L0C_TO_GM);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L0C_TO_GM, "copy_matrix_cc_to_gm", cycles, dst, src, xmReg, xtReg);
}
inline void copy_ubuf_to_gm_align_b16(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap,
    auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b16", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_gm_align_b32(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap,
    auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b32", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_gm_align_b8(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap,
    auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b8", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_ubuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcGap, auto dstGap)
{
    const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::UB_TO_UB);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_UB, "copy_ubuf_to_ubuf", cycles, dst, src, sid, nBurst, lenBurst,
        srcGap, dstGap);
}
inline void create_cbuf_matrix(auto dst, auto repeatConfig, auto value)
{
    const uint64_t repeatTimes = ExtractBits(repeatConfig, 0, 0x7fffULL);
    const uint64_t blockLen = ExtractBits(repeatConfig, 16, 0xffffULL);
    const uint64_t bytes = repeatTimes * blockLen * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_FILL);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_FILL, "create_cbuf_matrix", cycles, dst, repeatConfig, value);
}
inline void create_cbuf_matrix_bf16(auto dst, auto repeatConfig, auto value)
{
    const uint64_t repeatTimes = ExtractBits(repeatConfig, 0, 0x7fffULL);
    const uint64_t blockLen = ExtractBits(repeatConfig, 16, 0xffffULL);
    const uint64_t bytes = repeatTimes * blockLen * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_FILL);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_FILL, "create_cbuf_matrix_bf16", cycles, dst, repeatConfig, value);
}
inline void img2colv2_cbuf_to_ca(
    auto dst, auto src, auto stepK, auto stepM, auto posK, auto posM, auto strideW, auto strideH, auto filterW,
    auto filterH, auto dilationW, auto dilationH, auto highFilterW, auto highFilterH, auto transpose, auto fmatrixCtrl,
    auto channelSize)
{
    const uint64_t cycles = EstimateBandwidthCycles(stepK * stepM * 2, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "img2colv2_cbuf_to_ca", cycles, dst, src, stepK, stepM, posK,
        posM, strideW, strideH, filterW, filterH, dilationW, dilationH, highFilterW, highFilterH, transpose,
        fmatrixCtrl, channelSize);
}
inline void img2colv2_cbuf_to_cb(
    auto dst, auto src, auto stepK, auto stepM, auto posK, auto posM, auto strideW, auto strideH, auto filterW,
    auto filterH, auto dilationW, auto dilationH, auto highFilterW, auto highFilterH, auto transpose, auto fmatrixCtrl,
    auto channelSize)
{
    const uint64_t cycles = EstimateBandwidthCycles(stepK * stepM * 2, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "img2colv2_cbuf_to_cb", cycles, dst, src, stepK, stepM, posK,
        posM, strideW, strideH, filterW, filterH, dilationW, dilationH, highFilterW, highFilterH, transpose,
        fmatrixCtrl, channelSize);
}
inline void load_cbuf_to_ca(auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto sid, auto transpose)
{
    const uint64_t cycles = EstimateBandwidthCycles(
        repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "load_cbuf_to_ca", cycles, dst, src, baseIdx, repeat, srcStride,
        sid, transpose);
}
inline void load_cbuf_to_ca_transpose(
    auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto addrCalMode, auto dstFracStride)
{
    const uint64_t cycles = EstimateBandwidthCycles(
        repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "load_cbuf_to_ca_transpose", cycles, dst, src, baseIdx, repeat,
        srcStride, dstStride, addrCalMode, dstFracStride);
}
inline void load_cbuf_to_cb(
    auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto sid, auto transpose,
    auto addrCalMode)
{
    const uint64_t cycles = EstimateBandwidthCycles(
        repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "load_cbuf_to_cb", cycles, dst, src, baseIdx, repeat, srcStride,
        dstStride, sid, transpose, addrCalMode);
}
inline void load_cbuf_to_cb_transpose(
    auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto addrCalMode, auto dstFracStride)
{
    const uint64_t cycles = EstimateBandwidthCycles(
        repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "load_cbuf_to_cb_transpose", cycles, dst, src, baseIdx, repeat,
        srcStride, dstStride, addrCalMode, dstFracStride);
}
