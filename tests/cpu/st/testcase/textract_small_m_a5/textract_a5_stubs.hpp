/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_A5_STUBS_H_
#define TEXTRACT_A5_STUBS_H_

// Declare unused device intrinsics so host tests can include the complete A5 backend.
// They have no definitions: accidentally executing an unsupported path fails to link.
#include <pto/common/cpu_stub.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/utils.hpp>
using std::max;
using std::min;
struct vector_f32 {};
struct vector_f16 {};
struct vector_u8 {};
struct vector_u16 {};
struct vector_s8 {};
struct vector_s16 {};
struct vector_s32 {};
struct vector_u32 {};
struct vector_s64 {};
struct vector_u64 {};
struct vector_bool {};
struct vector_address {};
struct vector_align {};
namespace pto {
enum class QuantMode_t {
    NoQuant,
    F322F16,
    F322BF16,
    QF322B8_PRE,
    QF322HIF8_PRE,
    QF322F16_PRE,
    QF322BF16_PRE,
    QF322FP8_PRE,
    VQF322FP8_PRE,
    REQ8,
    DEQF16,
    QS322BF16_PRE,
    VQF322B8_PRE,
    VQF322HIF8_PRE,
    VQF322F16_PRE,
    VQF322BF16_PRE,
    VREQ8,
    VDEQF16,
    VQS322BF16_PRE
};
enum class DistVST;
} // namespace pto
using DistVST = pto::DistVST;
enum { NORM, POST_UPDATE, MODE_ZEROING, MODE_NORM, PAT_ALL, CCE_VL = 256 };
#define __VEC_SCOPE__
template <typename... Args>
vector_bool plt_b8(Args&&...);
template <typename... Args>
vector_bool plt_b16(Args&&...);
template <typename... Args>
vector_bool plt_b32(Args&&...);
template <typename... Args>
vector_bool pset_b8(Args&&...);
template <typename... Args>
vector_bool pset_b16(Args&&...);
template <typename... Args>
vector_bool pset_b32(Args&&...);
template <typename... Args>
void load_cbuf_to_ca_mx(Args&&...);
template <typename... Args>
void load_cbuf_to_cb_mx(Args&&...);
template <typename... Args>
void set_loop3_para(Args&&...);
template <typename... Args>
void set_channel_para(Args&&...);
template <typename... Args>
void set_fmatrix(Args&&...);
template <typename... Args>
void set_fmatrix_b(Args&&...);
template <typename... Args>
void set_l3d_rpt(Args&&...);
template <typename... Args>
void set_l3d_rpt_b(Args&&...);
template <typename... Args>
void set_padding(Args&&...);
template <typename... Args>
void set_padding_b(Args&&...);
template <typename... Args>
void pto_copy_ubuf_to_ubuf(Args&&...);
template <typename... Args>
void set_fpc(Args&&...);
template <typename... Args>
void set_quant_pre(Args&&...);

#endif // TEXTRACT_A5_STUBS_H_
