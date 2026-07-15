/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cassert>
#include "acl/acl.h"
#include "aclrtlaunch_fa_custom.h"
#include "tiling/platform/platform_ascendc.h"
#include "../kernel/fa_custom.h"
#include "utils.h"

#define SUPPORTED_HEAD_SIZE 64
#define SUPPORTED_HEAD_SIZE2 128

#define TILING_KEY_128 128
#define TILING_KEY_256 256
#define TILING_KEY_512 512
#define TILING_KEY_1K 1024

#define MIN_SEQ_LEN_FOR_TK_256 256
#define MAX_SEQ_LEN_FOR_TK_256 1024
#define MIN_SEQ_LEN_FOR_TK_512 1024
#define MAX_SEQ_LEN_FOR_TK_512 8192
#define MIN_SEQ_LEN_FOR_TK_1K 1024

namespace ascendc_path {

#define CEIL(x, y) ((((x) + (y) - 1) / (y)) * y)

at::Tensor run_fa_custom(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool is_causal)
{
    at::Tensor out = at::empty(q.sizes(), at::TensorOptions().dtype(at::kFloat).device(q.options().device()));

    unsigned s0 = q.sizes()[0];
    unsigned s1 = k.sizes()[0];
    unsigned head_size = q.sizes()[1];

    constexpr int CUBE_S0 = kFaCubeS0;
    constexpr int CUBE_S1 = kFaCubeS1;

    assert(
        (head_size == SUPPORTED_HEAD_SIZE || head_size == SUPPORTED_HEAD_SIZE2) &&
        "Head Size has to be 64 or 128 for now");
    assert(s0 % CUBE_S0 == 0 && "S0 has to be CUBE_S0 multiple");

    // Dynamic tiling selection logic
    int tile_s1 = kFaTileS1; // Default
    if (s1 >= MIN_SEQ_LEN_FOR_TK_256 && s1 <= MAX_SEQ_LEN_FOR_TK_256) {
        tile_s1 = TILING_KEY_256;
    } else if (s1 >= MIN_SEQ_LEN_FOR_TK_512 && s1 <= MAX_SEQ_LEN_FOR_TK_512) {
        tile_s1 = TILING_KEY_512;
    } else if (s1 >= MIN_SEQ_LEN_FOR_TK_1K) {
        tile_s1 = TILING_KEY_1K;
    } else {
        tile_s1 = TILING_KEY_128;
    }

    // Verify tiling constraint
    assert(s1 % tile_s1 == 0 && "S1 has to be Tile_S1 multiple");

    size_t tile_factor = tile_s1 / CUBE_S1;
    size_t blockRow = s0 / CUBE_S0;
    size_t qk_fifo_stride = kFaCvFifoSize * CUBE_S0 * tile_s1;
    size_t qk_fifo_size = CEIL(qk_fifo_stride * blockRow * sizeof(float), 512);
    size_t p_fifo_half_size = CEIL(qk_fifo_stride * blockRow * 2, 512);
    size_t p_fifo_float_size = CEIL(kFaCvFifoSize * CUBE_S0 * blockRow * sizeof(float), 512);
    size_t num_tiles = s1 / tile_s1;
    size_t gsum_size = CEIL(s0 * num_tiles * sizeof(float), 512);
    size_t pvPart_size = s0 * head_size * sizeof(float);
    size_t oPartsTotal_size = CEIL(pvPart_size * num_tiles, 512);
    size_t pv_fifo_stride = kFaCvFifoSize * CUBE_S0 * head_size;
    size_t pv_fifo_size = CEIL(pv_fifo_stride * blockRow * sizeof(float), 512);

    size_t user_workspace_size =
        p_fifo_half_size + p_fifo_float_size + 2 * gsum_size + oPartsTotal_size + qk_fifo_size + pv_fifo_size;
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    size_t workspace_size = user_workspace_size + system_workspace_size;
    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(q.options().device()));

    EXEC_KERNEL_CMD(fa_custom, blockRow, q, k, v, out, s0, s1, head_size, is_causal, tile_s1, workspace_tensor);
    return out;
}
} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    // Declare the custom operator schema
    m.def("my_fa(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    // Register the custom operator implementation function
    m.impl("my_fa", TORCH_FN(ascendc_path::run_fa_custom));
}
} // namespace
