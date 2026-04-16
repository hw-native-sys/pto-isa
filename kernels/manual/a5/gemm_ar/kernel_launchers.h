/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

void launchGemmCommAll(uint8_t *gemm_output, uint8_t *reduced_output, uint8_t *signal_matrix, uint8_t *queue_set,
                       uint8_t *hcclCtx, int rank, int nranks, void *stream, int num_compute_blocks);

void launchGemmCompute(uint8_t *gemm_output, uint8_t *src0, uint8_t *src1, uint8_t *queue_set, int rank, void *stream,
                       int block_num, uint32_t k_per_rank);
