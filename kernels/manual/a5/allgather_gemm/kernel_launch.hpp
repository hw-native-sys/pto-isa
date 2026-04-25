/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef KERNEL_LAUNCH_H_
#define KERNEL_LAUNCH_H_

#include <cstdint>

void launchRingCommStreaming(uint8_t *shmem_input, uint8_t *chunk_flags, uint8_t *hccl_ctx, int n_ranks, void *stream);

void launchAllGatherGemmComputeStreaming(uint8_t *output, uint8_t *shmem_input, uint8_t *src1, uint8_t *chunk_flags,
                                         void *stream, int launch_block_count);

#endif // KERNEL_LAUNCH_H_
