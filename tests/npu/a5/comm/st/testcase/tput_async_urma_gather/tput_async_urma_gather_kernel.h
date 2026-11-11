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

#include <cstdint>

bool RunPutAsyncUrmaGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherMixed(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSharedPoolMixed(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSinglePeer(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherFullSge(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSimtSplit(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint32_t segments_per_destination,
    bool shared_pool);

bool RunPutAsyncUrmaGatherMultiQueue(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherLargeBatch(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSubmitModes(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool shared_pool);

bool RunPutAsyncUrmaGatherSlotReuse(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool shared_pool);

bool RunPutAsyncUrmaGatherRingBoundary(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool shared_pool);

bool RunPutAsyncUrmaGatherByteBoundary(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherMultiAiv(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool shared_pool);

bool RunPutAsyncUrmaGatherSharedPool(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint32_t destinations_per_peer,
    uint32_t segments_per_destination, uint32_t jetties_per_core);

bool RunPutAsyncUrmaGatherPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSplitPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherNoCqeFinal(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherSharedPoolNoCqeFinal(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherMultiRound(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

bool RunPutAsyncUrmaGatherMultiRoundSingleQueue(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
