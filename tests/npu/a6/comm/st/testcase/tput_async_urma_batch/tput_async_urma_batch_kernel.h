/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUT_ASYNC_URMA_BATCH_KERNEL_H
#define TPUT_ASYNC_URMA_BATCH_KERNEL_H

bool RunTPutAsyncUrmaBatchBasic(int nRanks, int nDevices, int firstRankId, int firstDeviceId, bool sharedPool);
bool RunTPutAsyncUrmaBatchDocExample(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
bool RunTPutAsyncUrmaBatchPolicySuite(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
bool RunTPutAsyncUrmaLargeMultiWqe(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
bool RunTPutAsyncUrmaBatchConsume(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, bool sharedPool, bool oldNotifyPrefix = false);
bool RunTPutAsyncUrmaBatchMultiJetty(int nRanks, int nDevices, int firstRankId, int firstDeviceId);

#endif // TPUT_ASYNC_URMA_BATCH_KERNEL_H
