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

// Host entry for the four GTests in main.cpp.
//
// slot_stride is int32 elements between ranks on rank 0's array:
//   1  = packed (4 B) — same 64 B line — MUST PASS with word-safe Set
//   16 = one cache line per rank — MUST PASS (SYNCALL_SOFT_WORKSPACE_INT32 in type.hpp)
//
// Do not "fix" the bug by changing the packed tests to stride 16. The packed
// case is the contract under test.
bool RunSetFalseShare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int slot_stride);
