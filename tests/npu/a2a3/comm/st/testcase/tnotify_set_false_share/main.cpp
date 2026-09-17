/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cerrno>
#include <cstdlib>
#include <gtest/gtest.h>

#include "tnotify_set_false_share_kernel.h"
#include "../comm_mpi.h"

// ---------------------------------------------------------------------------
// GTest names MUST keep the _2Ranks / _4Ranks suffixes: run_st.py -n 4 launches
// mpirun -n 2 with filter *2Ranks* then mpirun -n 4 with *4Ranks*.
//
// Default first_device_id=0 like other comm STs. Override with
// PTO_COMM_ST_FIRST_DEVICE_ID (this 910B2 image used 4 → davinci4–7).
// SKIP_IF_RANKS_LT comes from comm_mpi.h.
//
// Packed*    = stride 1, same 64 B line. Word-safe Set: all 50 iters must pass.
// CacheLine* = stride 16. Must stay green (Set still works when neighbors
//              do not share a line). Overwrite 1000→42 is asserted in the
//              kernel (proves Set is not SUM).
// ---------------------------------------------------------------------------

namespace {

int FirstDeviceId()
{
    const char* value = std::getenv("PTO_COMM_ST_FIRST_DEVICE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && parsed >= 0 ? static_cast<int>(parsed) : 0;
}

} // namespace

TEST(TNotifySetFalseShare, PackedInt32Slots_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunSetFalseShare(2, 2, 0, FirstDeviceId(), /*slot_stride=*/1));
}

TEST(TNotifySetFalseShare, CacheLineStridedSlots_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunSetFalseShare(2, 2, 0, FirstDeviceId(), /*slot_stride=*/16));
}

TEST(TNotifySetFalseShare, PackedInt32Slots_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunSetFalseShare(4, 4, 0, FirstDeviceId(), /*slot_stride=*/1));
}

TEST(TNotifySetFalseShare, CacheLineStridedSlots_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunSetFalseShare(4, 4, 0, FirstDeviceId(), /*slot_stride=*/16));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
