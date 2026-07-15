/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_ST_COST_CHECK_HPP
#define PTO_COSTMODEL_ST_COST_CHECK_HPP

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include "pto/costmodel/trace.hpp"

// Each st/st_a5_fit test measures one isolated PTO op and expects it to pay its own
// startup latency. The stream-aware change in BeginPtoInstr preserves the VECTOR pipe
// queue across PTO-instruction boundaries (so consecutive vec ops stream and share
// startup). Because g_trace_state is a single thread_local shared across all gtest
// cases, that queue would otherwise leak from one test into the next, making every op
// after the first skip its startup latency and under-report cycles. Reset the trace at
// the start of every test so each measured op runs in isolation. perf_sim_st is
// unaffected: it does not include this header and resets its stream via LAUNCH_KERNEL.
namespace pto_test_detail {
class PtoTraceResetListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& /*test_info*/) override { ::pto::mocker::ResetTrace(); }
};

inline int RegisterPtoTraceResetListener()
{
    ::testing::UnitTest::GetInstance()->listeners().Append(new PtoTraceResetListener);
    return 0;
}
} // namespace pto_test_detail

namespace {
const int g_pto_trace_reset_listener_registered = pto_test_detail::RegisterPtoTraceResetListener();
} // namespace

// Compare the cycle count stored in the most recently executed PTO trace record
// against an expected `profiling` value. The check passes when relative
// precision `1 - |profiling - actual| / profiling` is at least `accuracy`.
#define EXPECT_CYCLE_NEAR(profiling, accuracy)                                                                       \
    do {                                                                                                             \
        float _pto_actual = static_cast<float>(::pto::mocker::GetLastPtoInstrCycles());                              \
        float _pto_expected = static_cast<float>(profiling);                                                         \
        float _pto_precision = (_pto_expected == 0.0f) ?                                                             \
                                   ((_pto_actual == 0.0f) ? 1.0f : 0.0f) :                                           \
                                   std::max(0.0f, (1.0f - std::fabs(_pto_expected - _pto_actual) / _pto_expected));  \
        std::cout << "[CYCLE] " << ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() << "." \
                  << ::testing::UnitTest::GetInstance()->current_test_info()->name() << " actual=" << _pto_actual    \
                  << " expected=" << _pto_expected << " precision=" << _pto_precision                                \
                  << " accuracy=" << static_cast<float>(accuracy) << std::endl;                                      \
        EXPECT_GE(_pto_precision, static_cast<float>(accuracy));                                                     \
    } while (0)

#endif // PTO_COSTMODEL_ST_COST_CHECK_HPP
