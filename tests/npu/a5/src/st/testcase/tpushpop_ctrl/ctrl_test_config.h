/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CTRL_TEST_CONFIG_H_
#define PTO_CTRL_TEST_CONFIG_H_

#include <cstdint>

namespace CtrlTest {
constexpr uint32_t JOBS = 3;
constexpr uint32_t MESSAGES_PER_JOB = 4;
constexpr uint32_t MESSAGE_COUNT = JOBS * MESSAGES_PER_JOB;
// Separate writers by one GM cache line to avoid shared scalar-store lines.
constexpr uint32_t OUTPUT_STRIDE = 128;
constexpr uint32_t OUTPUT_WORDS = 3 * OUTPUT_STRIDE;

template <int Key>
struct Case {
    static constexpr bool MERGE_NONE = Key == 3 || Key == 4;
    static constexpr bool MERGE_AND = Key == 2 || Key == 11 || Key == 12;
    static constexpr int SINGLE_LANE =
        (Key == 3 || Key == 9 || Key == 11) ? 0 : ((Key == 4 || Key == 10 || Key == 12) ? 1 : -1);
    static constexpr bool FORWARD = Key != 5 && Key != 7;
    static constexpr bool BACKWARD = Key != 6 && Key != 7;
    static constexpr bool VEC_PRODUCER = Key == 8;
};
} // namespace CtrlTest
#endif
