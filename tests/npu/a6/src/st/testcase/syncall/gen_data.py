#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np


if __name__ == "__main__":
    # Each participant reports 1 once it has seen every peer across all barriers.
    for case_name, participants in (
        ("SYNCALLTest.case_hard_aiv_only_all_blocks", 18),
        ("SYNCALLTest.case_hard_aic_only_all_blocks", 18),
        ("SYNCALLTest.case_hard_mix_1_1_all_blocks", 36),
        ("SYNCALLTest.case_soft_aiv_only_all_blocks", 18),
        ("SYNCALLTest.case_soft_aic_only_all_blocks", 18),
        ("SYNCALLTest.case_soft_mix_1_1_all_blocks", 36),
    ):
        golden = np.ones(participants, dtype=np.int32)
        os.makedirs(case_name, exist_ok=True)
        golden.tofile(os.path.join(case_name, "golden.bin"))

    # Partial cases: the first half join the barrier (1), the rest run but skip it (2).
    idle_mark = 2
    for case_name, launch_blocks, sync_blocks in (
        ("SYNCALLTest.case_soft_aiv_only_partial_blocks", 18, 9),
        ("SYNCALLTest.case_soft_aic_only_partial_blocks", 18, 9),
    ):
        golden = np.array([1 if i < sync_blocks else idle_mark for i in range(launch_blocks)], dtype=np.int32)
        os.makedirs(case_name, exist_ok=True)
        golden.tofile(os.path.join(case_name, "golden.bin"))
