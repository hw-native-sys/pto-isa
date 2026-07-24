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
    for case_name in (
        "SYNCALLTest.case_aiv_only_all_blocks",
        "SYNCALLTest.case_soft_aiv_only_all_blocks",
        "SYNCALLTest.case_mix_1_1_all_blocks",
        "SYNCALLTest.case_soft_mix_1_1_all_blocks",
    ):
        golden = np.ones(48, dtype=np.int32)
        os.makedirs(case_name, exist_ok=True)
        golden.tofile(os.path.join(case_name, "golden.bin"))
    for case_name in (
        "SYNCALLTest.case_mix_1_2_all_blocks",
        "SYNCALLTest.case_soft_mix_1_2_all_blocks",
    ):
        golden = np.ones(72, dtype=np.int32)
        os.makedirs(case_name, exist_ok=True)
        golden.tofile(os.path.join(case_name, "golden.bin"))
