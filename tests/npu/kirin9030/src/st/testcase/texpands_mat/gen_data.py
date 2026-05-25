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
import struct
import numpy as np
import ml_dtypes
from typing import Tuple
from typing import Optional

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def gen_golden_data(case_name, param):
    src_type = param.dtype
    if param.is_conv_tile:
        if param.shape_nc1hwc0 is not None:
            n, c1, h, w, c0 = param.shape_nc1hwc0
            golden = np.full([n, c1, h, w, c0], param.value, dtype=src_type)
        elif param.shape_ndc1hwc0 is not None:
            n, d, c1, h, w, c0 = param.shape_ndc1hwc0
            golden = np.full([n, d, c1, h, w, c0], param.value, dtype=src_type)
    else:
        golden = np.full([param.m, param.n], param.value, dtype=src_type)
    golden.tofile("./golden.bin")


class TexpandsParams:
    def __init__(
        self,
        dtype,
        value,
        m: Optional[int] = None,
        n: Optional[int] = None,
        shape_nc1hwc0: Optional[Tuple[int, int, int, int, int]] = None,
        shape_ndc1hwc0: Optional[Tuple[int, int, int, int, int, int]] = None,
        is_conv_tile=False,
    ):
        self.dtype = dtype
        self.value = value
        self.m = m
        self.n = n
        self.shape_nc1hwc0 = shape_nc1hwc0
        self.shape_ndc1hwc0 = shape_ndc1hwc0
        self.is_conv_tile = is_conv_tile


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXPANDSTest.case1",
        "TEXPANDSTest.case2",
        "TEXPANDSTest.case3",
        "TEXPANDSTest.case4",
    ]

    case_params_list = [
        # tile
        TexpandsParams(np.float16, value=2, m=128, n=128, is_conv_tile=False),
        TexpandsParams(np.int16, value=5, m=32, n=64, is_conv_tile=False),
        TexpandsParams(np.float16, value=3, shape_nc1hwc0=(1, 16, 7, 7, 16), is_conv_tile=True),
        TexpandsParams(np.int16, value=8, shape_nc1hwc0=(2, 5, 2, 3, 8), is_conv_tile=True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
