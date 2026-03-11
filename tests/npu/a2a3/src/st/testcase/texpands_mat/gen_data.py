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
        c1, h, w, c0 = param.shape_nc1hwc0[1], param.shape_nc1hwc0[2], param.shape_nc1hwc0[3], param.shape_nc1hwc0[4]
        n = param.shape_nc1hwc0[0]
        golden = np.full([n, c1, h, w, c0], param.value, dtype=src_type)
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
        is_conv_tile=False,
    ):
        self.dtype = dtype
        self.value = value
        self.m = m
        self.n = n
        self.shape_nc1hwc0 = shape_nc1hwc0
        self.is_conv_tile = is_conv_tile


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXPANDSTest.case1",
        "TEXPANDSTest.case2",
        "TEXPANDSTest.case3",
        "TEXPANDSTest.case4",
        "TEXPANDSTest.case5",
        "TEXPANDSTest.case6",
        "TEXPANDSTest.case7",
        "TEXPANDSTest.case8",
        "TEXPANDSTest.case9",
    ]

    case_params_list = [
        # tile
        TexpandsParams(np.float16, value=2, m=128, n=128, is_conv_tile=False),
        TexpandsParams(np.int16, value=5, m=32, n=64, is_conv_tile=False),
        TexpandsParams(np.float32, value=3, m=32, n=32, is_conv_tile=False),
        TexpandsParams(np.int8, value=1, m=32, n=32, is_conv_tile=False),
        TexpandsParams(bfloat16, value=7, m=256, n=256, is_conv_tile=False),
        # conv tile (N, C1, H, W, C0)
        TexpandsParams(np.float16, value=3, shape_nc1hwc0=(1, 16, 7, 7, 16), is_conv_tile=True),
        TexpandsParams(np.int16, value=8, shape_nc1hwc0=(2, 5, 2, 3, 8), is_conv_tile=True),
        TexpandsParams(np.int32, value=5, shape_nc1hwc0=(2, 5, 5, 1, 8), is_conv_tile=True),
        TexpandsParams(np.uint32, value=11, shape_nc1hwc0=(3, 4, 5, 1, 8), is_conv_tile=True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
