#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

import numpy as np
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def gen_golden(param):
    m, n = param.m, param.n

    x1_gm = NumExt.astype(np.random.random([m, n]) * 4.41, param.srctype)
    if param.mode == "RoundMode::CAST_RINT":
        if param.srctype == np.float32:
            if param.dsttype == np.float16 or param.dsttype == NumExt.bf16:
                golden = NumExt.astype(x1_gm, param.dsttype)
            else:
                golden = np.rint(x1_gm).astype(param.dsttype)
        elif param.srctype == np.float16 or param.srctype == NumExt.bf16:
            if param.dsttype == np.float32:
                golden = x1_gm.astype(param.dsttype)
            else:
                golden = np.rint(x1_gm).astype(param.dsttype)
        else:
            golden = x1_gm.astype(param.dsttype)
    NumExt.write_array("./x1_gm.bin", x1_gm, param.srctype)
    NumExt.write_array("./golden.bin", golden, param.dsttype)


class TCvtParams:
    def __init__(self, srctype, dsttype, m, n, mode):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode

if __name__ == "__main__":
    case_name_list = [
        "TCVTTest.case1",
        "TCVTTest.case2",
        "TCVTTest.case3",
        "TCVTTest.case4",
        "TCVTTest.case5",
        "TCVTTest.case6",
        "TCVTTest.case7",
        "TCVTTest.case8",
        "TCVTTest.case9"
    ]

    case_params_list = [
        TCvtParams(np.float32, np.int32, 128, 128, "RoundMode::CAST_RINT"),
        TCvtParams(np.int32, np.float32, 256, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int16, 16, 32, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 32, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.int32, 2, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 4, 4096, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.float16, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT")
    ]
    if ENABLE_BF16:
        case_name_list.extend([
            "TCVTTest.case10",
            "TCVTTest.case11",
        ])
        case_params_list.extend([
            TCvtParams(np.float32, NumExt.bf16, 64, 64, "RoundMode::CAST_RINT"),
            TCvtParams(NumExt.bf16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        ])

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_params_list[i])

        os.chdir(original_dir)
