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
np.random.seed(19)


def get_limits(t):
    try:
        info = np.iinfo(t)
        return info.min, info.max
    except ValueError:
        info = np.finfo(t)
        return info.min, info.max


def gen_golden(param):
    m, n = param.m, param.n

    s_min, s_max = get_limits(param.srctype)
    d_min, d_max = get_limits(param.dsttype)

    x1_gm = np.random.uniform(s_min + 5, s_max - 5, size=[m, n]).astype(param.srctype)

    if param.saturation_mode == "SatMode::ON":
        data_to_cast = np.clip(x1_gm, d_min, d_max)
    else:
        data_to_cast = x1_gm

    if param.mode == "RoundMode::CAST_RINT":
        rounded_data = np.rint(data_to_cast)
        golden = rounded_data.astype(param.dsttype)
    else:
        golden = data_to_cast.astype(param.dsttype)

    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


class TCvtParams:
    def __init__(self, srctype, dsttype, m, n, mode, saturation_mode="SatMode::OFF"):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode
        self.saturation_mode = saturation_mode

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
        "TCVTTest.case9",

        "TCVTTest.case10",
        "TCVTTest.case11",
        "TCVTTest.case12",
        "TCVTTest.case13",
        "TCVTTest.case14",
        "TCVTTest.case15"
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
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT"),

        TCvtParams(np.int32, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int8, np.float32, 128, 128, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float32, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int32, np.int16, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.int8, 32, 32, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON")
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_params_list[i])

        os.chdir(original_dir)
