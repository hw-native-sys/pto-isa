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
import struct
import ctypes
import numpy as np

np.random.seed(19)


def gen_golden_data(case_name, gInfo):
    data_type = gInfo.datatype
    gShape0 = gInfo.gShape0
    gShape1 = gInfo.gShape1
    gShape2 = gInfo.gShape2
    gShape3 = gInfo.gShape3
    gShape4 = gInfo.gShape4
    gWholeShape0 = gInfo.gWholeShape0
    gWholeShape1 = gInfo.gWholeShape1
    gWholeShape2 = gInfo.gWholeShape2
    gWholeShape3 = gInfo.gWholeShape3
    gWholeShape4 = gInfo.gWholeShape4
    if gInfo.format == "ND" or gInfo.format == "NZ":
        input_arr = np.random.randint(
            -5, 5, size=(gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4)
        ).astype(data_type)
        output_arr = np.zeros(
            shape=(gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4), dtype=data_type
        )
        output_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3, 0:gShape4] = input_arr[
            0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3, 0:gShape4
        ]
    elif gInfo.format == "DN":
        input_arr = np.random.randint(
            -5, 5, size=(gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape4, gWholeShape3)
        ).astype(data_type)
        output_arr = np.zeros(
            shape=(gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape4, gWholeShape3), dtype=data_type
        )
        output_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape4, 0:gShape3] = input_arr[
            0:gShape0, 0:gShape1, 0:gShape2, 0:gShape4, 0:gShape3
        ]

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class GlobalTensorInfo:
    def __init__(
        self,
        datatype,
        format,
        gShape0,
        gShape1,
        gShape2,
        gShape3,
        gShape4,
        gWholeShape0,
        gWholeShape1,
        gWholeShape2,
        gWholeShape3,
        gWholeShape4,
    ):
        self.datatype = datatype
        self.format = format
        self.gShape0 = gShape0
        self.gShape1 = gShape1
        self.gShape2 = gShape2
        self.gShape3 = gShape3
        self.gShape4 = gShape4
        self.gWholeShape0 = gWholeShape0
        self.gWholeShape1 = gWholeShape1
        self.gWholeShape2 = gWholeShape2
        self.gWholeShape3 = gWholeShape3
        self.gWholeShape4 = gWholeShape4


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TStoreTest.case1",
        "TStoreTest.case2",
        "TStoreTest.case3",
        "TStoreTest.case4",
        "TStoreTest.case5",
        "TStoreTest.case6",
        "TStoreTest.case7",
        "TStoreTest.case8",
        "TStoreTest.case9",
        "TStoreTest.case10",
        "TStoreTest.case11",
        "TStoreTest.case12",
    ]

    case_params_list = [
        GlobalTensorInfo(np.float32, "NZ", 1, 1, 1, 16, 8, 1, 1, 2, 16, 8),
        GlobalTensorInfo(np.uint8, "NZ", 1, 2, 1, 16, 32, 2, 4, 2, 16, 32),
        GlobalTensorInfo(np.int16, "NZ", 2, 2, 2, 16, 16, 5, 3, 3, 16, 16),
        GlobalTensorInfo(np.float32, "ND", 2, 1, 1, 39, 47, 3, 2, 1, 43, 61),
        GlobalTensorInfo(np.int16, "ND", 1, 2, 1, 23, 121, 3, 2, 2, 35, 125),
        GlobalTensorInfo(np.int8, "ND", 2, 2, 3, 23, 47, 3, 3, 4, 32, 50),
        GlobalTensorInfo(np.float32, "DN", 1, 1, 1, 4, 21, 1, 1, 1, 8, 32),
        GlobalTensorInfo(np.uint16, "DN", 3, 1, 1, 1, 124, 5, 1, 1, 2, 128),
        GlobalTensorInfo(np.int8, "DN", 2, 3, 7, 47, 13, 2, 3, 7, 55, 29),
        GlobalTensorInfo(np.int64, "ND", 1, 1, 2, 16, 16, 2, 2, 2, 16, 16),
        GlobalTensorInfo(np.uint64, "DN", 1, 1, 2, 16, 64, 2, 2, 2, 16, 64),
        GlobalTensorInfo(np.int64, "ND", 1, 1, 2, 39, 47, 2, 2, 2, 43, 50),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
