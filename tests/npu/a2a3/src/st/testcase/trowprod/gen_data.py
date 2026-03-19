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
from dataclasses import dataclass
import numpy as np

np.random.seed(42)


@dataclass
class TRowProdParams:
    name: str
    data_type: np.dtype
    dst_row: int
    src_row: int
    valid_row: int
    src_col: int
    src_valid_col: int


def gen_golden_data(p: TRowProdParams):
    data_type = p.data_type
    valid_row = p.valid_row

    # Use appropriate value range based on data type
    if np.issubdtype(data_type, np.integer):
        # For integer types, use small values to avoid overflow in product
        input_arr = np.random.randint(low=1, high=5, size=(p.src_row, p.src_col)).astype(data_type)
        # Use integer arithmetic for product to avoid overflow issues
        output_arr = np.zeros((p.dst_row, 1), dtype=np.int64)
        output_arr[:valid_row, 0] = input_arr[:valid_row, : p.src_valid_col].prod(axis=1, dtype=np.int64)
        output_arr = output_arr.astype(data_type)
    else:
        # For float types, use the original range
        input_arr = np.random.uniform(low=0.5, high=1.5, size=(p.src_row, p.src_col)).astype(data_type)
        output_arr = np.zeros((p.dst_row, 1)).astype(data_type)
        output_arr[:valid_row, 0] = input_arr[:valid_row, : p.src_valid_col].prod(axis=1)

    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


if __name__ == "__main__":
    case_params_list = [
        TRowProdParams("TROWPRODTest.case1", np.float32, 8, 1, 1, 8, 8),
        TRowProdParams("TROWPRODTest.case2", np.float32, 8, 1, 1, 16, 16),
        TRowProdParams("TROWPRODTest.case3", np.float32, 8, 1, 1, 128, 128),
        TRowProdParams("TROWPRODTest.case4", np.float32, 8, 1, 1, 8, 5),
        TRowProdParams("TROWPRODTest.case5", np.float32, 8, 1, 1, 16, 11),
        TRowProdParams("TROWPRODTest.case6", np.float32, 8, 3, 2, 8, 8),
        TRowProdParams("TROWPRODTest.case7", np.float32, 8, 3, 2, 24, 16),
        TRowProdParams("TROWPRODTest.case8", np.float32, 8, 4, 3, 16, 9),
        TRowProdParams("TROWPRODTest.case9", np.float16, 16, 1, 1, 16, 16),
        TRowProdParams("TROWPRODTest.case10", np.float16, 32, 26, 19, 32, 26),
        # int32 test cases
        TRowProdParams("TROWPRODTest.case11", np.int32, 8, 1, 1, 8, 8),
        TRowProdParams("TROWPRODTest.case12", np.int32, 8, 1, 1, 16, 16),
        TRowProdParams("TROWPRODTest.case13", np.int32, 8, 1, 1, 128, 128),
        TRowProdParams("TROWPRODTest.case14", np.int32, 8, 1, 1, 8, 5),
        TRowProdParams("TROWPRODTest.case15", np.int32, 8, 3, 2, 24, 16),
        # int16 test cases
        TRowProdParams("TROWPRODTest.case16", np.int16, 16, 1, 1, 16, 16),
        TRowProdParams("TROWPRODTest.case17", np.int16, 32, 26, 19, 32, 32),
        TRowProdParams("TROWPRODTest.case18", np.int16, 16, 1, 1, 16, 16),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
