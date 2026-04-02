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
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, is_bfloat16_dtype, write_array

print_C_case = True

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"

def type2str(t):
    if t is np.float16:
        return "half"
    if t is np.float32:
        return "float"
    if is_bfloat16_dtype(t):
        return "bfloat16_t"
    return np.dtype(t).name+"_t"

def gen_golden_data(case_name, param):
    src_type = param.src_type
    dst_type = param.dst_type

    rows, cols, valid_rows, valid_cols, idx_row, idx_col = param.rows, param.cols, param.valid_rows, param.valid_cols, param.idx_row, param.idx_col

    #gm = np.rand(1, 1e6, [m, k]).astype(src_type)
    gm = cast_for_compute(np.arange(1, valid_rows * valid_cols + 1).reshape([valid_rows, valid_cols]), src_type)

    golden = cast_for_compute(gm[idx_row:, idx_col:], dst_type)

    write_array("./input.bin", gm, src_type)
    write_array("./golden.bin", golden, dst_type)

    if print_C_case:
        print(f"TEST_F(TEXTRACTTest, {case_name}) "+"{")
        print(f"    textract_test<{type2str(src_type)}, {type2str(dst_type)}, {rows}, {cols}, {valid_rows}, {valid_cols}, {idx_row}, {idx_col}, {param.src_layout}, {param.dst_layout}>();"+"\n}\n")

class textractParams:
    def __init__(self, src_type, dst_type, rows, cols, valid_rows, valid_cols, idx_row, idx_col,src_layout, dst_layout) :
        self.src_type, self.dst_type = src_type, dst_type
        self.rows, self.cols, self.valid_rows, self.valid_cols, self.idx_row, self.idx_col = rows, cols, valid_rows, valid_cols, idx_row, idx_col
        self.src_layout, self.dst_layout = src_layout, dst_layout
        
def gen_case_name(param):
    return f"case_{type2str(param.src_type)}_{type2str(param.dst_type)}_{param.rows}_{param.cols}_{param.valid_rows}_{param.valid_cols}_IDX_{param.idx_row}_{param.idx_col}_L_{param.src_layout}_{param.dst_layout}"

if __name__ == "__main__":
    case_params_list = [
        textractParams(np.float16, np.float16, 32, 32, 32, 32, 0, 0, 0, 0),
        textractParams(np.float16, np.float32, 32, 32, 32, 32, 0, 0, 0, 0),
        textractParams(np.float32, np.float32, 128, 96, 128, 96, 0, 0, 0, 0),
        textractParams(np.int32,   np.float32, 128, 96, 128, 96, 0, 0, 0, 0),
        textractParams(np.int8,    np.int32, 128, 64, 128, 64, 0, 0, 0, 0),

        textractParams(np.float16, np.float16, 32, 32, 32, 32, 8, 16, 0, 0),
        textractParams(np.float16, np.float32, 32, 32, 32, 32, 8, 16, 0, 0),
        textractParams(np.float32, np.float32, 128, 96, 128, 96, 8, 16, 0, 0),
        textractParams(np.int32,   np.float32, 128, 96, 128, 96, 8, 16, 0, 0),
        textractParams(np.int8,    np.int32, 128, 64, 128, 64, 8, 16, 0, 0),

        textractParams(np.float16, np.float16, 32, 32, 31, 31, 8, 16, 0, 0),
        textractParams(np.float16, np.float32, 32, 32, 31, 31, 8, 16, 0, 0),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 0, 0),
        textractParams(np.int32,   np.float32, 128, 96, 125, 93, 8, 16, 0, 0),
        textractParams(np.int8,    np.int32, 128, 64, 125, 61, 8, 16, 0, 0),

        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 0, 1),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 0, 2),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 1, 0),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 1, 1),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 1, 2),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 2, 0),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 2, 1),
        textractParams(np.float32, np.float32, 128, 96, 125, 93, 8, 16, 2, 2),
    ]
    if ENABLE_BF16:
        case_params_list.extend([
            textractParams(BF16_DTYPE, BF16_DTYPE, 32, 32, 32, 32, 0, 0, 0, 0),
            textractParams(BF16_DTYPE, np.float32, 32, 32, 32, 32, 8, 16, 0, 0),
            textractParams(BF16_DTYPE, BF16_DTYPE, 32, 32, 31, 31, 8, 16, 0, 0),
        ])

    for case_param in case_params_list:
        case_name = gen_case_name(case_param)
        full_name = "TEXTRACTTest."+case_name
        if not os.path.exists(full_name):
            os.makedirs(full_name)
        original_dir = os.getcwd()
        os.chdir(full_name)

        gen_golden_data(case_name, case_param)

        os.chdir(original_dir)

