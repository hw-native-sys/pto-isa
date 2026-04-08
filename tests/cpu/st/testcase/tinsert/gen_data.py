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

PRINT_C_CASE = False

np.random.seed(19)


def type2str(t):
    return "half" if t is np.float16 else "float" if t is np.float32 else np.dtype(t).name + "_t"


def gen_golden_data(case_name, param):
    src_type = param.src_type
    dst_type = param.dst_type

    rows, cols, src_valid_rows, src_valid_cols, dst_valid_rows, dst_valid_cols = (
        param.rows,
        param.cols,
        param.src_valid_rows,
        param.src_valid_cols,
        param.dst_valid_rows,
        param.dst_valid_cols,
    )

    # To fill random values use this instead the line below: gm = np.rand(1, 1e6, [m, k]).astype(src_type)
    in_data = (
        np.arange(1, src_valid_rows * src_valid_cols + 1).reshape([src_valid_rows, src_valid_cols]).astype(src_type)
    )

    golden = np.zeros([dst_valid_rows, dst_valid_cols], dst_type)
    golden[dst_valid_rows - src_valid_rows :, dst_valid_cols - src_valid_cols :] = in_data.astype(dst_type)

    in_data.tofile("./input.bin")
    golden.tofile("./golden.bin")

    if PRINT_C_CASE:
        print(f"TEST_F(TINSERTTest, {case_name}) " + "{")
        print(
            f"    tinsert_test<{type2str(src_type)}, {type2str(dst_type)}, TileType::{param.src_loc}, "
            + f"TileType::{param.dst_loc}, {rows}, {cols}, {src_valid_rows}, {src_valid_cols}, "
            + f"{dst_valid_rows}, {dst_valid_cols}, {param.src_layout}, {param.dst_layout}>();"
            + "\n}\n"
        )


class TInsertParams:
    def __init__(
        self,
        src_type,
        dst_type,
        src_loc,
        dst_loc,
        rows,
        cols,
        src_valid_rows,
        src_valid_cols,
        dst_valid_rows,
        dst_valid_cols,
        src_layout,
        dst_layout,
    ):
        self.src_type, self.dst_type = src_type, dst_type
        self.src_loc, self.dst_loc = src_loc, dst_loc
        self.rows, self.cols, self.src_valid_rows, self.src_valid_cols, self.dst_valid_rows, self.dst_valid_cols = (
            rows,
            cols,
            src_valid_rows,
            src_valid_cols,
            dst_valid_rows,
            dst_valid_cols,
        )
        self.src_layout, self.dst_layout = src_layout, dst_layout


def gen_case_name(param):
    return (
        f"case_{type2str(param.src_type)}_{type2str(param.dst_type)}_{param.src_loc}_{param.dst_loc}_"
        f"{param.rows}_{param.cols}_{param.src_valid_rows}_{param.src_valid_cols}_DST_"
        f"{param.src_valid_rows}_{param.src_valid_cols}_L_{param.src_layout}_{param.dst_layout}"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_params_list = [
        TInsertParams(np.float16, np.float16, "Mat", "Mat", 32, 32, 32, 32, 32, 32, 0, 0),
        TInsertParams(np.float16, np.float32, "Mat", "Mat", 32, 32, 32, 32, 32, 32, 0, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 128, 96, 128, 96, 0, 0),
        TInsertParams(np.int32, np.float32, "Mat", "Mat", 128, 96, 128, 96, 128, 96, 0, 0),
        TInsertParams(np.int8, np.int32, "Mat", "Mat", 128, 64, 128, 64, 128, 64, 0, 0),
        # ---------------------------------------------------
        TInsertParams(np.float16, np.float16, "Mat", "Mat", 32, 32, 24, 16, 32, 32, 0, 0),
        TInsertParams(np.float16, np.float32, "Mat", "Mat", 32, 32, 24, 16, 32, 32, 0, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 24, 16, 128, 96, 0, 0),
        TInsertParams(np.int32, np.float32, "Mat", "Mat", 128, 96, 24, 16, 128, 96, 0, 0),
        TInsertParams(np.int8, np.int32, "Mat", "Mat", 128, 64, 24, 16, 128, 64, 0, 0),
        # ---------------------------------------------------
        TInsertParams(np.float16, np.float16, "Mat", "Mat", 32, 32, 23, 16, 31, 31, 0, 0),
        TInsertParams(np.float16, np.float32, "Mat", "Mat", 32, 32, 23, 16, 31, 31, 0, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 23, 16, 125, 93, 0, 0),
        TInsertParams(np.int32, np.float32, "Mat", "Mat", 128, 96, 23, 16, 125, 93, 0, 0),
        TInsertParams(np.int8, np.int32, "Mat", "Mat", 128, 64, 23, 16, 125, 61, 0, 0),
        # ---------------------------------------------------
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 0, 1),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 0, 2),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 1, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 1, 1),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 1, 2),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 2, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 2, 1),
        TInsertParams(np.float32, np.float32, "Mat", "Mat", 128, 96, 18, 16, 125, 93, 2, 2),
        # ---------------------------------------------------
        TInsertParams(np.float16, np.float16, "Mat", "Vec", 32, 32, 8, 16, 32, 32, 0, 0),
        TInsertParams(np.float16, np.float32, "Mat", "Vec", 32, 32, 8, 16, 32, 32, 0, 0),
        TInsertParams(np.float32, np.float32, "Mat", "Vec", 128, 96, 8, 16, 128, 96, 0, 0),
        TInsertParams(np.int32, np.float32, "Mat", "Vec", 128, 96, 8, 16, 128, 96, 0, 0),
        TInsertParams(np.int8, np.int32, "Mat", "Vec", 128, 64, 8, 16, 128, 64, 0, 0),
        # ---------------------------------------------------
        TInsertParams(np.float16, np.float16, "Vec", "Vec", 32, 32, 8, 16, 32, 32, 0, 0),
        TInsertParams(np.float16, np.float32, "Vec", "Vec", 32, 32, 8, 16, 32, 32, 0, 0),
        TInsertParams(np.float32, np.float32, "Vec", "Vec", 128, 96, 8, 16, 128, 96, 0, 0),
        TInsertParams(np.int32, np.float32, "Vec", "Vec", 128, 96, 8, 16, 128, 96, 0, 0),
        TInsertParams(np.int8, np.int32, "Vec", "Vec", 128, 64, 8, 16, 128, 64, 0, 0),
    ]

    for case_param in case_params_list:
        case_name = gen_case_name(case_param)
        full_name = os.path.join(script_dir, "TINSERTTest." + case_name)
        os.makedirs(full_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(full_name)

        gen_golden_data(case_name, case_param)

        os.chdir(original_dir)
