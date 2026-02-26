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
np.random.seed(19)


def gen_golden_data_tdequant(case_name, param):
    dst_dtype = param.dst_dtype
    src_dtype = param.src_dtype

    m, n = [param.dst_valid_rows, param.dst_valid_cols]
    dst_tile_shape = (param.dst_rows, param.dst_cols)
    src_tile_shape = (param.src_rows, param.src_cols)
    para_tile_shape = (param.para_rows, param.para_cols)

    if src_dtype == np.int8:
        src_valid = np.random.randint(-128, 128, size=(m, n), dtype=np.int8)
    elif src_dtype == np.int16:
        src_valid = np.random.randint(-32768, 32768, size=(m, n), dtype=np.int16)
    else:
        raise ValueError(f"Unsupported src dtype: {src_dtype}")
    src = np.zeros(src_tile_shape, dtype=src_dtype)
    src[:m, :n] = src_valid
    
    scale_valid = np.random.uniform(0.001, 1.0, size=(m, 1)).astype(np.float32)
    scale = np.zeros(para_tile_shape, dtype=np.float32)
    scale[:m, :1] = scale_valid

    if src_dtype == np.int8:
        offset_valid = np.random.uniform(-128, 127, size=(m, 1)).astype(np.float32)
    else:
        offset_valid = np.random.uniform(-32768, 32767, size=(m, 1)).astype(np.float32)
    offset = np.zeros(para_tile_shape, dtype=np.float32)
    offset[:m, :1] = offset_valid

    src_float = src_valid.astype(np.float32)
    offset_broadcast = np.broadcast_to(offset_valid[:, :1], (m, n))
    scale_broadcast = np.broadcast_to(scale_valid[:, :1], (m, n))

    dst_valid = (src_float - offset_broadcast) * scale_broadcast
    dst_valid = dst_valid.astype(np.float32)
    dst = np.zeros(dst_tile_shape, dtype=np.float32)
    dst[:m, :n] = dst_valid

    dst.tofile("golden.bin")
    src.tofile("srcInput.bin")
    scale.tofile("scaleInput.bin")
    offset.tofile("offsetInput.bin")

    return dst, src, scale, offset


class TDequantParams:
    def __init__(self, name, dst_dtype, src_dtype, dst_rows, dst_cols, src_rows, src_cols, 
                 dst_valid_rows, dst_valid_cols, para_rows, para_cols):
        self.name = name
        self.dst_dtype = dst_dtype
        self.src_dtype = src_dtype
        self.dst_rows = dst_rows
        self.dst_cols = dst_cols
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.dst_valid_rows = dst_valid_rows
        self.dst_valid_cols = dst_valid_cols
        self.para_rows = para_rows
        self.para_cols = para_cols

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TDequantParams("TDEQUANTTest.case1", np.float32, np.int8, 32, 32, 32, 32, 32, 32, 32, 32),
        TDequantParams("TDEQUANTTest.case2", np.float32, np.int16, 32, 32, 32, 32, 32, 32, 32, 32),
        TDequantParams("TDEQUANTTest.case3", np.float32, np.int8, 64, 64, 32, 64, 31, 31, 48, 32),
        TDequantParams("TDEQUANTTest.case4", np.float32, np.int16, 32, 32, 16, 32, 15, 15, 24, 16),
        TDequantParams("TDEQUANTTest.case5", np.float32, np.int8, 5, 512, 4, 512, 4, 511, 4, 32),
        TDequantParams("TDEQUANTTest.case6", np.float32, np.int16, 4, 256, 4, 256, 4, 255, 4, 16),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tdequant(case_name, param)
        os.chdir(original_dir)