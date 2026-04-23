#!/usr/bin/python3
#Copyright(c) 2026 Huawei Technologies Co., Ltd.
#This program is free software, you can redistribute it and / or modify it under the terms and conditions of
#CANN Open Software License Agreement Version 2.0(the "License").
#Please refer to the License for details.You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
import os
import numpy as np
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def gen_golden_data_tget_scale_addr(case_name, param):
    dtype = param.dtype

    src_row, src_col = [param.src_row, param.src_col]
    dst_row, dst_col = [param.dst_row, param.dst_col]

    input_data = NumExt.astype(np.random.randn(src_row, src_col), dtype)

    out_data = input_data[:dst_row, :dst_col]

    NumExt.write_array("input.bin", input_data, dtype)
    NumExt.write_array("golden.bin", out_data, dtype)


class TGetScaleAddrParams:
    def __init__(self, dtype, src_row, src_col, dst_row, dst_col):
        self.dtype = dtype
        self.src_row = src_row
        self.src_col = src_col
        self.dst_row = dst_row
        self.dst_col = dst_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)

    def substring(a, b):
        return f"_{a}x{b}"

    name = f"TGET_SCALE_ADDRTest.case_{dtype_str}"
    name += substring(param.src_row, param.src_col)
    name += substring(param.dst_row, param.dst_col)

    return name


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TGetScaleAddrParams(np.float32, 64, 64, 64, 64),
        TGetScaleAddrParams(np.int32, 64, 64, 64, 64),
        TGetScaleAddrParams(np.int16, 64, 64, 64, 64),
        TGetScaleAddrParams(np.float16, 16, 256, 16, 256),
    ]
    if ENABLE_BF16:
        case_params_list.append(TGetScaleAddrParams(NumExt.bf16, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tget_scale_addr(case_name, param)
        os.chdir(original_dir)