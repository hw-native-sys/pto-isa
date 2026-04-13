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

def gen_golden_data(case_name, param):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col] 
    row_valid, col_valid = [param.valid_row, param.valid_col]

    #Generate random input arrays
    input1 = NumExt.astype(np.random.randint(1, 10, size=[row, col]), dtype)

    #Save the input and golden data to binary files
    NumExt.write_array("input.bin", input1, dtype)
    NumExt.write_array("golden.bin", input1, dtype)


class TGetParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype 
        self.global_row = global_row 
        self.global_col = global_col 
        self.tile_row = tile_row 
        self.tile_col = tile_col 
        self.valid_row = valid_row 
        self.valid_col = valid_col

def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)

    def substring(a, b) -> str: 
        return f"_{a}x{b}"

    name = f"TGETTest.case_{dtype_str}" 
    name += substring(param.global_row, param.global_col) 
    name += substring(param.tile_row, param.tile_col) 
    name += substring(param.valid_row, param.valid_col)
    return name

if __name__ == "__main__":
    #Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    testcases_dir = os.path.join(script_dir, "testcases")

    #Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TGetParams(np.float32, 64, 64, 64, 64, 64, 64), 
        TGetParams(np.int32, 64, 64, 64, 64, 64, 64), 
        TGetParams(np.int16, 64, 64, 64, 64, 64, 64), 
        TGetParams(np.float16, 16, 256, 16, 256, 16, 256)
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TGetParams(NumExt.bf16, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param) 
        if not os.path.exists(case_name):
            os.makedirs(case_name) 
        original_dir = os.getcwd() 
        os.chdir(case_name) 
        gen_golden_data(case_name, param) 
        os.chdir(original_dir)
