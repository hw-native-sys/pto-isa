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
import math
import numpy as np
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"

def matmul_reference(a, b, out_dtype):
    """
    Reference matmul that avoids BLAS calls (some macOS Python distributions may
    ship a broken/unsupported BLAS backend that returns incorrect results).

    a: (m, k)
    b: (k, n)
    returns: (m, n)
    """
    a = a.astype(out_dtype, copy=False)
    b = b.astype(out_dtype, copy=False)
    # (m, k, 1) * (1, k, n) -> (m, k, n) -> sum over k
    return (a[:, :, None] * b[None, :, :]).sum(axis=1, dtype=out_dtype)

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False
    repeats = param.repeats

    x1_gm = NumExt.astype(np.random.randint(1, 5, [repeats, m, k]), src_type)
    x2_gm = NumExt.astype(np.random.randint(1, 5, [repeats, k, n]), src_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(param.bias_type)
    golden=np.zeros([m,n], dst_type)

    for i in range(repeats):
        golden = golden + matmul_reference(x1_gm[i], x2_gm[i], dst_type).astype(dst_type)

        if is_atrans:
            x1_gm[i] = x1_gm[i].transpose()
        if is_btrans:
            x2_gm[i] = x2_gm[i].transpose()

    if is_bias:
        golden += bias_gm

    NumExt.write_array("./x1_gm.bin", x1_gm, src_type)
    NumExt.write_array("./x2_gm.bin", x2_gm, src_type)
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type = None, repeats=1):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.repeats = repeats
        self.is_bias = is_bias
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1",
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",

        "TMATMULTest.case_bias_1",
        "TMATMULTest.case_bias_2",
        "TMATMULTest.case_bias_5",
    ]
    if ENABLE_BF16:
        case_name_list.extend([
            "TMATMULTest.case_bf16_1",
            "TMATMULTest.case_bf16_bias_1",
        ])

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 40, 50, 60, False),
        tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, False,repeats=5),
        tmatmulParams(np.float32, np.float32, np.float32, 120, 110, 50, False),

        tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True,np.int32),
        tmatmulParams(np.float16, np.float16, np.float32, 16, 15, 16, True, np.float32),
        tmatmulParams(np.float32, np.float32, np.float32, 127, 128, 63, True, np.float32),
    ]
    if ENABLE_BF16:
        case_params_list.extend([
            tmatmulParams(NumExt.bf16, NumExt.bf16, np.float32, 40, 50, 60, False),
            tmatmulParams(NumExt.bf16, NumExt.bf16, np.float32, 16, 15, 16, True, np.float32),
        ])

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
