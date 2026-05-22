#!/user/bin/python3
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


def ncdhw_to_fractal_z_3d(ncdhw_tensor: np.ndarray, c0: int, n0: int) -> np.ndarray:
    if ncdhw_tensor.ndim != 5:
        raise ValueError(f"The input must be a 5-dimensional NCDHW tensor, current dim : {ncdhw_tensor.ndim}")

    n, c, d, h, w = ncdhw_tensor.shape
    c1 = (c + c0 - 1) // c0
    n1 = (n + n0 - 1) // n0
    pad_c = c1 * c0 - c
    pad_n = n1 * n0 - n

    padded = ncdhw_tensor
    if pad_c > 0:
        padded = np.pad(padded, ((0, 0), (0, pad_c), (0, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
    if pad_n > 0:
        padded = np.pad(padded, ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)

    reshaped = padded.reshape(n1 * n0, c1, c0, d, h, w)
    transposed = np.transpose(reshaped, axes=(3, 1, 4, 5, 0, 2))
    fractal = transposed.reshape(d * c1 * h * w, n1, n0, c0)
    return fractal


def gen_golden_data(g_info):
    data_type = g_info.data_type
    src_n = g_info.src_n
    src_c = g_info.src_c
    src_d = g_info.src_d
    src_h = g_info.src_h
    src_w = g_info.src_w
    dst_n0 = g_info.dst_n0
    dst_c0 = g_info.dst_c0

    input_arr = np.random.randint(1, 5, size=(src_n, src_c, src_d, src_h, src_w)).astype(data_type)
    output_arr = ncdhw_to_fractal_z_3d(input_arr, dst_c0, dst_n0)

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class TTRANS3DParams:
    def __init__(self, case_name, data_type, src_n, src_c, src_d, src_h, src_w, dst_n0, dst_c0):
        self.case_name = case_name
        self.data_type = data_type
        self.src_n = src_n
        self.src_c = src_c
        self.src_d = src_d
        self.src_h = src_h
        self.src_w = src_w
        self.dst_n0 = dst_n0
        self.dst_c0 = dst_c0


if __name__ == "__main__":
    case_params_list = [
        TTRANS3DParams("TTRANS3DTest.case1_float32_2_4_2_2_2", np.float32, 2, 4, 2, 2, 2, 16, 8),
        TTRANS3DParams("TTRANS3DTest.case2_float32_4_5_2_2_4", np.float32, 4, 5, 2, 2, 4, 16, 8),
        TTRANS3DParams("TTRANS3DTest.case3_int32_17_3_3_2_2", np.int32, 17, 3, 3, 2, 2, 16, 8),
        TTRANS3DParams("TTRANS3DTest.case4_half_5_6_2_2_4", np.float16, 5, 6, 2, 2, 4, 16, 16),
        TTRANS3DParams("TTRANS3DTest.case5_half_19_14_2_4_2", np.float16, 19, 14, 2, 4, 2, 16, 16),
        TTRANS3DParams("TTRANS3DTest.case6_int16_8_13_2_3_4", np.int16, 8, 13, 2, 3, 4, 16, 16),
        TTRANS3DParams("TTRANS3DTest.case7_uint16_4_8_2_2_3", np.uint16, 4, 8, 2, 2, 3, 16, 16),
        TTRANS3DParams("TTRANS3DTest.case8_int8_7_28_2_3_4", np.int8, 7, 28, 2, 3, 4, 16, 32),
        TTRANS3DParams("TTRANS3DTest.case9_int8_16_26_3_3_4", np.int8, 16, 26, 3, 3, 4, 16, 32),
        TTRANS3DParams("TTRANS3DTest.case10_uint8_9_18_2_2_4", np.uint8, 9, 18, 2, 2, 4, 16, 32),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params)
        os.chdir(original_dir)
