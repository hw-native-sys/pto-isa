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
from enum import Enum

np.random.seed(19)


class DataFormat(Enum):
    NCHW2NC1HWC0 = 1
    NC1HWC02C1HWN1N0C0 = 2
    GNCHW2GNC1HWC0 = 3
    GNC1HWC02C1HWN1N0C0 = 4
    NC1HWC02NC1C0HW = 5
    GNC1HWC02GNC1C0HW = 6


def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int) -> np.ndarray:
    if nchw_tensor.ndim != 4:
        raise ValueError(f"The input must be a 4-dimensional NCHW tensor, current dim : {nchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    n, c, h, w = nchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, pad_c), (0, 0), (0, 0))
        nchw_padded = np.pad(nchw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        nchw_padded = nchw_tensor

    nc1c0hw_tensor = nchw_padded.reshape(n, c1, c0, h, w)

    # NC1C0HW → NC1HWC0
    # origin index：0(n),1(c1),2(C0),3(h),4(w) → new index ：0,1,3,4,2
    nc1hwc0_tensor = np.transpose(nc1c0hw_tensor, axes=(0, 1, 3, 4, 2))

    return nc1hwc0_tensor


def gnchw_to_gnc1hwc0(gnchw_tensor: np.ndarray, c0: int) -> np.ndarray:
    if gnchw_tensor.ndim != 5:
        raise ValueError(f"The input must be a 5-dimensional GNCHW tensor, current dim : {gnchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    g, n, c, h, w = gnchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, 0), (0, pad_c), (0, 0), (0, 0))
        gnchw_padded = np.pad(gnchw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        gnchw_padded = gnchw_tensor

    gnc1c0hw_tensor = gnchw_padded.reshape(g, n, c1, c0, h, w)

    # GNC1C0HW → GNC1HWC0
    # origin index：0(g),1(n),2(c1),3(C0),4(h),5(w) → new index ：0,1,2,4,5,3
    gnc1hwc0_tensor = np.transpose(gnc1c0hw_tensor, axes=(0, 1, 2, 4, 5, 3))

    return gnc1hwc0_tensor


def _golden_nchw2_nc1hwc0(g_info):
    """NCHW -> NC1HWC0 golden; pads channel on full tensor when needed."""
    data_type = g_info.data_type
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4

    input_arr = np.random.randint(
        1, 5, size=(g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4)
    ).astype(data_type)
    golden_nchw = np.zeros(shape=(g_shape0, g_shape1 * g_shape4, g_shape2, g_shape3), dtype=data_type)
    golden_nchw = input_arr[0:g_shape0, 0:g_shape1 * g_shape4, 0:g_shape2, 0:g_shape3]
    pad_c1 = g_shape1 * g_shape4 - g_whole_shape2
    if pad_c1 > 0:
        pad_width1 = ((0, 0), (0, pad_c1), (0, 0), (0, 0))
        input_arr = np.pad(input_arr, pad_width1, mode="constant", constant_values=0)
    output_arr = nchw_to_nc1hwc0(golden_nchw, g_shape4)
    return input_arr, output_arr


def _golden_nc1hwc0_to_c1hwn1n0c0(g_info):
    """[N, C1, H, W, C0] -> [C1, H, W, N1, N0, C0]; pads N when needed."""
    data_type = g_info.data_type
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    dst_n1 = g_shape3
    dst_n0 = g_shape4
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4

    input_ori = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
    pad_n1 = dst_n1 * dst_n0 - g_info.g_whole_shape0
    if pad_n1 > 0:
        pad_width1 = ((0, pad_n1), (0, 0), (0, 0), (0, 0), (0, 0))
        input_arr = np.pad(input_ori, pad_width1, mode="constant", constant_values=0)
    else:
        input_arr = input_ori
    output_arr = input_arr.transpose([1, 2, 3, 0, 4])
    return input_arr, output_arr


def _golden_gnchw2_gnc1hwc0(g_info):
    """GNCHW -> GNC1HWC0 golden; pads channel on full tensor when needed."""
    data_type = g_info.data_type
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_shape6 = g_info.g_shape6

    input_arr = np.random.randint(
        1, 5, size=(g_shape6, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4)
    ).astype(data_type)
    golden_gnchw = np.zeros(shape=(g_shape6, g_shape0, g_shape1 * g_shape4, g_shape2, g_shape3), dtype=data_type)
    golden_gnchw = input_arr[0:g_shape6, 0:g_shape0, 0:g_shape1 * g_shape4, 0:g_shape2, 0:g_shape3]
    pad_c1 = g_shape1 * g_shape4 - g_whole_shape2
    if pad_c1 > 0:
        pad_width1 = ((0, 0), (0, 0), (0, pad_c1), (0, 0), (0, 0))
        input_arr = np.pad(input_arr, pad_width1, mode="constant", constant_values=0)
    output_arr = gnchw_to_gnc1hwc0(golden_gnchw, g_shape4)
    return input_arr, output_arr


def _golden_gnc1hwc0_to_c1hwn1n0c0(g_info):
    """[G, N, C1, H, W, C0] -> [G, C1, H, W, N1, N0, C0]; pads N when needed."""
    data_type = g_info.data_type
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    src_g = g_info.g_whole_shape5
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4
    dst_n1 = g_shape3
    dst_n0 = g_shape4

    input_ori = np.random.randint(1, 5, size=(src_g, src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
    pad_n1 = dst_n1 * dst_n0 - g_info.g_whole_shape0
    if pad_n1 > 0:
        pad_width1 = ((0, 0), (0, pad_n1), (0, 0), (0, 0), (0, 0), (0, 0))
        input_arr = np.pad(input_ori, pad_width1, mode="constant", constant_values=0)
    else:
        input_arr = input_ori
    output_arr = input_arr.transpose([0, 2, 3, 4, 1, 5])
    return input_arr, output_arr


def _golden_nc1hwc0_to_nc1c0hw(g_info):
    """[N, C1, H, W, C0] -> [N, C1, C0, H, W]; pads ??? when needed."""
    dtype = g_info.data_type
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4

    input_arr = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(dtype)

    # NC1HWC0 → NC1C0HW
    # origin index：0(n),1(c1),2(h),3(w),4(C0) → new index ：0,1,4,2,3
    output_arr = np.transpose(input_arr, axes=(0, 1, 4, 2, 3))
    output_arr = np.reshape(output_arr, (src_n, src_c1 * src_c0, src_h, src_w))
    return input_arr, output_arr


def _golden_gnc1hwc0_to_gnc1c0hw(g_info):
    """[G, N, C1, H, W, C0] -> [G, N, C1, C0, H, W]; pads ??? when needed."""
    dtype = g_info.data_type
    src_g = g_info.g_whole_shape0
    src_n = g_info.g_whole_shape1
    src_c1 = g_info.g_whole_shape2
    src_h = g_info.g_whole_shape3
    src_w = g_info.g_whole_shape4
    src_c0 = g_info.g_whole_shape5

    input_arr = np.random.randint(1, 5, size=(src_g, src_n, src_c1, src_h, src_w, src_c0)).astype(dtype)

    # GNC1HWC0 → GNC1C0HW
    # origin index：0(g),1(n),2(c1),3(h),4(w),5(C0) → new index ：0,1,2,5,3,4
    output_arr = np.transpose(input_arr, axes=(0, 1, 2, 5, 3, 4))
    output_arr = np.reshape(output_arr, (src_g, src_n, src_c1 * src_c0, src_h, src_w))
    return input_arr, output_arr


def gen_golden_data(g_info):
    shape1 = g_info.shape
    if shape1 == DataFormat["NCHW2NC1HWC0"].value:
        input_arr, output_arr = _golden_nchw2_nc1hwc0(g_info)
    elif shape1 == DataFormat["NC1HWC02C1HWN1N0C0"].value:
        input_arr, output_arr = _golden_nc1hwc0_to_c1hwn1n0c0(g_info)
    elif shape1 == DataFormat["GNCHW2GNC1HWC0"].value:
        input_arr, output_arr = _golden_gnchw2_gnc1hwc0(g_info)
    elif shape1 == DataFormat["GNC1HWC02C1HWN1N0C0"].value:
        input_arr, output_arr = _golden_gnc1hwc0_to_c1hwn1n0c0(g_info)
    elif shape1 == DataFormat["NC1HWC02NC1C0HW"].value:
        input_arr, output_arr = _golden_nc1hwc0_to_nc1c0hw(g_info)
    elif shape1 == DataFormat["GNC1HWC02GNC1C0HW"].value:
        input_arr, output_arr = _golden_gnc1hwc0_to_gnc1c0hw(g_info)
    else:
        data_type = g_info.data_type
        g_shape3 = g_info.g_shape3
        g_shape4 = g_info.g_shape4
        input_arr = np.random.randint(1, 5, [g_shape3, g_shape4]).astype(data_type)
        output_arr = np.zeros([g_shape3, g_shape4]).astype(data_type)

    input_arr.tofile("./input.bin")  # already pad N
    output_arr.tofile("./golden.bin")


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        g_shape0,
        g_shape1,
        g_shape2,
        g_shape3,
        g_shape4,
        g_shape5,
        g_whole_shape0,
        g_whole_shape1,
        g_whole_shape2,
        g_whole_shape3,
        g_whole_shape4,
        g_shape6=1,
        g_whole_shape5=1,
    ):
        self.case_name = case_name
        self.data_type = data_type
        self.shape = shape
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_shape5 = g_shape5
        self.g_shape6 = g_shape6
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4
        self.g_whole_shape5 = g_whole_shape5


if __name__ == "__main__":
    case_params_list = [
        # N, C1, H, W, C0, 1 <- 1, N, C, H, W
        TTRANSParams(
            "TTRANSConvTest.float32_1_32_6_56",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            4,
            6,
            56,
            8,
            1,
            1,
            1,
            32,
            6,
            56,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_1_8_1_8", np.int32, DataFormat["NCHW2NC1HWC0"].value, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_5_57_4_16",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            4,
            4,
            16,
            16,
            1,
            1,
            5,
            57,
            4,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.half_1_30_2_16",
            np.float16,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            2,
            2,
            16,
            16,
            1,
            1,
            1,
            30,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int16_7_53_6_16",
            np.int16,
            DataFormat["NCHW2NC1HWC0"].value,
            7,
            4,
            6,
            16,
            16,
            1,
            1,
            7,
            53,
            6,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_3_64_2_64",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            3,
            2,
            2,
            64,
            32,
            1,
            1,
            3,
            64,
            2,
            64,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_1_63_2_128",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            2,
            2,
            128,
            32,
            1,
            1,
            1,
            63,
            2,
            128,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_5_58_2_16",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            2,
            2,
            16,
            32,
            1,
            1,
            5,
            58,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint8_9_87_6_16",
            np.uint8,
            DataFormat["NCHW2NC1HWC0"].value,
            9,
            3,
            6,
            16,
            32,
            1,
            1,
            9,
            87,
            6,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_1_32_6_48",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            8,
            6,
            48,
            4,
            1,
            1,
            1,
            32,
            6,
            48,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint16_1_26_2_16",
            np.uint16,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            7,
            2,
            16,
            4,
            1,
            1,
            1,
            26,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_5_18_2_16",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            5,
            2,
            16,
            4,
            1,
            1,
            5,
            18,
            2,
            16,
        ),
        # C1, H, W, N1, N0, C0 <- N, C1, H, W, C0
        TTRANSParams(
            "TTRANSConvTest.float32_3_2_2_16_4",
            np.float32,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_37_2_3_10_8",
            np.int32,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            3,
            10,
            3,
            16,
            8,
            37,
            2,
            3,
            10,
            8,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_7_2_1_8_16",
            np.float16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            16,
            7,
            2,
            1,
            8,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_7_2_1_8_4",
            np.float16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            4,
            7,
            2,
            1,
            8,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint16_45_3_2_7_16",
            np.uint16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            3,
            2,
            7,
            3,
            16,
            16,
            45,
            3,
            2,
            7,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_25_5_1_6_32",
            np.int8,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            5,
            1,
            6,
            2,
            16,
            32,
            25,
            5,
            1,
            6,
            32,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint8_11_2_7_7_32",
            np.uint8,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            7,
            7,
            1,
            16,
            32,
            11,
            2,
            7,
            7,
            32,
        ),
        # G, N, C1, H, W, C0, 1 <- G, N, C, H, W
        TTRANSParams(
            "TTRANSConvTest.float32_1_1_32_6_56",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            4,
            6,
            56,
            8,
            1,
            1,
            1,
            32,
            6,
            56,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_4_1_8_1_8",
            np.int32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            1,
            1,
            8,
            8,
            1,
            1,
            1,
            8,
            1,
            8,
            4,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_5_30_4_16",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            5,
            2,
            4,
            16,
            16,
            1,
            1,
            5,
            30,
            4,
            16,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.half_1_1_30_2_16",
            np.float16,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            2,
            2,
            16,
            16,
            1,
            1,
            1,
            30,
            2,
            16,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_1_32_6_12",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            8,
            6,
            12,
            4,
            1,
            1,
            1,
            32,
            6,
            12,
            2,
            2,
        ),
        # G, C1, H, W, N1, N0, C0 <- G, N, C1, H, W, C0
        TTRANSParams(
            "TTRANSConvTest.float32_1_3_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_3_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_4_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            4,
            2,
            2,
            16,
            4,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_1_7_2_1_8_16",
            np.float16,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            16,
            7,
            2,
            1,
            8,
            16,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_4_7_2_1_8_4",
            np.float16,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            4,
            7,
            2,
            1,
            8,
            4,
            4,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_0",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            1,
            1,
            2,
            4,
            8,
            1,
            1,
            1,
            2,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_1",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            2,
            2,
            2,
            2,
            4,
            8,
            2,
            2,
            2,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_2",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            2,
            2,
            2,
            3,
            4,
            8,
            2,
            2,
            3,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_0",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            1,
            1,
            1,
            2,
            4,
            8,
            1,
            1,
            1,
            2,
            4,
            1,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_1",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            2,
            2,
            2,
            2,
            4,
            8,
            2,
            2,
            2,
            2,
            4,
            1,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_2",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            2,
            2,
            2,
            3,
            4,
            8,
            2,
            2,
            2,
            3,
            4,
            1,
            8
        ),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)
        os.chdir(original_dir)
