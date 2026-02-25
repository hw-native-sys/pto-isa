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
from enum import Enum
np.random.seed(19)
np.set_printoptions(threshold=np.inf)

class DataFormat(Enum):
    ND2NZ = 1
    DN2NZ = 2
    ND2ND = 3
    NZ2NZ = 4
    DN2DN = 5
    DN2ZN = 6
    NC1HWC02NC1HWC0 = 7
    FZ2FZ = 8
    FZ4D2FZ4D = 9
    NHWC2NC1HWC0 = 10
    NCHW2NC1HWC0 = 11
    NCHW2FZ4D = 12
    NCDHW2NDC1HWC0 = 13
    NCDHW2FZ3D = 14


def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
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


def ncdhw_to_ndc1hwc0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    if ncdhw_tensor.ndim != 5:
        raise ValueError(f"The input must be a 5-dimensional NCDHW tensor, current dim: {ncdhw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    n, c, d, h, w = ncdhw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, pad_c), (0, 0), (0, 0), (0, 0))
        ncdhw_padded = np.pad(ncdhw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        ncdhw_padded = ncdhw_tensor

    nc1c0dhw_tensor = ncdhw_padded.reshape(n, c1, c0, d, h, w)

    # NC1C0DHW → NDC1HWC0
    ndc1hwc0_tensor = np.transpose(nc1c0dhw_tensor, axes=(0, 3, 1, 4, 5, 2))

    return ndc1hwc0_tensor


def nhwc_to_nc1hwc0(nhwc_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    if nhwc_tensor.ndim != 4:
        raise ValueError(f"The input must be a 4-dimensional NHWC tensor, current dim :{nhwc_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is :{c0}")

    n, h, w, c = nhwc_tensor.shape
    c1 = (c + c0 - 1) // c0  # mean math.ceil(c / c0)，
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_c))
        nhwc_padded = np.pad(nhwc_tensor, pad_width, mode="constant", constant_values=0)
    else:
        nhwc_padded = nhwc_tensor

    # NHWC → NCHW
    nchw_tensor = np.transpose(nhwc_padded, axes=(0, 3, 1, 2))  # shape=(n, C1×C0, h, w)

    # split c to C1×C0，NC1C0HW
    nc1c0hw_tensor = nchw_tensor.reshape(n, c1, c0, h, w)

    # NC1C0HW → NC1HWC0
    nc1hwc0_tensor = np.transpose(nc1c0hw_tensor, axes=(0, 1, 3, 4, 2))
    return nc1hwc0_tensor


def nchw_to_c1hw_n16_16_c0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    if nchw_tensor.ndim != 4:
        raise ValueError(f"The input must be a 4-dimensional NCHW tensor, current dim :{nchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is :{c0}")

    n_ori, c_ori, h, w = nchw_tensor.shape
    n_pad = ((n_ori + 15) // 16) * 16
    n_div_16 = n_pad // 16
    c_pad = ((c_ori + c0 - 1) // c0) * c0
    c1 = c_pad // c0
    c1hw = c1 * h * w

    pad_width = ((0, n_pad - n_ori), (0, c_pad - c_ori), (0, 0), (0, 0))
    nchw_padded = np.pad(nchw_tensor, pad_width, mode="constant", constant_values=0)
    nc1c0hw = nchw_padded.reshape(n_pad, c1, c0, h, w)
    n16_c1c0hw = nc1c0hw.reshape(n_div_16, 16, c1, c0, h, w)

    # (n_div_16),1(16),2(c1),3(C0),4(h),5(w) -> (c1),4(h),5(w),0(n_div_16),1(16),3(C0)
    rearranged = np.transpose(n16_c1c0hw, axes=(2, 4, 5, 0, 1, 3))

    # [c1hw, N/16, 16, C0]
    final_tensor = rearranged.reshape(c1hw, n_div_16, 16, c0)

    return final_tensor


def ncdhw_to_c1dhw_n16_16_c0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    if ncdhw_tensor.ndim != 5:
        raise ValueError(f"The input must be a 5-dimensional NCDHW tensor, current dim: {ncdhw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be a power of 2 (1/8/16/32), now is: {c0}")

    n_ori, c_ori, d, h, w = ncdhw_tensor.shape
    # Pad N to multiple of 16
    n_pad = ((n_ori + 15) // 16) * 16
    n_div_16 = n_pad // 16
    # Pad C to multiple of c0
    c_pad = ((c_ori + c0 - 1) // c0) * c0
    c1 = c_pad // c0
    c1dhw = c1 * d * h * w

    pad_width = ((0, n_pad - n_ori), (0, c_pad - c_ori), (0, 0), (0, 0), (0, 0))
    ncdhw_padded = np.pad(ncdhw_tensor, pad_width, mode="constant", constant_values=0)

    # Reshape to (N, C1, C0, D, H, W)
    nc1c0dhw = ncdhw_padded.reshape(n_pad, c1, c0, d, h, w)

    # Reshape N dimension to (N/16, 16, C1, C0, D, H, W)
    n16_c1c0dhw = nc1c0dhw.reshape(n_div_16, 16, c1, c0, d, h, w)

    # Transpose: (n_div_16, 16, c1, c0, d, h, w) -> (c1, d, h, w, n_div_16, 16, c0)
    transposed = np.transpose(n16_c1c0dhw, axes=(2, 4, 5, 6, 0, 1, 3))
    # Reshape to [C1DHW, N/16, 16, C0]
    final_tensor = transposed.reshape(c1dhw, n_div_16, 16, c0)
    return final_tensor


def gen_golden_data(case_name, param):
    src_type = param.atype
    shape0 = param.shape0
    shape1 = param.shape1
    shape2 = param.shape2
    shape3 = param.m
    shape4 = param.k
    whole_shape0 = param.ws0
    whole_shape1 = param.ws1
    whole_shape2 = param.ws2
    whole_shape3 = param.ws3
    whole_shape4 = param.ws4
    convtile_formats = {DataFormat["NC1HWC02NC1HWC0"].value, DataFormat["FZ2FZ"].value, DataFormat["FZ4D2FZ4D"].value}

    M, K, BASEM, BASEK, is_atrans = param.m, param.k, param.basem, param.basek, False
    c0_size = 16
    if src_type == np.float32:
        c0_size = 8
    elif src_type == np.int8 or src_type == np.uint8:
        c0_size = 32

    x1_gm = np.random.randint(1, 5, [M, K]).astype(src_type)
    golden = np.zeros([BASEM, BASEK]).astype(src_type)

    if param.load_type == DataFormat['ND2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小
        min_m = min(M, golden.shape[0])
        min_k = min(K, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
    elif param.load_type == DataFormat['DN2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape4, whole_shape3]).astype(src_type)
        golden = np.zeros([BASEK, BASEM]).astype(src_type)  # L1中Tile大小
        min_k = min(K, golden.shape[0])
        min_m = min(M, golden.shape[1])
        golden[:min_k, :min_m] = x1_gm[:min_k, :min_m]
    elif param.load_type == DataFormat['DN2ZN'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape4, whole_shape3]).astype(src_type)
        golden = np.zeros([BASEK, BASEM]).astype(src_type)  # L1中Tile大小
        min_k = min(K, golden.shape[0])
        min_m = min(M, golden.shape[1])
        golden[:min_k, :min_m] = x1_gm[:min_k, :min_m]
    elif param.load_type == DataFormat['ND2ND'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小

        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=3）
            0:M,         # d3: 截取前M个元素（对应目标 d3=64）
            0:K         # d4: 截取K个元素（对应目标 d4=128）
        ]
        flattened_submatrix = submatrix.reshape(BASEM, K)
        min_m = min(flattened_submatrix.shape[0], golden.shape[0])
        min_k = min(flattened_submatrix.shape[1], golden.shape[1])
        golden[:min_m, :min_k] = flattened_submatrix[:min_m, :min_k]
    elif param.load_type == DataFormat['DN2DN'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape4, whole_shape3]).astype(src_type)
        golden = np.zeros([BASEK, BASEM]).astype(src_type)

        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=3）
            0:K,         # d3: 截取前M个元素（对应目标 d3=64）
            0:M         # d4: 截取K个元素（对应目标 d4=128）
        ]

        flattened_submatrix = submatrix.reshape(BASEK, M)
        min_k = min(flattened_submatrix.shape[0], golden.shape[0])
        min_m = min(flattened_submatrix.shape[1], golden.shape[1])
        golden[:min_k, :min_m] = flattened_submatrix[:min_k, :min_m]
    elif param.load_type == DataFormat['NZ2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(src_type)

        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=4）
            0:M,         # d3: 截取前M个元素（对应目标 d3=16）
            0:K         # d4: 截取K个元素（对应目标 d4=8）
        ]
        new_submatrix = submatrix.reshape(
            submatrix.shape[0] * submatrix.shape[1], submatrix.shape[2], submatrix.shape[3], submatrix.shape[4])

        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小 [80,48]
        c0Size = 16
        if src_type == np.float32:
            c0Size = 8
        elif src_type == np.int8 or src_type == np.uint8:
            c0Size = 32
        assert (BASEK % c0Size) == 0, "BASEK should be c0Size aligned when matrix is NZ format"
        assert (BASEM % 16) == 0, "BASEM should be 16 aligned when matrix is NZ format"
        golden = golden.reshape((int(BASEM / 16), 16, int(BASEK / c0Size), c0Size)
                                ).transpose(2, 0, 1, 3).astype(src_type)  # [80,48] -> [6,5,16,8]

        golden[:new_submatrix.shape[0], :new_submatrix.shape[1],
               :new_submatrix.shape[2], :new_submatrix.shape[3]] = new_submatrix
    elif param.load_type in convtile_formats:
        x1_gm = np.random.randint(-5, 5, size=(whole_shape0, whole_shape1,
                                    whole_shape2, whole_shape3, whole_shape4)).astype(src_type)
        golden = np.zeros(shape=(shape0, shape1, shape2, shape3, shape4), dtype=src_type)
        golden = x1_gm[0:shape0, 0:shape1, 0:shape2, 0:shape3, 0:shape4]
    elif param.load_type == DataFormat['NHWC2NC1HWC0'].value:
        x1_gm = np.random.randint(1, 5, size=(whole_shape1,
                                    whole_shape2, whole_shape3, whole_shape4)).astype(src_type)
        golden_nhwc = np.zeros(shape=(shape0, shape2, shape3, shape1 * shape4), dtype=src_type)
        golden_nhwc = x1_gm[0:shape0, 0:shape2, 0:shape3, 0:shape1 * shape4]
        golden = nhwc_to_nc1hwc0(golden_nhwc, c0=c0_size)
    elif param.load_type == DataFormat['NCHW2NC1HWC0'].value:
        x1_gm = np.random.randint(1, 5, size=(whole_shape1,
                                    whole_shape2, whole_shape3, whole_shape4)).astype(src_type)
        golden_nchw = np.zeros(shape=(shape0, shape1 * shape4, shape2, shape3), dtype=src_type)
        golden_nchw = x1_gm[0:shape0, 0:shape1 * shape4, 0:shape2, 0:shape3]
        golden = nchw_to_nc1hwc0(golden_nchw, c0=c0_size)
    elif param.load_type == DataFormat['NCHW2FZ4D'].value:
        # [C1HW,N/16,16,C0,src_n,src_c,src_h,src_w,N,C,H,W]
        c1_h_w = shape0
        n_16 = shape1
        src_n = shape4
        src_c = whole_shape0
        src_h = whole_shape1
        src_w = whole_shape2
        n = whole_shape3
        c = whole_shape4
        h = param.basem
        w = param.basek

        x1_gm = np.random.randint(1, 5, size=(n, c, h, w)).astype(src_type)
        #  golden [NCHW] -> [C1HW,N/16,16,C0]
        golden_nchw = np.zeros(shape=(n_16 * 16, c1_h_w * c0_size // (src_h * src_w), src_h, src_w), dtype=src_type)
        golden_nchw = x1_gm[0:src_n, 0:src_c, 0:src_h, 0:src_w]
        golden = nchw_to_c1hw_n16_16_c0(golden_nchw, c0=c0_size)
    elif param.load_type == DataFormat['NCDHW2NDC1HWC0'].value:
        # 参数说明:
        # shape0: 输出的N维度
        # shape1: 输出的D维度  
        # shape2: 输出的C1维度
        # shape3: 输出的H维度
        # shape4: 输出的W维度
        # c0_size: C0大小由数据类型决定
        
        # whole_shape0: 输入x1_gm的N维度
        # whole_shape1: 输入x1_gm的C维度
        # whole_shape2: 输入x1_gm的D维度
        # whole_shape3: 输入x1_gm的H维度
        # whole_shape4: 输入x1_gm的W维度

        x1_gm = np.random.randint(1, 5, size=(whole_shape0, whole_shape1,
                                    whole_shape2, whole_shape3, whole_shape4)).astype(src_type)
        target_n = shape0
        target_c = shape2 * c0_size
        target_d = shape1
        target_h = shape3
        target_w = shape4
        golden_ncdhw = np.zeros((target_n, target_c, target_d, target_h, target_w), dtype=src_type)
        golden_ncdhw[:] = x1_gm[:target_n, :target_c, :target_d, :target_h, :target_w]
        # 转换为NDC1HWC0格式
        golden = ncdhw_to_ndc1hwc0(golden_ncdhw, c0=c0_size)
    elif param.load_type == DataFormat['NCDHW2FZ3D'].value:
        # whole_shape is guaranteed to be >= corresponding shape
        src_n = shape0
        src_c = shape1
        src_d = shape2
        src_h = shape3
        src_w = shape4
        
        n = whole_shape0
        c = whole_shape1
        d = whole_shape2
        h = whole_shape3
        w = whole_shape4

        c1_d_h_w = param.basem
        n_16 = param.basek
        # Generate random input data
        x1_gm = np.random.randint(1, 5, size=(n, c, d, h, w)).astype(src_type)
        
        # Create golden NCDHW tensor from the input
        golden_nchw = np.zeros(shape=(n_16 * 16, c1_d_h_w * c0_size // (src_h * src_w), src_h, src_w), dtype=src_type)
        golden_ncdhw = x1_gm[0:src_n, 0:src_c, 0:src_d, 0:src_h, 0:src_w]
        # Convert to FZ3D format: [C1DHW, N/16, 16, C0]
        golden = ncdhw_to_c1dhw_n16_16_c0(golden_ncdhw, c0=c0_size)

    x2_gm = np.random.randint(1, 5, [M, K]).astype(src_type)
    if param.load_type == DataFormat['ND2NZ'].value:
        assert (BASEM % 16) == 0, "BASEM should be 16 aligned when matrix A is NZ format"
        assert (BASEK % c0_size) == 0, "BASEK should be c0_size aligned when matrix A is NZ format"
        golden = golden.reshape(
            (int(BASEM / 16), 16, int(BASEK / c0_size), c0_size)).transpose(2, 0, 1, 3).astype(src_type)
    elif param.load_type == DataFormat['DN2NZ'].value:
        golden = golden.transpose()
        assert (BASEK % 16) == 0, "BASEK should be 16 aligned when matrix A is NZ format"
        assert (BASEM % c0_size) == 0, "BASEM should be c0_size aligned when matrix A is NZ format"
        golden = golden.reshape(
            (int(BASEM / 16), 16, int(BASEK / c0_size), c0_size)).transpose(2, 0, 1, 3).astype(src_type)
    elif param.load_type == DataFormat['DN2ZN'].value:
        assert (BASEK % 16) == 0, "BASEK should be 16 aligned when matrix A is NZ format"
        assert (BASEM % c0_size) == 0, "BASEM should be c0_size aligned when matrix A is NZ format"
        golden = golden.reshape(
            (int(BASEK / 16), 16, int(BASEM / c0_size), c0_size)).transpose(2, 0, 1, 3).astype(src_type)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TloadParams:
    def __init__(self, atype, shape0, shape1, shape2, m, k, ws0, ws1, ws2, ws3, ws4, basem, basek, load_type):
        self.atype = atype
        self.m = m
        self.k = k
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2

        self.ws0 = ws0
        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3
        self.ws4 = ws4

        self.basem = basem  # L1 row
        self.basek = basek  # L1 col
        self.load_type = load_type


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TLOADMIXTest.1_1_1_128_128_half_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_int8_t_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_float_ND2NZ",
        "TLOADMIXTest.1_1_1_64_128_half_DN2NZ",
        "TLOADMIXTest.1_1_1_63_127_half_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_float_ND2ND",
        "TLOADMIXTest.1_1_1_37_126_int8_t_ND2ND",
        "TLOADMIXTest.1_2_3_64_128_1_3_4_128_128_384_128_half_ND2ND",

        "TLOADMIXTest.1_2_3_33_99_1_2_3_33_99_int8_t_ND2ND",
        "TLOADMIXTest.1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ",

        "TLOADMIXTest.1_1_1_51_123_1_1_1_64_128_64_128_float_DN2NZ",
        "TLOADMIXTest.1_1_1_63_127_1_1_1_63_127_64_128_half_DN2NZ",

        "TLOADMIXTest.1_1_1_128_128_1_1_1_128_128_128_128_float_DN2DN",
        "TLOADMIXTest.1_1_1_37_126_1_1_1_37_126_64_126_int8_t_DN2DN",
        "TLOADMIXTest.1_2_3_64_128_1_3_4_96_128_64_768_half_DN2DN",

        "TLOADMIXTest.2_2_4_16_8_2_2_4_16_8_80_48_float_NZ2NZ",
        "TLOADMIXTest.1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ",
        "TLOADMIXTest.1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ",

        "TLOADMIXTest.1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND",
        "TLOADMIXTest.1_2_1_64_128_1_3_4_128_128_128_128_uint64_t_ND2ND",

        "TLOADMIXTest.1_2_1_64_128_1_3_4_128_128_128_128_fp4x2_e1m2_t_ND2ND",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_fp4x2_e2m1_t_ND2NZ",
        "TLOADMIXTest.1_8_4_16_32_1_9_4_16_32_80_256_fp4x2_e1m2_t_NZ2NZ",
        "TLOADMIXTest.1_1_1_37_126_1_1_1_37_126_64_126_fp4x2_e1m2_t_DN2DN",

        "TLOADMIXTest.1_1_1_33_99_1_1_1_64_128_48_112_half_DN2ZN",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_DN2ZN",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_fp4x2_e1m2_t_DN2ZN",

        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_1_3_16_128_32_3_4_1024_1024_32",  # cut N H
        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_3_2_128_8_32_3_2_128_128_32",  # cut W
        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_3_2_8_128_32_3_8_8_128_32",  # cut C1
        "TLOADMIXTest.NC1HWC02NC1HWC0_bfloat16_1_6_10_100_16_1_6_100_100_16",  # cut H
        "TLOADMIXTest.NC1HWC02NC1HWC0_bfloat16_10_16_16_2_16_256_16_100_16_16",  # cut N C1 W
        "TLOADMIXTest.NC1HWC02NC1HWC0_bfloat16_1_1_1_8192_16_8_16_16_8192_16",  # cut N C1 H
        "TLOADMIXTest.NC1HWC02NC1HWC0_float_1_1_56_112_8_2_3_224_224_8",  # cut N C1 H W

        "TLOADMIXTest.FZ2FZ_bfloat16_1_7_7_20_16_3_7_7_100_16",  # cut N C1
        "TLOADMIXTest.FZ2FZ_bfloat16_64_7_7_2_16_256_7_7_16_16",  # cut N C1
        "TLOADMIXTest.FZ2FZ_bfloat16_96_3_3_8_16_256_3_3_8_16",  # cut C1
        "TLOADMIXTest.FZ2FZ_int8_t_1_3_3_64_32_3_3_3_128_32",  # cut N C1
        "TLOADMIXTest.FZ2FZ_int8_t_8_5_5_32_32_8_5_5_128_32",  # cut N
        "TLOADMIXTest.FZ2FZ_float_70_7_7_2_8_256_7_7_256_8",  # cut C1 N

        "TLOADMIXTest.FZ4D2FZ4D_bfloat16_1_49_7_16_16_1_980_32_16_16",  # cut C1HW N
        "TLOADMIXTest.FZ4D2FZ4D_bfloat16_1_81_3_16_16_1_90_3_16_16",  # cut C1HW
        "TLOADMIXTest.FZ4D2FZ4D_int8_t_1_63_3_16_32_1_63_9_16_32",  # cut N
        "TLOADMIXTest.FZ4D2FZ4D_int8_t_1_125_3_16_32_1_250_5_16_32",  # cut C1HW N
        "TLOADMIXTest.FZ4D2FZ4D_float_1_126_3_16_8_1_4704_7_16_8",  # cut C1HW N

        "TLOADMIXTest.NHWC2NC1HWC0_int8_t_1_3_11_109_32_1_3_1023_1000_111",
        "TLOADMIXTest.NHWC2NC1HWC0_int8_t_3_2_121_9_32_1_3_128_127_65",
        "TLOADMIXTest.NHWC2NC1HWC0_bfloat16_1_6_10_100_16_1_1_100_100_96",
        "TLOADMIXTest.NHWC2NC1HWC0_bfloat16_10_16_16_2_16_1_256_100_16_255",
        "TLOADMIXTest.NHWC2NC1HWC0_float_1_1_56_112_8_1_2_224_224_25",
        "TLOADMIXTest.NHWC2NC1HWC0_float_2_1_56_43_8_1_3_333_188_19",

        "TLOADMIXTest.NCHW2NC1HWC0_int8_t_1_3_11_109_32_1_3_111_1023_109",
        "TLOADMIXTest.NCHW2NC1HWC0_int8_t_3_2_121_9_32_1_3_65_128_127",
        "TLOADMIXTest.NCHW2NC1HWC0_bfloat16_1_6_10_100_16_1_1_96_100_100",
        "TLOADMIXTest.NCHW2NC1HWC0_bfloat16_10_16_16_2_16_1_256_255_100_16",
        "TLOADMIXTest.NCHW2NC1HWC0_float_1_1_56_112_8_1_2_25_224_112",
        "TLOADMIXTest.NCHW2NC1HWC0_float_2_1_56_43_8_1_3_19_333_188",

        "TLOADMIXTest.NCHW2FZ4D_int8_t_75_3_16_32_48_95_5_5_50_111_5_5", # [C1HW,N/16,16,C0,src_n,src_c,src_h,src_w,N,C,H,W]
        "TLOADMIXTest.NCHW2FZ4D_int8_t_98_4_16_32_64_58_7_7_121_127_7_7", # src_c <= C1*C0
        "TLOADMIXTest.NCHW2FZ4D_bfloat16_63_6_16_16_96_111_3_3_220_96_3_3",
        "TLOADMIXTest.NCHW2FZ4D_bfloat16_75_4_16_16_64_48_5_5_100_50_5_5",
        "TLOADMIXTest.NCHW2FZ4D_float_50_3_16_8_48_14_5_5_224_224_5_5",
        "TLOADMIXTest.NCHW2FZ4D_float_27_2_16_8_32_24_3_3_333_188_3_3",

        "TLOADMIXTest.NCDHW2NDC1HWC0_int8_t_1_2_3_11_109_3_111_2_1023_109",
        "TLOADMIXTest.NCDHW2NDC1HWC0_int8_t_3_3_2_15_9_3_65_4_30_50",
        "TLOADMIXTest.NCDHW2NDC1HWC0_bfloat16_1_4_6_10_10_1_96_6_100_10",
        "TLOADMIXTest.NCDHW2NDC1HWC0_bfloat16_10_2_8_16_2_256_128_2_100_7",
        "TLOADMIXTest.NCDHW2NDC1HWC0_float_1_5_1_25_31_2_25_7_112_31",
        "TLOADMIXTest.NCDHW2NDC1HWC0_float_2_2_1_43_43_3_19_2_155_180",

        "TLOADMIXTest.NCDHW2FZ3D_int8_t_48_95_2_5_5_50_111_4_5_5_150_3", # [srcN srcC srcD_srcH srcW N C D H W C1DHW N/16]
        "TLOADMIXTest.NCDHW2FZ3D_int8_t_32_58_2_7_7_63_127_2_7_7_196_2", # src_c <= C1*C0
        "TLOADMIXTest.NCDHW2FZ3D_bfloat16_48_111_2_3_3_110_112_2_3_3_126_3",
        "TLOADMIXTest.NCDHW2FZ3D_bfloat16_32_48_3_3_3_70_50_4_3_3_81_2",
        "TLOADMIXTest.NCDHW2FZ3D_float_48_14_5_2_2_224_224_7_2_2_40_3",
        "TLOADMIXTest.NCDHW2FZ3D_float_32_24_2_3_3_333_188_2_3_3_54_2",
    ]

    case_params_list = [
        TloadParams(np.float16, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.int8, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float16, 1, 1, 1, 64, 128, 1, 1, 1, 64, 128, 64, 128, DataFormat["DN2NZ"].value),
        TloadParams(np.float16, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2ND"].value),
        TloadParams(np.int8, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128, DataFormat["ND2ND"].value),
        TloadParams(np.float16, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128, DataFormat["ND2ND"].value),
        TloadParams(np.int8, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128, DataFormat["ND2ND"].value),
        TloadParams(np.float16, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112, DataFormat["ND2NZ"].value),
        TloadParams(np.int8, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 51, 123, 1, 1, 1, 64, 128, 64, 128, DataFormat["DN2NZ"].value),
        TloadParams(np.float16, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128, DataFormat["DN2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["DN2DN"].value),
        TloadParams(np.int8, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126, DataFormat["DN2DN"].value),
        TloadParams(np.float16, 1, 2, 3, 64, 128, 1, 3, 4, 96, 128, 64, 768, DataFormat["DN2DN"].value),
        TloadParams(np.float32, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48, DataFormat["NZ2NZ"].value),
        TloadParams(np.float16, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160, DataFormat["NZ2NZ"].value),
        TloadParams(np.int8, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256, DataFormat["NZ2NZ"].value),
        TloadParams(np.int64, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120, DataFormat["ND2ND"].value),
        TloadParams(np.uint64, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128, DataFormat["ND2ND"].value),
        # fp4 input use uint8 for test
        TloadParams(np.uint8, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128, DataFormat["ND2ND"].value),
        TloadParams(np.uint8, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.uint8, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256, DataFormat["NZ2NZ"].value),
        TloadParams(np.uint8, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126, DataFormat["DN2DN"].value),
        TloadParams(np.float16, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112, DataFormat["DN2ZN"].value),
        TloadParams(np.int8, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128, DataFormat["DN2ZN"].value),
        TloadParams(np.uint8, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128, DataFormat["DN2ZN"].value),  # fp4
        TloadParams(np.int8, 1, 3, 16, 128, 32, 3, 4, 1024, 1024, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 128, 8, 32, 3, 2, 128, 128, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 8, 128, 32, 3, 8, 8, 128, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float32, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 96, 3, 3, 8, 16, 256, 3, 3, 8, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.int8, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.int8, 8, 5, 5, 32, 32, 8, 5, 5, 128, 32, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float32, 70, 7, 7, 2, 8, 256, 7, 7, 256, 8, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.float16, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.int8, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.int8, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.float32, 1, 126, 3, 16, 8, 1, 4704, 7, 16, 8, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.int8, 1, 3, 11, 109, 32, 1, 3, 1023, 1000, 111, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 121, 9, 32, 1, 3, 128, 127, 65, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.float16, 1, 6, 10, 100, 16, 1, 1, 100, 100, 96, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.float16, 10, 16, 16, 2, 16, 1, 256, 100, 16, 255, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.float32, 1, 1, 56, 112, 8, 1, 2, 224, 224, 25, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.float32, 2, 1, 56, 43, 8, 1, 3, 333, 188, 19, 1, 1, DataFormat["NHWC2NC1HWC0"].value),
        TloadParams(np.int8, 1, 3, 11, 109, 32, 1, 3, 111, 1023, 109, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 121, 9, 32, 1, 3, 65, 128, 127, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.float16, 1, 6, 10, 100, 16, 1, 1, 96, 100, 100, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.float16, 10, 16, 16, 2, 16, 1, 256, 255, 100, 16, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.float32, 1, 1, 56, 112, 8, 1, 2, 25, 224, 112, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.float32, 2, 1, 56, 43, 8, 1, 3, 19, 333, 188, 1, 1, DataFormat["NCHW2NC1HWC0"].value),
        TloadParams(np.int8, 75, 3, 16, 32, 48, 95, 5, 5, 50, 111, 5, 5, DataFormat["NCHW2FZ4D"].value),
        TloadParams(np.int8, 98, 4, 16, 32, 64, 58, 7, 7, 121, 127, 7, 7, DataFormat["NCHW2FZ4D"].value),
        TloadParams(np.float16, 63, 6, 16, 16, 96, 111, 3, 3, 220, 112, 3, 3, DataFormat["NCHW2FZ4D"].value),
        TloadParams(np.float16, 75, 4, 16, 16, 64, 48, 5, 5, 100, 50, 5, 5, DataFormat["NCHW2FZ4D"].value),
        TloadParams(np.float32, 50, 3, 16, 8, 48, 14, 5, 5, 224, 224, 5, 5, DataFormat["NCHW2FZ4D"].value),
        TloadParams(np.float32, 27, 2, 16, 8, 32, 24, 3, 3, 333, 188, 3, 3, DataFormat["NCHW2FZ4D"].value),
                         ## N D C1 H W  N C D H W
        TloadParams(np.int8, 1, 2, 3, 11, 109, 3, 111, 2, 1023, 109, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
        TloadParams(np.int8, 3, 3, 2, 15, 9, 3, 65, 4, 30, 50, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
        TloadParams(np.float16, 1, 4, 6, 10, 10, 1, 96, 6, 100, 10, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
        TloadParams(np.float16, 10, 2, 8, 16, 2, 256, 128, 2, 100, 7, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
        TloadParams(np.float32, 1, 5, 1, 25, 31, 2, 25, 7, 112, 31, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
        TloadParams(np.float32, 2, 2, 1, 43, 43, 3, 19, 2, 155, 180, 1, 1, DataFormat["NCDHW2NDC1HWC0"].value),
                         ## srcN srcC srcD, srcH srcW N C D H W C1DHW N/16
        TloadParams(np.int8, 48, 95, 2, 5, 5, 50, 111, 4, 5, 5, 150, 3, DataFormat["NCDHW2FZ3D"].value),
        TloadParams(np.int8, 32, 58, 2, 7, 7, 63, 127, 2, 7, 7, 196, 2, DataFormat["NCDHW2FZ3D"].value),
        TloadParams(np.float16, 48, 111, 2, 3, 3, 110, 112, 2, 3, 3, 126, 3, DataFormat["NCDHW2FZ3D"].value),
        TloadParams(np.float16, 32, 48, 3, 3, 3, 70, 50, 4, 3, 3, 81, 2, DataFormat["NCDHW2FZ3D"].value),
        TloadParams(np.float32, 48, 14, 5, 2, 2, 224, 224, 7, 2, 2, 40, 3, DataFormat["NCDHW2FZ3D"].value),
        TloadParams(np.float32, 32, 24, 2, 3, 3, 333, 188, 2, 3, 3, 54, 2, DataFormat["NCDHW2FZ3D"].value),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)