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
from enum import Enum
import numpy as np


class DataFormat(Enum):
    NCHW2NC1HWC0 = 1
    NC1HWC02NCHW = 2
    NC1HWC02C1HWN1N0C0 = 3
    GNCHW2GNC1HWC0 = 4
    GNC1HWC02C1HWN1N0C0 = 5
    NCDHW2C1DHWN1N0C0 = 6
    NCHW2C1HWN1N0C0 = 7
    NCHW2C1DHWN1N0C0 = 8


def pack_matrix_to_fp4(matrix):
    # Ensure the matrix is flattened to make pairing easy
    # We assume the total number of elements is even
    flat = matrix.flatten()
    
    # Extract A (even indices) and B (odd indices)
    a = flat[0::2] & 0x0F
    b = flat[1::2] & 0x0F
    
    # Pack: A in low nibble, B in high nibble
    packed = (a | (b << 4)).astype(np.uint8)
    return packed


def golden_NCHW2NC1HWC0(g_info):
    assert g_info.src_shape_0 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_2
    assert g_info.src_shape_3 == g_info.dst_shape_3

    n = g_info.src_shape_0
    c = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.dst_shape_4

    dtype_size = np.dtype(g_info.data_type).itemsize
    assert c0 == 32 // dtype_size

    c1 = g_info.dst_shape_1
    assert c1 == (c + c0 - 1) // c0
    
    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    c1 = (c + c0 - 1) // c0
    
    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_NC1HWC02NCHW(g_info):
    assert g_info.src_shape_0 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_2
    assert g_info.src_shape_3 == g_info.dst_shape_3

    n = g_info.src_shape_0
    c1 = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.src_shape_4

    dtype_size = np.dtype(g_info.data_type).itemsize
    assert c0 == 32 // dtype_size

    c = g_info.dst_shape_1
    assert c == c1 * c0
    
    input_arr = np.random.randint(1, 5, size=(n, c1, h, w, c0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
     
    output_arr = input_arr.transpose(0, 1, 4, 2, 3).reshape(n, c, h, w)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_NC1HWC02C1HWN1N0C0(g_info):
    assert g_info.src_shape_1 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_1
    assert g_info.src_shape_3 == g_info.dst_shape_2

    n = g_info.src_shape_0
    c1 = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.src_shape_4

    n1 = g_info.dst_shape_3
    n0 = g_info.dst_shape_4

    assert n1 == (n + n0 - 1) // n0
        
    input_arr = np.random.randint(1, 5, size=(n, c1, h, w, c0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    padded_c = n1 * n0
    padding = padded_c - n
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, padding), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(n1, n0, c1, h, w, c0).transpose(2, 3, 4, 0, 1, 5)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_GNCHW2NC1HWC0(g_info):
    assert g_info.src_shape_0 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_2
    assert g_info.src_shape_3 == g_info.dst_shape_3

    is_mx_type = "MXFP4" in g_info.case_name

    group_n = g_info.group_n
    n = g_info.src_shape_0
    c = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.dst_shape_4

    c1 = g_info.dst_shape_1
    
    input_arr = np.random.randint(1, 5, size=(group_n, n, c, h, w)).astype(g_info.data_type)
    if is_mx_type:
        input_file = pack_matrix_to_fp4(input_arr)
        input_file.tofile("./input.bin")
    else:
        input_arr.tofile("./input.bin") 
    
    c1 = (c + c0 - 1) // c0
    
    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(group_n, n, c1, c0, h, w).transpose(0, 1, 2, 4, 5, 3)
    
    if is_mx_type:
        output_file = pack_matrix_to_fp4(output_arr)
        output_file.tofile("./golden.bin")
    else:
        output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_GNC1HWC02C1HWN1N0C0(g_info):
    assert g_info.src_shape_1 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_1
    assert g_info.src_shape_3 == g_info.dst_shape_2

    group_n = g_info.group_n
    n = g_info.src_shape_0
    c1 = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.src_shape_4

    n1 = g_info.dst_shape_3
    n0 = g_info.dst_shape_4

    assert n1 == (n + n0 - 1) // n0
        
    input_arr = np.random.randint(1, 5, size=(group_n, n, c1, h, w, c0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    padded_c = n1 * n0
    padding = padded_c - n
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(group_n, n1, n0, c1, h, w, c0).transpose(0, 3, 4, 5, 1, 2, 6)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_NCDHW2C1DHWN1N0C0(g_info):
    n = g_info.src_shape_0
    c = g_info.src_shape_1
    d = g_info.src_shape_2
    h = g_info.src_shape_3
    w = g_info.src_shape_4

    c1dhw = g_info.dst_shape_0
    n1 = g_info.dst_shape_1
    n0 = g_info.dst_shape_2
    c0 = g_info.dst_shape_3

    c1 = c1dhw // (d*w*h)
    dtype_size = np.dtype(g_info.data_type).itemsize

    assert c0 == 32 // dtype_size
    assert n1 == (n + n0 - 1) // n0
    assert c1 == (c + c0 - 1) // c0
        
    input_arr = np.random.randint(1, 5, size=(n, c, d, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    padded_n = n1 * n0
    padding_n = padded_n - n
    padded_c = c1 * c0
    padding_c = padded_c - c
    if padding_n > 0 or padding_c > 0:
        input_arr = np.pad(input_arr, ((0, padding_n), (0, padding_c), (0, 0), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(n1, n0, c1, c0, d, h, w).transpose(4, 2, 5, 6, 0, 1, 3)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_NCHW2C1HWN1N0C0(g_info):
    # NCHW (N, C, H, W) -> FRACTAL_Z / FRACTAL_Z_3D (C1HW, N1, N0, C0)
    # For a 4D source, FRACTAL_Z_3D uses an implicit D = 1, so both share this layout.
    n = g_info.src_shape_0
    c = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3

    c1hw = g_info.dst_shape_0
    n1 = g_info.dst_shape_1
    n0 = g_info.dst_shape_2
    c0 = g_info.dst_shape_3

    c1 = c1hw // (h * w)
    dtype_size = np.dtype(g_info.data_type).itemsize

    assert c0 == 32 // dtype_size
    assert n1 == (n + n0 - 1) // n0
    assert c1 == (c + c0 - 1) // c0

    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    padded_c = c1 * c0
    padding_c = padded_c - c
    if padding_c > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding_c), (0, 0), (0, 0)), mode='constant')

    # NCHW -> NC1HWC0: (n, c, h, w) -> (n, c1, h, w, c0)
    nc1hwc0 = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)

    # NC1HWC0 -> FRACTAL_Z: pad N, then (n, c1, h, w, c0) -> (c1, h, w, n1, n0, c0)
    padded_n = n1 * n0
    padding_n = padded_n - n
    if padding_n > 0:
        nc1hwc0 = np.pad(nc1hwc0, ((0, padding_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')

    output_arr = nc1hwc0.reshape(n1, n0, c1, h, w, c0).transpose(2, 3, 4, 0, 1, 5)

    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")

    return input_arr, output_arr


def golden_NCHW2NC1HWC0_Valid(g_info):
    # Valid shape 用例：NCHW -> NC1HWC0
    #
    #   raw shape（声明的 ConvTileShape）只表达每一维的 stride 与容量，与格式转换无关；
    #   转换的全部尺寸由两侧各自的 valid shape 决定：
    #     src valid = (Nv, Cv, Hv, Wv)
    #     dst valid = (Nv, C1v, Hv, Wv, C0)，C1v = ceil(Cv / C0)
    #   其中 C0 = 32 / sizeof(T) 是格式常量：raw 的 C0 维可以大于它（那只是内存步长），
    #   而 valid 的 C0 必须等于它。
    #   写入范围 = dst valid 盒；盒内没有源对应的位置（末 C1 块的 C0 尾巴）补 0；
    #   盒外（raw 中超出 valid 的区域）保持哨兵 -12345 不变。
    n, c, h, w = g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3
    dn, dc1, dh, dw, dc0 = (
        g_info.dst_shape_0,
        g_info.dst_shape_1,
        g_info.dst_shape_2,
        g_info.dst_shape_3,
        g_info.dst_shape_4,
    )
    vn, vc, vh, vw = g_info.src_valid_shape
    dvn, dvc1, dvh, dvw, dvc0 = g_info.dst_valid_shape

    dtype_size = np.dtype(g_info.data_type).itemsize
    c0 = dvc0
    assert c0 == 32 // dtype_size, "valid 的 C0 必须等于格式常量 32 / sizeof(T)"

    # 格式转换关系：只涉及两侧 valid，与 raw 无关
    assert dvc1 == (vc + c0 - 1) // c0, "dst 的 C1 valid 必须是 ceil(Cv / C0)"
    assert (dvn, dvh, dvw) == (vn, vh, vw), "N/H/W 两侧 valid 必须一致"
    # raw 只负责"装得下"：容量必须覆盖 valid 盒
    assert 0 < vn <= n and 0 < vh <= h and 0 < vw <= w and 0 < vc <= c
    assert dvn <= dn and dvc1 <= dc1 and dvh <= dh and dvw <= dw and dvc0 <= dc0

    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    # 取 src valid 盒，在盒内按格式规则做 NCHW -> NC1HWC0：
    #   pad C 到 C1v*C0（补零即 valid C 之外的 C0 尾巴）-> reshape 分出 C0 -> transpose 成 [Nv, C1v, Hv, Wv, C0]
    src_valid = input_arr[:vn, :vc, :vh, :vw]
    pad_c = dvc1 * c0 - vc
    if pad_c > 0:
        src_valid = np.pad(src_valid, ((0, 0), (0, pad_c), (0, 0), (0, 0)), mode='constant')
    converted = src_valid.reshape(vn, dvc1, c0, vh, vw).transpose(0, 1, 3, 4, 2)

    # 输出按 raw 维分配并整体预置哨兵；只把 dst valid 盒写进去，盒外保持哨兵
    sentinel = np.array(-12345, dtype=g_info.data_type)
    output_arr = np.full((dn, dc1, dh, dw, dc0), sentinel, dtype=g_info.data_type)
    output_arr[:dvn, :dvc1, :dvh, :dvw, :dvc0] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_NC1HWC02NCHW_Valid(g_info):
    # 场景：NC1HWC0 -> NCHW。raw 只表达 stride/容量，转换域由两侧 valid 决定；盒外保持哨兵
    sn, sc1, sh, sw, sc0 = (
        g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3, g_info.src_shape_4,
    )
    dn, dc, dh, dw = g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3
    vn, vc1, vh, vw, vc0 = g_info.src_valid_shape
    dvn, dvc, dvh, dvw = g_info.dst_valid_shape

    assert vc0 == 32 // np.dtype(g_info.data_type).itemsize, "C0 是格式常量"
    assert (dvn, dvc, dvh, dvw) == (vn, vc1 * vc0, vh, vw), "dst valid 必须是 src valid 的像"
    assert 0 < vn <= sn and 0 < vc1 <= sc1 and 0 < vh <= sh and 0 < vw <= sw and 0 < vc0 <= sc0
    assert dvn <= dn and dvc <= dc and dvh <= dh and dvw <= dw

    input_arr = np.random.randint(1, 5, size=(sn, sc1, sh, sw, sc0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vn, :vc1, :vh, :vw, :vc0]
    converted = sv.transpose(0, 1, 4, 2, 3).reshape(vn, vc1 * vc0, vh, vw)

    output_arr = np.full((dn, dc, dh, dw), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type)
    output_arr[:dvn, :dvc, :dvh, :dvw] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_NC1HWC02C1HWN1N0C0_Valid(g_info):
    # 场景：NC1HWC0 -> FRACTAL_Z。扁平维按 valid 的 H/W 分解，写入范围是 [0, C1v*Hv*Wv) 连续前缀
    sn, sc1, sh, sw, sc0 = (
        g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3, g_info.src_shape_4,
    )
    dd0, dn1, dn0, dc0 = (
        g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3,
    )
    vn, vc1, vh, vw, vc0 = g_info.src_valid_shape
    d0, dn1v, dn0v, dc0v = g_info.dst_valid_shape

    assert vc0 == 32 // np.dtype(g_info.data_type).itemsize and dc0v == vc0, "C0 是格式常量"
    assert dn0v == 16, "N0 是格式常量"
    vn1 = (vn + dn0v - 1) // dn0v
    assert (d0, dn1v, dn0v, dc0v) == (vc1 * vh * vw, vn1, dn0v, vc0), "dst valid 必须是 src valid 的像"
    assert 0 < vn <= sn and 0 < vc1 <= sc1 and 0 < vh <= sh and 0 < vw <= sw and 0 < vc0 <= sc0
    assert d0 <= dd0 and dn1v <= dn1 and dn0v <= dn0 and dc0v <= dc0

    input_arr = np.random.randint(1, 5, size=(sn, sc1, sh, sw, sc0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vn, :vc1, :vh, :vw, :vc0]
    pad_n = vn1 * dn0v - vn
    if pad_n > 0:
        sv = np.pad(sv, ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
    converted = sv.reshape(vn1, dn0v, vc1, vh, vw, vc0).transpose(2, 3, 4, 0, 1, 5).reshape(
        vc1 * vh * vw, vn1, dn0v, vc0
    )

    output_arr = np.full((dd0, dn1, dn0, dc0), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type)
    output_arr[:d0, :dn1v, :dn0v, :dc0v] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_GNCHW2GNC1HWC0_Valid(g_info):
    # 场景：GNCHW -> GNC1HWC0（G 由 valid 元组首维给出）
    sg = g_info.group_n  # GNCHW 的 G 沿用既有注册约定（group_n）
    sn, sc, sh, sw = g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3
    dg = g_info.group_n  # G 落在 group_n；dst 形状槽位是 G 之后的 (N, C1, H, W, C0)
    dn, dc1, dh, dw, dc0 = (
        g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3, g_info.dst_shape_4,
    )
    vg, vn, vc, vh, vw = g_info.src_valid_shape
    dgv, dnv, dc1v, dhv, dwv, dc0v = g_info.dst_valid_shape

    assert dc0v == 32 // np.dtype(g_info.data_type).itemsize, "C0 是格式常量"
    assert (dgv, dnv, dc1v, dhv, dwv, dc0v) == (vg, vn, (vc + dc0v - 1) // dc0v, vh, vw, dc0v)
    assert 0 < vg <= sg and 0 < vn <= sn and 0 < vc <= sc and 0 < vh <= sh and 0 < vw <= sw
    assert dgv <= dg and dnv <= dn and dc1v <= dc1 and dhv <= dh and dwv <= dw and dc0v <= dc0

    input_arr = np.random.randint(1, 5, size=(sg, sn, sc, sh, sw)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vg, :vn, :vc, :vh, :vw]
    pad_c = dc1v * dc0v - vc
    if pad_c > 0:
        sv = np.pad(sv, ((0, 0), (0, 0), (0, pad_c), (0, 0), (0, 0)), mode='constant')
    converted = sv.reshape(vg, vn, dc1v, dc0v, vh, vw).transpose(0, 1, 2, 4, 5, 3)

    output_arr = np.full(
        (dg, dn, dc1, dh, dw, dc0), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type
    )
    output_arr[:dgv, :dnv, :dc1v, :dhv, :dwv, :dc0v] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_GNC1HWC02C1HWN1N0C0_Valid(g_info):
    # 场景：GNC1HWC0 -> FRACTAL_Z（raw G 取 group_n，valid G 取 src_valid_shape 首维）
    sg = g_info.group_n
    sn, sc1, sh, sw, sc0 = (
        g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3, g_info.src_shape_4,
    )
    dd0, dn1, dn0, dc0 = (
        g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3,
    )
    vg, vn, vc1, vh, vw, vc0 = g_info.src_valid_shape
    d0, dn1v, dn0v, dc0v = g_info.dst_valid_shape

    assert vc0 == 32 // np.dtype(g_info.data_type).itemsize and dc0v == vc0, "C0 是格式常量"
    assert dn0v == 16, "N0 是格式常量"
    vn1 = (vn + dn0v - 1) // dn0v
    assert (d0, dn1v, dn0v, dc0v) == (vg * vc1 * vh * vw, vn1, dn0v, vc0)
    assert 0 < vn <= sn and 0 < vc1 <= sc1 and 0 < vh <= sh and 0 < vw <= sw and 0 < vc0 <= sc0
    assert d0 <= dd0 and dn1v <= dn1 and dn0v <= dn0 and dc0v <= dc0

    input_arr = np.random.randint(1, 5, size=(sg, sn, sc1, sh, sw, sc0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vg, :vn, :vc1, :vh, :vw, :vc0]
    pad_n = vn1 * dn0v - vn
    if pad_n > 0:
        sv = np.pad(sv, ((0, 0), (0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
    converted = sv.reshape(vg, vn1, dn0v, vc1, vh, vw, vc0).transpose(0, 3, 4, 5, 1, 2, 6).reshape(
        vg * vc1 * vh * vw, vn1, dn0v, vc0
    )

    output_arr = np.full((dd0, dn1, dn0, dc0), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type)
    output_arr[:d0, :dn1v, :dn0v, :dc0v] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_NCDHW2C1DHWN1N0C0_Valid(g_info):
    # 场景：NCDHW -> FRACTAL_Z_3D。扁平维序为 (d, c1, h, w)，均按 valid 分解
    sn, sc, sd, sh, sw = (
        g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3, g_info.src_shape_4,
    )
    dd0, dn1, dn0, dc0 = (
        g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3,
    )
    vn, vc, vd, vh, vw = g_info.src_valid_shape
    d0, dn1v, dn0v, dc0v = g_info.dst_valid_shape

    assert dc0v == 32 // np.dtype(g_info.data_type).itemsize, "C0 是格式常量"
    assert dn0v == 16, "N0 是格式常量"
    vc1 = (vc + dc0v - 1) // dc0v
    vn1 = (vn + dn0v - 1) // dn0v
    assert (d0, dn1v, dn0v, dc0v) == (vd * vc1 * vh * vw, vn1, dn0v, dc0v)
    assert 0 < vn <= sn and 0 < vc <= sc and 0 < vd <= sd and 0 < vh <= sh and 0 < vw <= sw
    assert d0 <= dd0 and dn1v <= dn1 and dn0v <= dn0 and dc0v <= dc0

    input_arr = np.random.randint(1, 5, size=(sn, sc, sd, sh, sw)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vn, :vc, :vd, :vh, :vw]
    pad_c = vc1 * dc0v - vc
    if pad_c > 0:
        sv = np.pad(sv, ((0, 0), (0, pad_c), (0, 0), (0, 0), (0, 0)), mode='constant')
    sv = sv.reshape(vn, vc1, dc0v, vd, vh, vw)
    pad_n = vn1 * dn0v - vn
    if pad_n > 0:
        sv = np.pad(sv, ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
    converted = sv.reshape(vn1, dn0v, vc1, dc0v, vd, vh, vw).transpose(4, 2, 5, 6, 0, 1, 3).reshape(
        vd * vc1 * vh * vw, vn1, dn0v, dc0v
    )

    output_arr = np.full((dd0, dn1, dn0, dc0), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type)
    output_arr[:d0, :dn1v, :dn0v, :dc0v] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def golden_NCHW2C1HWN1N0C0_Valid(g_info):
    # 场景：NCHW -> FRACTAL_Z / FRACTAL_Z_3D（4D 源的 3D 情形等价于隐式 D = 1，共用本函数）
    sn, sc, sh, sw = g_info.src_shape_0, g_info.src_shape_1, g_info.src_shape_2, g_info.src_shape_3
    dd0, dn1, dn0, dc0 = (
        g_info.dst_shape_0, g_info.dst_shape_1, g_info.dst_shape_2, g_info.dst_shape_3,
    )
    vn, vc, vh, vw = g_info.src_valid_shape
    d0, dn1v, dn0v, dc0v = g_info.dst_valid_shape

    assert dc0v == 32 // np.dtype(g_info.data_type).itemsize, "C0 是格式常量"
    assert dn0v == 16, "N0 是格式常量"
    vc1 = (vc + dc0v - 1) // dc0v
    vn1 = (vn + dn0v - 1) // dn0v
    assert (d0, dn1v, dn0v, dc0v) == (vc1 * vh * vw, vn1, dn0v, dc0v)
    assert 0 < vn <= sn and 0 < vc <= sc and 0 < vh <= sh and 0 < vw <= sw
    assert d0 <= dd0 and dn1v <= dn1 and dn0v <= dn0 and dc0v <= dc0

    input_arr = np.random.randint(1, 5, size=(sn, sc, sh, sw)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")

    sv = input_arr[:vn, :vc, :vh, :vw]
    pad_c = vc1 * dc0v - vc
    if pad_c > 0:
        sv = np.pad(sv, ((0, 0), (0, pad_c), (0, 0), (0, 0)), mode='constant')
    sv = sv.reshape(vn, vc1, dc0v, vh, vw)
    pad_n = vn1 * dn0v - vn
    if pad_n > 0:
        sv = np.pad(sv, ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
    converted = sv.reshape(vn1, dn0v, vc1, dc0v, vh, vw).transpose(2, 4, 5, 0, 1, 3).reshape(
        vc1 * vh * vw, vn1, dn0v, dc0v
    )

    output_arr = np.full((dd0, dn1, dn0, dc0), np.array(-12345, dtype=g_info.data_type), dtype=g_info.data_type)
    output_arr[:d0, :dn1v, :dn0v, :dc0v] = converted
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape} (src valid = {g_info.src_valid_shape}, dst valid = {g_info.dst_valid_shape})")


def gen_golden_data(g_info):
    """
    Generates aligned runtime raw binaries for C++ unit test validation suites.
    """
    data_type = g_info.data_type
    mode = g_info.shape

    # -------------------------------------------------------------
    # MODE 1: NCHW -> NC1HWC0
    # -------------------------------------------------------------
    if mode == DataFormat.NCHW2NC1HWC0.value:
        if g_info.src_valid_shape is not None:
            golden_NCHW2NC1HWC0_Valid(g_info)
        else:
            golden_NCHW2NC1HWC0(g_info)

    # -------------------------------------------------------------
    # MODE 2: NC1HWC0 -> NCHW
    # -------------------------------------------------------------
    elif mode == DataFormat.NC1HWC02NCHW.value:
        if g_info.src_valid_shape is not None:
            golden_NC1HWC02NCHW_Valid(g_info)
        else:
            golden_NC1HWC02NCHW(g_info)
    # -------------------------------------------------------------
    # MODE 3: NC1HWC0 -> C1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.NC1HWC02C1HWN1N0C0.value:
        if g_info.src_valid_shape is not None:
            golden_NC1HWC02C1HWN1N0C0_Valid(g_info)
        else:
            golden_NC1HWC02C1HWN1N0C0(g_info)

    # -------------------------------------------------------------
    # MODE 4: GNCHW -> GNC1HWC0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNCHW2GNC1HWC0.value:
        if g_info.src_valid_shape is not None:
            golden_GNCHW2GNC1HWC0_Valid(g_info)
        else:
            golden_GNCHW2NC1HWC0(g_info)

    # -------------------------------------------------------------
    # MODE 5: GNC1HWC0 -> GC1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNC1HWC02C1HWN1N0C0.value:
        if g_info.src_valid_shape is not None:
            golden_GNC1HWC02C1HWN1N0C0_Valid(g_info)
        else:
            golden_GNC1HWC02C1HWN1N0C0(g_info)
    # -------------------------------------------------------------
    # MODE 6: NCDHW -> C1DHWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.NCDHW2C1DHWN1N0C0.value:
        if g_info.src_valid_shape is not None:
            golden_NCDHW2C1DHWN1N0C0_Valid(g_info)
        else:
            golden_NCDHW2C1DHWN1N0C0(g_info)

    # -------------------------------------------------------------
    # MODE 7: NCHW -> C1HWN1N0C0 (FRACTAL_Z)
    # -------------------------------------------------------------
    elif mode == DataFormat.NCHW2C1HWN1N0C0.value:
        if g_info.src_valid_shape is not None:
            golden_NCHW2C1HWN1N0C0_Valid(g_info)
        else:
            golden_NCHW2C1HWN1N0C0(g_info)

    # -------------------------------------------------------------
    # MODE 8: NCHW -> C1HWN1N0C0 (FRACTAL_Z_3D, implicit D = 1)
    # -------------------------------------------------------------
    elif mode == DataFormat.NCHW2C1DHWN1N0C0.value:
        if g_info.src_valid_shape is not None:
            golden_NCHW2C1HWN1N0C0_Valid(g_info)
        else:
            golden_NCHW2C1HWN1N0C0(g_info)

    else:
        pass


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        src_shape_0,
        src_shape_1,
        src_shape_2,
        src_shape_3,
        src_shape_4,
        dst_shape_0,
        dst_shape_1,
        dst_shape_2,
        dst_shape_3,
        dst_shape_4,
        dst_shape_5=1,
        group_n=1,
        src_valid_shape=None,
        dst_valid_shape=None
    ):
        self.case_name = case_name
        self.data_type = data_type
        self.shape = shape
        self.src_shape_0 = src_shape_0
        self.src_shape_1 = src_shape_1
        self.src_shape_2 = src_shape_2
        self.src_shape_3 = src_shape_3
        self.src_shape_4 = src_shape_4
        self.dst_shape_0 = dst_shape_0
        self.dst_shape_1 = dst_shape_1
        self.dst_shape_2 = dst_shape_2
        self.dst_shape_3 = dst_shape_3
        self.dst_shape_4 = dst_shape_4
        self.dst_shape_5 = dst_shape_5
        self.group_n = group_n
        self.src_valid_shape = src_valid_shape  # None = 未使用 valid（旧用例）
        self.dst_valid_shape = dst_valid_shape


test_cases_registry = [
    TTRANSParams("NCHW2NC1HWC0_1", np.float32, DataFormat.NCHW2NC1HWC0.value, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8),
    TTRANSParams("NCHW2NC1HWC0_2", np.int32, DataFormat.NCHW2NC1HWC0.value, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8),
    TTRANSParams("NCHW2NC1HWC0_3", np.uint16, DataFormat.NCHW2NC1HWC0.value, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16),
    TTRANSParams("NCHW2NC1HWC0_4", np.int32, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8),
    TTRANSParams("NCHW2NC1HWC0_5", np.int8, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32),

    TTRANSParams("NCHW2NC1HWC0_MX_e8m0", np.uint8, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32),

    TTRANSParams("NC1HWC02NCHW_1", np.float32, DataFormat.NC1HWC02NCHW.value, 5, 3, 3, 4, 8, 5, 24, 3, 4, 1),
    TTRANSParams("NC1HWC02NCHW_2", np.int32, DataFormat.NC1HWC02NCHW.value, 5, 2, 4, 5, 8, 5, 16, 4, 5, 1),

    TTRANSParams("NC1HWC02C1HWN1N0C0_1", np.float32, DataFormat.NC1HWC02C1HWN1N0C0.value, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_2", np.int32, DataFormat.NC1HWC02C1HWN1N0C0.value, 15, 2, 3, 16, 8, 2, 3, 16, 2, 8, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_3", np.uint16, DataFormat.NC1HWC02C1HWN1N0C0.value, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16),
    TTRANSParams("NC1HWC02C1HWN1N0C0_4", np.int32, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_5", np.int8, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32),

    TTRANSParams("NC1HWC02C1HWN1N0C0_MX_e4m3", np.uint8, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32),

    TTRANSParams("GNCHW2GNC1HWC0_1", np.float32, DataFormat.GNCHW2GNC1HWC0.value, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 4),
    TTRANSParams("GNCHW2GNC1HWC0_2", np.int32, DataFormat.GNCHW2GNC1HWC0.value, 5, 14, 13, 8, 1, 5, 2, 13, 8, 8, 1, 2),
    TTRANSParams("GNCHW2GNC1HWC0_3", np.uint16, DataFormat.GNCHW2GNC1HWC0.value, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 3),
    TTRANSParams("GNCHW2GNC1HWC0_4", np.int32, DataFormat.GNCHW2GNC1HWC0.value, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1),
    TTRANSParams("GNCHW2GNC1HWC0_5", np.int8, DataFormat.GNCHW2GNC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 3),

    TTRANSParams("GNCHW2GNC1HWC0_MXFP4_e2m1", np.uint8, DataFormat.GNCHW2GNC1HWC0.value, 4, 64, 3, 14, 1, 4, 1, 3, 14, 64, 1, 1),

    TTRANSParams("GNC1HWC02C1HWN1N0C0_1", np.float32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 25, 4, 3, 4, 8, 4, 3, 4, 2, 16, 8, 2),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_2", np.int32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 15, 2, 3, 4, 8, 2, 3, 4, 2, 8, 8, 3),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_3", np.uint16, DataFormat.GNC1HWC02C1HWN1N0C0.value, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 2),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_4", np.int32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 4, 8, 3, 7, 8, 8, 3, 7, 1, 4, 8, 3),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_5", np.int8, DataFormat.GNC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1),

    TTRANSParams("NCDHW2C1DHWN1N0C0_1", np.float32, DataFormat.NCDHW2C1DHWN1N0C0.value, 5, 3, 3, 4, 8, 96, 1, 16, 8, 1),
    TTRANSParams("NCDHW2C1DHWN1N0C0_2", np.int32, DataFormat.NCDHW2C1DHWN1N0C0.value, 18, 2, 4, 5, 3, 60, 2, 16, 8, 1),

    TTRANSParams("NCHW2C1HWN1N0C0_1", np.float32, DataFormat.NCHW2C1HWN1N0C0.value, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1),
    TTRANSParams("NCHW2C1HWN1N0C0_2", np.int32, DataFormat.NCHW2C1HWN1N0C0.value, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1),
    TTRANSParams("NCHW2C1HWN1N0C0_3", np.int8, DataFormat.NCHW2C1HWN1N0C0.value, 4, 8, 3, 8, 1, 24, 1, 32, 32, 1),

    TTRANSParams("NCHW2C1DHWN1N0C0_1", np.float32, DataFormat.NCHW2C1DHWN1N0C0.value, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1),
    TTRANSParams("NCHW2C1DHWN1N0C0_2", np.int32, DataFormat.NCHW2C1DHWN1N0C0.value, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1),

    # -------------------------------------------------------------------------------------
    # Valid shape 用例：NCHW -> NC1HWC0，dst 侧的每一维都收缩（raw = 2 × valid）
    #   raw:   src NCHW (N=1, C=16, H=8, W=16) / dst NC1HWC0 (N=2, C1=4, H=8, W=32, C0=16)
    #   valid: src (1, 16, 4, 16) / dst (1, 2, 4, 16, 8)
    #   - dst raw 的 C0=16 只是内存步长（2×）；valid 的 C0=8 才是格式常量 32/sizeof(float)
    #   - C1v = ceil(Cv / C0) = 2，与 raw 的 C1 = 4 无关
    #   - Cv = 16 = C1v * C0，本例无格式补零
    # -------------------------------------------------------------------------------------
    TTRANSParams(
        "valid_NCHW2NC1HWC0_dst_all_shrunk", np.float32, DataFormat.NCHW2NC1HWC0.value,
        1, 16, 8, 16, 1, 2, 4, 8, 32, 16,
        src_valid_shape=(1, 16, 4, 16), dst_valid_shape=(1, 2, 4, 16, 8)
    ),

    # --- Valid shape：其余 7 个场景，src 与 dst 的 raw/valid 每一维都不同 ---
    TTRANSParams(
        "valid_NC1HWC02NCHW_all_dims_differ", np.float32, DataFormat.NC1HWC02NCHW.value,
        3, 5, 7, 9, 16, 3, 32, 7, 9, 1, 1, 1,
        src_valid_shape=(2, 3, 4, 5, 8), dst_valid_shape=(2, 24, 4, 5)
    ),
    TTRANSParams(
        "valid_NC1HWC02C1HWN1N0C0_all_dims_differ", np.float32, DataFormat.NC1HWC02C1HWN1N0C0.value,
        6, 3, 4, 5, 16, 16, 2, 24, 16, 1, 1, 1,
        src_valid_shape=(5, 2, 2, 3, 8), dst_valid_shape=(12, 1, 16, 8)
    ),
    TTRANSParams(
        "valid_GNCHW2GNC1HWC0_all_dims_differ", np.float32, DataFormat.GNCHW2GNC1HWC0.value,
        4, 10, 5, 6, 1, 4, 3, 5, 6, 16, 1, 3,
        src_valid_shape=(2, 3, 8, 4, 5), dst_valid_shape=(2, 3, 1, 4, 5, 8)
    ),
    TTRANSParams(
        "valid_GNC1HWC02C1HWN1N0C0_all_dims_differ", np.float32, DataFormat.GNC1HWC02C1HWN1N0C0.value,
        4, 3, 4, 5, 16, 28, 2, 24, 16, 1, 1, 3,
        src_valid_shape=(2, 3, 1, 3, 4, 8), dst_valid_shape=(24, 1, 16, 8)
    ),
    TTRANSParams(
        "valid_NCDHW2C1DHWN1N0C0_all_dims_differ", np.float32, DataFormat.NCDHW2C1DHWN1N0C0.value,
        3, 10, 3, 4, 5, 32, 2, 24, 16, 1, 1, 1,
        src_valid_shape=(2, 8, 2, 3, 4), dst_valid_shape=(24, 1, 16, 8)
    ),
    TTRANSParams(
        "valid_NCHW2C1HWN1N0C0_all_dims_differ", np.float32, DataFormat.NCHW2C1HWN1N0C0.value,
        8, 10, 4, 6, 1, 16, 2, 24, 16, 1, 1, 1,
        src_valid_shape=(6, 8, 3, 4), dst_valid_shape=(12, 1, 16, 8)
    ),
    TTRANSParams(
        "valid_NCHW2C1DHWN1N0C0_all_dims_differ", np.float32, DataFormat.NCHW2C1DHWN1N0C0.value,
        6, 18, 6, 5, 1, 32, 2, 24, 16, 1, 1, 1,
        src_valid_shape=(5, 16, 4, 3), dst_valid_shape=(24, 1, 16, 8)
    )
]

SUITE_NAME = "TTRANSConvTest"

if __name__ == "__main__":
    print(f"Beginning validation matrix deployment for {len(test_cases_registry)} test scenarios...\n")
    
    for case in test_cases_registry:
        dirname = f"{SUITE_NAME}.{case.case_name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        original_dir = os.getcwd()
        os.chdir(dirname)
        print(f"Running Transformation Mode [{case.shape}] | Configuration Identifier: {case.case_name}")
        gen_golden_data(case)
        os.chdir(original_dir)
        
    print("\nAll binary test files (input.bin, golden.bin) have been generated successfully.")