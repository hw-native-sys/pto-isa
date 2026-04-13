#!/usr/bin/env python3
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


class NumExt:
    bf16 = "bfloat16"

    @staticmethod
    def is_bf16(dtype: object) -> bool:
        if dtype == NumExt.bf16:
            return True
        name = getattr(dtype, "__name__", None)
        if name == NumExt.bf16:
            return True
        return str(dtype) == NumExt.bf16

    @staticmethod
    def astype(values: np.ndarray, dtype: object) -> np.ndarray:
        if NumExt.is_bf16(dtype):
            return NumExt._bfloat16_bits_to_float32(NumExt._float32_to_bfloat16_bits(values))
        return np.asarray(values).astype(dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int], dtype: object) -> np.ndarray:
        if NumExt.is_bf16(dtype):
            return np.zeros(shape, dtype=np.float32)
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def write_array(path: str | os.PathLike[str], values: np.ndarray, dtype: object) -> None:
        if NumExt.is_bf16(dtype):
            NumExt._float32_to_bfloat16_bits(values).tofile(path)
        else:
            np.asarray(values).astype(dtype).tofile(path)
        dtype_str = NumExt.get_short_type_name(dtype)

    @staticmethod
    def _float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float32)
        bits = data.view(np.uint32)
        lsb = (bits >> 16) & np.uint32(1)
        rounded = bits + np.uint32(0x7FFF) + lsb
        return (rounded >> 16).astype(np.uint16)

    @staticmethod
    def _bfloat16_bits_to_float32(values: np.ndarray) -> np.ndarray:
        bits = np.asarray(values, dtype=np.uint16).astype(
            np.uint32) << np.uint32(16)
        return bits.view(np.float32)

    @staticmethod
    def get_short_type_name(dtype: object):
        if NumExt.is_bf16(dtype):
            return "bf16"
        else:
            return {np.float32: 'float',
                    np.float16: 'half',
                    np.int8: 'int8',
                    np.int16: 'int16',
                    np.int32: 'int32',
                    np.uint8: 'uint8',
                    np.uint16: 'uint16',
                    np.uint32: 'uint32'
                    }[dtype]
