#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import ctypes
import os
import subprocess
from pathlib import Path

import torch

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/cann-9.0.0")
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", str(Path(__file__).resolve().parents[3]))


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _npu_arch_flag() -> str:
    return os.environ.get("NPU_ARCH", "dav-2201").strip()


def compile_paged_attention(kernel_cpp: str, verbose: bool = False, timeout: int = 300) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "pa_highperf_jit.so")
    manual_dir = os.path.join(PTO_LIB_PATH, "kernels/manual/a2a3/paged_attention_highperf")
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={_npu_arch_flag()}",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "-Wno-macro-redefined",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{manual_dir}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    cmd = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)
    return lib_path


def load_paged_attention_lib(lib_path: str, check_type: bool = True):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # block table
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # s
            ctypes.c_void_p,  # p
            ctypes.c_void_p,  # o_tmp
            ctypes.c_void_p,  # go
            ctypes.c_void_p,  # o_core_tmp
            ctypes.c_void_p,  # l
            ctypes.c_void_p,  # gm_k16
            ctypes.c_void_p,  # gm_v16
            ctypes.c_void_p,  # tiling
            ctypes.c_void_p,  # null
            ctypes.c_uint32,  # block dim
        ]
        lib.call_kernel.restype = None

    workspace = {}
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def _alloc(device, workspace_sizes, tiling):
        tiling_cpu = tuple(int(x) for x in tiling.detach().cpu().tolist())
        sizes_key = tuple(sorted((name, int(size)) for name, size in workspace_sizes.items()))
        key = (str(device), sizes_key, tiling_cpu)
        if workspace.get("key") == key:
            return
        workspace.clear()
        workspace["key"] = key
        for name, size in workspace_sizes.items():
            workspace[name] = torch.empty((int(size),), device=device, dtype=torch.uint8)
        workspace["null"] = torch.zeros((1,), device=device, dtype=torch.uint8)
        workspace["tiling"] = tiling.to(device=device, dtype=torch.int32)

    def paged_attention(q, k, v, block_table, workspace_sizes, tiling, stream_ptr=default_stream_ptr, block_dim: int = 24):
        _alloc(q.device, workspace_sizes, tiling)
        out = torch.empty_like(q)
        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(block_table),
            torch_to_ctypes(out),
            torch_to_ctypes(workspace["s"]),
            torch_to_ctypes(workspace["p"]),
            torch_to_ctypes(workspace["o_tmp"]),
            torch_to_ctypes(workspace["go"]),
            torch_to_ctypes(workspace["o_core_tmp"]),
            torch_to_ctypes(workspace["l"]),
            torch_to_ctypes(workspace["k16"]),
            torch_to_ctypes(workspace["v16"]),
            torch_to_ctypes(workspace["tiling"]),
            torch_to_ctypes(workspace["null"]),
            block_dim,
        )
        return out

    return paged_attention


def jit_compile_paged_attention(verbose: bool = False, clean_up: bool = True, kernel_cpp: str = "pa_kernel.cpp"):
    lib_path = compile_paged_attention(kernel_cpp, verbose=verbose)
    fn = load_paged_attention_lib(lib_path)
    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass
    return fn
