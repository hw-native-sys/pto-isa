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

from typing import Callable, List, Literal, Union


def do_bench(
    fn: Callable,
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "us",
) -> Union[float, List[float]]:
    """
    Benchmark a given function with warmup.

    Args:
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.
        aggregation: Aggregation mode for benchmark times.
        unit: Time unit of the benchmarks.
    Returns:
        Runtime, or list of runtimes, in specified units.
    """
    import torch
    import torch_npu

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    for _ in range(warmup_iters):
        fn()
    torch_npu.npu.synchronize()

    for i in range(benchmark_iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch_npu.npu.synchronize()
    factor = {"s": 1e-3, "ms": 1e0, "us": 1e3, "ns": 1e6}[unit]
    times = [
        factor * start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    if aggregation == "mean":
        return sum(times) / len(times)
    return times
