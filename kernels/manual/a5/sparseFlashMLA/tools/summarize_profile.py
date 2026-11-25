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

"""Summarize one benchmark process from an msprof op_summary CSV."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import statistics


def summarize(path, warmup, samples, kernel_name):
    if warmup < 0 or samples <= 0:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    launches = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Op Name", "Task Start Time(us)", "Task Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        for row in reader:
            if any(row.get(name) is None for name in required):
                raise ValueError(f"CSV row {reader.line_num} is missing required cells")
            if kernel_name not in row["Op Name"]:
                continue
            start = Decimal(row["Task Start Time(us)"])
            duration = float(row["Task Duration(us)"])
            if not start.is_finite() or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Kernel timestamps must be finite and durations must be positive")
            launches.append((start, duration))
    if len(launches) != warmup + samples:
        raise ValueError(
            f"Expected {warmup + samples} matching launches, found {len(launches)}; "
            "select one process/device CSV and match the benchmark's warmup and samples"
        )
    launches.sort(key=lambda item: item[0])
    durations = [duration for _, duration in launches[warmup:]]
    return {
        "csv": str(path.resolve()),
        "kernel_name_filter": kernel_name,
        "warmup": warmup,
        "samples": samples,
        "durations_us": durations,
        "mean_us": statistics.mean(durations),
        "median_us": statistics.median(durations),
        "min_us": min(durations),
        "max_us": max(durations),
        "timing_source": "msprof Task Duration(us); correctness must be checked in the benchmark log",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="One op_summary CSV containing all benchmark launches")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--kernel-name", default="flashMlaBf16Kernel", help="Substring matched against Op Name")
    args = parser.parse_args()
    if not args.kernel_name:
        parser.error("--kernel-name must not be empty")
    try:
        result = summarize(args.csv, args.warmup, args.samples, args.kernel_name)
    except (OSError, ValueError, InvalidOperation) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
