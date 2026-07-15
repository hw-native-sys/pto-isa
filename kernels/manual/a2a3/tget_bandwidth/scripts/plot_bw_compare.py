#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
Plot TGET vs TGET_ASYNC bandwidth trends.

Host-side bandwidth comes from the latest benchmark output.
Device-side bandwidth follows the repo convention used in flash attention:
  us = cycles * 20 / 1000
which implies:
  GB/s = bytes / us / 1e3
"""

from pathlib import Path

import matplotlib.pyplot as plt


BYTES = [4096, 16384, 65536, 262144, 1048576, 4194304]
SIZE_LABELS = ["4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]

HOST_BW = {
    "TGET": [0.25, 0.89, 1.98, 3.28, 3.81, 4.02],
    "TGET_ASYNC": [0.22, 1.03, 3.24, 7.35, 11.88, 13.92],
}

DEVICE_AVG_CYCLES = {
    "TGET": [50.54, 200.72, 786.95, 3363.55, 12776.61, 53561.08],
    "TGET_ASYNC": [150.03, 198.85, 352.52, 1015.10, 3666.07, 13303.60],
}


def cycles_to_bandwidth_gbps(nbytes: int, avg_cycles: float) -> float:
    avg_us = avg_cycles * 20.0 / 1000.0
    return nbytes / avg_us / 1e3


DEVICE_BW = {
    name: [cycles_to_bandwidth_gbps(nbytes, cycles) for nbytes, cycles in zip(BYTES, values)]
    for name, values in DEVICE_AVG_CYCLES.items()
}


def plot_subplot(ax, title, series):
    colors = {"TGET": "#1f77b4", "TGET_ASYNC": "#d62728"}
    markers = {"TGET": "o", "TGET_ASYNC": "s"}
    for name, values in series.items():
        ax.plot(
            SIZE_LABELS,
            values,
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=2.2,
            markersize=7,
        )
    ax.set_title(title)
    ax.set_xlabel("Transfer Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()


def main():
    output_path = Path(__file__).resolve().parent.parent / "tget_bw_compare.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    plot_subplot(axes[0], "Host-side Measured Bandwidth", HOST_BW)
    plot_subplot(axes[1], "Device-side Bandwidth (20ns/tick)", DEVICE_BW)

    fig.suptitle("TGET vs TGET_ASYNC Bandwidth Trend on A2/A3", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"saved plot to: {output_path}")


if __name__ == "__main__":
    main()
