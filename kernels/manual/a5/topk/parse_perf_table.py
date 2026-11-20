# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------


# Parse camodel perf logs (perf/<case>/) and print rvec_busy_cycle per testcase.
# Source: perf/<case>/core*_summary_log
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

# Wired gtest cases (same order as collect_perf.sh / main.cpp).
CASES = [
    "iter_1x128_k3",
    "iter_1x256_k8",
    "iter_1x256_k12",
    "iter_1x288_k8",
    "iter_1x384_k6",
    "iter_1x512_k12",
    "iter_1x896_k16",
    "radix_1x4096_k512",
    "radix_1x16384_k256",
    "radix_1x16384_k2048",
]

METRIC_RE = re.compile(r"^(\S+)\s*\|\s*(\d+)\s*$")
DONE_RE = re.compile(r"core: (\d+), subcore: (\d+)")


def parse_rvec(f: Path):
    """Return max rvec_busy_cycle across sections in one core*_summary_log."""
    best = None
    cur = 0
    saw = False
    for line in f.read_text().splitlines():
        s = line.strip()
        m = METRIC_RE.match(s)
        if m:
            name, val = m.group(1), int(m.group(2))
            if "rvec_" in name and name.endswith("busy_cycle"):
                cur += val
                saw = True
            continue
        if DONE_RE.search(s) and saw:
            best = cur if best is None else max(best, cur)
            cur = 0
            saw = False
    if saw:
        best = cur if best is None else max(best, cur)
    return best


def collect_rvec(case: str):
    cdir = ROOT / "perf" / case
    if not cdir.exists():
        return None
    vals = []
    for f in sorted(cdir.glob("core*_summary_log")):
        v = parse_rvec(f)
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def fmt(v):
    return f"{v:,}" if v is not None else "n/a"


def main():
    data = {}
    print("## TopK rvec cycles (A5 sim)")
    print()
    print("| Case | rvec cycles |")
    print("|------|------------:|")
    for case in CASES:
        rvec = collect_rvec(case)
        data[case] = {"rvec_busy_cycle": rvec}
        print(f"| `{case}` | {fmt(rvec)} |")
    print()

    out = ROOT / "perf" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
