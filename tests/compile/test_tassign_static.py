#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile TASSIGN capacity regressions sequentially; never run generated NPU code.

Use --mode cpu for a CPU_SIM footprint check following CPU CMake's standard
policy: C++23 for GNU 14+, otherwise C++20. For --mode npu, supply a
CANN A5/A6 compiler target (including packed FP4 controls) with --compiler
and its device-only flags through repeated
--compiler-arg=VALUE options (including the CANN and C++ standard-library paths).
Use --allocation auto to enable both the compiler auto pass and __PTO_AUTO__.
The device suite checks exact SA diagnostics and rejects unrelated compiler errors.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from standalone_compiler import assess_compiler  # noqa: E402


# ID, name, required SA diagnostic, buffer/alignment overrides.
# Some compiler versions also report the dependent address failure after SA-0352.
CASES = [
    (1, "ordinary-exact-end", (), ()),
    (2, "ordinary-over-end", ("SA-0353",), ()),
    (3, "ordinary-too-large", ("SA-0352",), ("PTO_UBUF_SIZE_BYTES=2016",)),
    (4, "default-nz-exact-end", (), ()),
    (5, "normal-exact-end", (), ()),
    (6, "rowplusone64-exact-end", (), ()),
    (7, "rowplusone64-logical-end", ("SA-0353",), ()),
    (8, "rowplusone64-too-large", ("SA-0352",), ("PTO_UBUF_SIZE_BYTES=8192",)),
    (9, "packed-baseline-exact-end", (), ()),
    (10, "packed-storage-only-end", ("SA-0353",), ()),
    (11, "packed-rowplusone64-exact-end", (), ()),
    (12, "packed-rowplusone64-over-end", ("SA-0353",), ()),
    (13, "legacy-rowplusone33-exact-end", (), ()),
    (14, "legacy-rowplusone33-over-end", ("SA-0353",), ()),
    (15, "legacy-packed-rowplusone33-exact-end", (), ()),
    (16, "legacy-packed-rowplusone33-over-end", ("SA-0353",), ()),
    (17, "dynamic-rowplusone33-exact-end", (), ()),
    (18, "dynamic-rowplusone33-legacy-end", ("SA-0353",), ()),
]

DIAGNOSTICS = {
    "SA-0351": "TASSIGN: memory space is not available on this architecture (capacity is 0). (Fix: FIX-A12)",
    "SA-0352": "TASSIGN: Tile storage size exceeds memory space capacity. "
    "Reduce tile dimensions or element size. (Fix: FIX-A12)",
    "SA-0353": "TASSIGN: addr + tile_size exceeds memory space capacity "
    "(out of bounds). Use a smaller address or reduce tile size. (Fix: FIX-A12)",
    "SA-0354": "TASSIGN: addr is not properly aligned for the target memory space. "
    "Addr must be a multiple of the alignment (e.g. 32 bytes). (Fix: FIX-A12)",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timeout_text(chunk):
    if isinstance(chunk, str):
        return chunk
    return (chunk or b"").decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--compiler-arg", action="append", default=[])
    parser.add_argument("--mode", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--allocation", choices=("manual", "auto"), default="manual")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[case[1] for case in CASES])
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "result.json"
    if report_path.exists():
        parser.error("output directory already contains result.json; use a fresh directory")
    if args.mode == "cpu" and args.case:
        parser.error("--case only applies to the NPU diagnostic suite")
    source = root / "tests/compile/tassign_static.cpp"
    cases = [(0, "cpu-footprints", (), ())] if args.mode == "cpu" else CASES
    if args.case:
        cases = [case for case in cases if case[1] in args.case]
    base = [args.compiler, *args.compiler_arg, "-I" + str(root / "include")]
    if args.allocation == "auto":
        base += ["-D__PTO_AUTO__"]
        if args.mode == "npu":
            base += ["--cce-pto-auto-enable"]
    report = {
        "passed": False,
        "mode": args.mode,
        "allocation": args.allocation,
        "npu_execution": False,
        "source_sha256": sha256(source),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "checks": [],
    }
    if args.mode == "cpu":
        compiler = assess_compiler(args.compiler, output, root, timeout=args.timeout, compiler_args=args.compiler_arg)
        report["compiler"] = compiler
        if compiler["status"] != "supported":
            report["reason"] = compiler["reason"]
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[FAIL] {compiler['reason']}")
            return 1
        base += [compiler["standard_flag"], "-D__CPU_SIM"]
    for case_id, name, expected, overrides in cases:
        obj = output / (name + ".o")
        log = output / (name + ".log")
        command = base + ["-DPTO_TASSIGN_TEST_CASE=" + str(case_id)]
        command += ["-D" + definition for definition in overrides]
        command += ["-c", str(source), "-o", str(obj)]
        timed_out = False
        try:
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, check=False, timeout=args.timeout
            )
            text = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            text = timeout_text(exc.stdout) + timeout_text(exc.stderr)
            exit_code = None
            timed_out = True
        log.write_text(text)
        errors = [line for line in text.splitlines() if re.search(r"\b(?:fatal )?error:", line)]
        found = sorted(set(re.findall(r"\[(SA-\d+)\]", "\n".join(errors))))
        allowed = set(expected)
        if "SA-0352" in allowed:
            allowed.add("SA-0353")
        unrelated = [line for line in errors if not any(f"[{code}] {DIAGNOSTICS[code]}" in line for code in allowed)]
        passed = not timed_out and set(expected).issubset(found) and set(found).issubset(allowed)
        passed = passed and (exit_code != 0 and bool(errors) and not unrelated if expected else exit_code == 0)
        record = {
            "name": name,
            "argv": command,
            "cwd": str(root),
            "exit": exit_code,
            "expected_exit": "nonzero" if expected else 0,
            "expected_diagnostics": list(expected),
            "allowed_diagnostics": sorted(allowed),
            "actual_diagnostics": found,
            "unrelated_errors": unrelated,
            "timed_out": timed_out,
            "log_path": str(log),
            "log_sha256": sha256(log),
            "passed": passed,
        }
        if obj.exists():
            record.update(object_path=str(obj), object_sha256=sha256(obj))
        report["checks"].append(record)
        report["passed"] = all(check["passed"] for check in report["checks"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {'PASS' if passed else 'FAIL'} exit={exit_code} diagnostics={found}", flush=True)
    print(json.dumps({"passed": report["passed"], "checks": len(cases), "result": str(report_path)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
