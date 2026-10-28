#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run focused tile_compact_storage CPU regressions.

Registered with tests/run_cpu.py and tests/script/all_cpu_tests.py. Select with
run_cpu.py --testcase tile_compact_storage, or invoke this driver directly
with a GCC/Clang-style --compiler and a fresh --output-dir to retain
exact commands and exits. Like CPU CMake, use C++23 for GNU 14+ and C++20
otherwise; GNU 14+ needs these overloads in existing half arithmetic.
Compiler support is assessed independently of the fixture. Only aggregate
invocations may skip a positively identified unsupported driver. This suite
does not exercise BF16.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from standalone_compiler import assess_compiler  # noqa: E402

try:
    import resource
except ImportError:  # Windows has no core-dump resource limit.
    resource = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-context", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        parser.error("output directory already contains result.json; use a fresh directory")
    source = root / "tests/cpu/tile_compact_storage.cpp"
    records = []
    compiler = assess_compiler(args.compiler, output, root)

    def write_result(status, reason=None):
        result = {
            "status": status,
            "passed": status == "passed",
            "compiler": compiler,
            "checks": records,
            "check_count": len(records),
            "output": str(output),
        }
        if reason is not None:
            result["reason"] = reason
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    if compiler["status"] != "supported":
        skipped = compiler["status"] == "unsupported" and args.aggregate_context
        reason = f"{args.compiler} ({compiler['family']}/{compiler['driver']}): {compiler['reason']}"
        result = write_result("skipped" if skipped else "failed", reason)
        print(("[SKIP] " if skipped else "[FAIL] ") + reason)
        print(json.dumps(result, indent=2))
        return 0 if skipped else 2
    base = [args.compiler, compiler["standard_flag"], "-D__CPU_SIM", "-I" + str(root / "include")]
    write_result("running")

    def command(name, argv, expected=0, needle=None):
        timed_out = False
        launch_error = None
        try:
            result = subprocess.run(
                argv,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=120,
            )
            text = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:

            def as_text(chunk):
                return chunk if isinstance(chunk, str) else (chunk or b"").decode(errors="replace")

            text = as_text(exc.stdout) + as_text(exc.stderr)
            exit_code = None
            timed_out = True
        except OSError as exc:
            text = f"{type(exc).__name__}: {exc}\n"
            exit_code = None
            launch_error = text.rstrip()
        (output / (name + ".log")).write_text(text, encoding="utf-8")
        passed = exit_code == expected if expected is not None else exit_code != 0
        passed = not timed_out and launch_error is None and passed and (needle is None or needle in text)
        records.append(
            {
                "name": name,
                "argv": list(map(str, argv)),
                "cwd": str(root),
                "exit": exit_code,
                "timed_out": timed_out,
                "launch_error": launch_error,
                "expected_exit": "nonzero" if expected is None else expected,
                "needle": needle,
                "log_path": str(output / (name + ".log")),
                "passed": passed,
            }
        )
        write_result("running" if passed else "failed")
        if not passed:
            raise RuntimeError(
                f"{name} failed: exit={exit_code}, timed_out={timed_out}; see {output / (name + '.log')}"
            )

    for mode in ("manual", "auto", "release-manual", "release-auto"):
        executable = output / ("tile-compact-" + mode + (".exe" if os.name == "nt" else ""))
        flags = ["-D__PTO_AUTO__"] if mode.endswith("auto") else []
        if mode.startswith("release-"):
            flags.append("-DNDEBUG")
        command(mode + "-compile", base + flags + [str(source), "-o", str(executable)])
        command(mode + "-run", [str(executable)])
        for suffix in ("region", "host", "legacy-region", "legacy-host", "dynamic-region", "dynamic-host"):
            command(
                mode + "-reject-" + suffix,
                [str(executable), "--invalid-" + suffix],
                None,
                "Tile assignment exceeds host storage capacity"
                if suffix.endswith("host")
                else "Tile assignment exceeds memory region capacity",
            )
    print(json.dumps(write_result("passed"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
