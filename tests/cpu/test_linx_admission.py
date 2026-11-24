#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check restricted admission, separately from numerical instruction pass counts.

Every previously failing native profile must now be explicitly blocked; every
admitted profile must pass execution, golden, canaries and model diagnostics.
Retest all rejected profiles at -O0/-O2 with NDEBUG and public/direct includes.
No source/header bypass macro exists. This command never reclassifies a blocked
instruction as a numerical pass.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--restricted", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() or args.jobs < 1:
        parser.error("use a fresh output directory and positive jobs")
    args.output_dir.mkdir(parents=True)
    baseline = json.loads(args.baseline.read_text())
    restricted = json.loads(args.restricted.read_text())
    before = {c["name"]: c for c in baseline["cases"]}
    rows = []
    tasks = []
    for case in restricted["cases"]:
        native = case["routes"]["device"]
        previous = before[case["name"]]["routes"]["device"]
        allowed = native["status"] == "pass"
        blocked = native["status"] == "blocked"
        # Admission must not expose any known failure. Conservative restrictions
        # on additional unvalidated profiles are reported without fabricating a defect.
        profile_matches = (
            case["instructions"] == before[case["name"]]["instructions"]
            and case["golden_sha256"] == before[case["name"]]["golden_sha256"]
        )
        passed = profile_matches and ((allowed and previous["passed"]) or blocked)
        row = {
            "case": case["name"],
            "instructions": case["instructions"],
            "baseline_status": previous["status"],
            "status": native["status"],
            "admission_passed": passed,
            "profile_matches": profile_matches,
            "numerically_passed": allowed,
        }
        rows.append(row)
        if blocked:
            command = native["build"]["command"]
            compiler = command[0]
            include = command[command.index("-I") + 1]
            source = Path(native["build"]["log"]).parent / "device.cpp"
            for opt in ("-O0", "-O2"):
                for direct in (False, True):
                    tasks.append((case["name"], compiler, include, source, opt, direct))

    def check(task):
        name, compiler, include, source, opt, direct = task
        stem = name + "-" + opt[1:] + ("-direct" if direct else "-public")
        copied = args.output_dir / (stem + ".cpp")
        text = source.read_text()
        if direct:
            text = text.replace("<pto/pto-inst.hpp>", "<pto/npu/linx/impl/common/pto_tileop.hpp>")
        copied.write_text(text)
        command = [
            compiler,
            "-std=c++20",
            "-mlxbc",
            "-fenable-matrix",
            opt,
            "-DNDEBUG",
            "-mllvm",
            "-enable-all-vector-as-tilereg=true",
            "-I",
            include,
            "-c",
            str(copied),
            "-o",
            str(copied.with_suffix(".o")),
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
            text = proc.stdout + proc.stderr
            code = proc.returncode
        except subprocess.TimeoutExpired as error:
            text = str(error)
            code = None
        log = args.output_dir / (stem + ".log")
        log.write_text(text)
        compiler_crash = any(
            marker in text
            for marker in ("PLEASE submit a bug report", "LLVM ERROR:", "frontend command failed", "Stack dump:")
        )
        passed = code == 1 and "PTO Linx experimental:" in text and not compiler_crash
        return {
            "case": name,
            "optimization": opt,
            "ndebug": True,
            "direct_include": direct,
            "command": command,
            "returncode": code,
            "compiler_crash": compiler_crash,
            "log": str(log),
            "passed": passed,
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        checks = list(pool.map(check, tasks))
    result = {
        "purpose": "admission only, never numerical success",
        "baseline": str(args.baseline.resolve()),
        "restricted": str(args.restricted.resolve()),
        "device_header_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for include in sorted({t[2] for t in tasks})
            for path in sorted((Path(include) / "pto/npu/linx").rglob("*.hpp"))
        },
        "profiles": rows,
        "checks": checks,
        "admitted_profiles": sum(r["numerically_passed"] for r in rows),
        "blocked_profiles": sum(r["status"] == "blocked" for r in rows),
        "unexpected_profiles": [r for r in rows if not r["admission_passed"]],
        "passed": len(before) == len(rows)
        and all(r["admission_passed"] for r in rows)
        and all(c["passed"] for c in checks),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("profiles", "checks")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
