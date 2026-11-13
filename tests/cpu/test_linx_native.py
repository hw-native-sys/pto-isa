#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute the native Linx API through CPU-SIM, with optional device compilation.

The instruction kernels are identical in both builds. Host launchers and
assertions are guarded by __CPU_SIM. --upstream verifies the pinned native
import; --survey additionally compiles every upstream TileOP API fixture,
retaining native rejections separately from CPU parity results.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from standalone_compiler import assess_compiler  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-context", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--linx-compiler")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--survey", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.survey and not (args.upstream and args.linx_compiler):
        parser.error("--survey requires --upstream and --linx-compiler")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        parser.error("use a fresh output directory")
    fixtures = root / "tests/cpu/linx"
    compiler = assess_compiler(args.compiler, output, root)
    records = []
    survey = []

    def save(status, reason=None):
        result = {
            "status": status,
            "passed": status == "passed",
            "compiler": compiler,
            "checks": records,
            "check_count": len(records),
            "survey": survey,
            "provenance": json.loads((fixtures / "upstream.json").read_text()),
        }
        if reason:
            result["reason"] = reason
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    if compiler["status"] != "supported":
        skip = compiler["status"] == "unsupported" and args.aggregate_context
        save("skipped" if skip else "failed", compiler["reason"])
        print(compiler["reason"])
        return 0 if skip else 2
    save("running")

    def command(name, argv, expected=0, needle=None, record=True):
        argv = list(map(str, argv))
        launch_error = None
        timed_out = False
        try:
            run = subprocess.run(argv, cwd=root, capture_output=True, text=True, errors="replace", timeout=180)
            text, code = run.stdout + run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            text, code, timed_out = str(exc), None, True
        except OSError as exc:
            text, code, launch_error = str(exc), None, str(exc)
        log = output / (name + ".log")
        log.write_text(text)
        passed = code == expected if expected is not None else code is not None and code != 0
        passed = passed and not timed_out and launch_error is None and (needle is None or needle in text)
        if needle == "PTO Linx experimental:":
            passed = (
                passed
                and code == 1
                and not any(
                    marker in text
                    for marker in ("PLEASE submit a bug report", "frontend command failed", "Stack dump:")
                )
            )
        result = {
            "name": name,
            "argv": argv,
            "cwd": str(root),
            "exit": code,
            "timed_out": timed_out,
            "launch_error": launch_error,
            "log_path": str(log),
            "passed": passed,
            "expected_exit": expected if expected is not None else "nonzero",
            "diagnostic": needle,
        }
        if record:
            records.append(result)
            print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)
        return result

    base = [
        args.compiler,
        compiler["standard_flag"],
        "-D__CPU_SIM",
        "-DPTO_CPU_SIM_TARGET_LINX",
        "-I" + str(root / "include"),
        "-ffp-contract=off",
        "-pthread",
    ]
    base += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else ["-O2"]
    if args.release:
        base.append("-DNDEBUG")
    native = [args.linx_compiler, "-std=c++20", "-mlxbc", "-O2", "-I" + str(root / "include")]
    sync = [sys.executable, root / "tests/script/sync_linx_headers.py"]
    if args.upstream:
        sync += ["--upstream", args.upstream.resolve()]
    command("header-parity", sync)

    def execute(source):
        binary = output / source.stem
        if command(source.stem + "-compile", base + [source, "-o", binary])["passed"]:
            command(source.stem + "-execute", [binary])
        if args.linx_compiler and source.stem != "numeric":
            syntax_flags = ["-DLINX_TEST_TSTORE_PART"] if source.stem == "storage" else []
            # These aggregate kernels deliberately include disabled profiles.
            # Per-instruction numerical execution/admission is audited separately
            # by test_linx_models.py --instruction-suite. A diagnostic here is
            # an admission check, never a numerical pass.
            restricted = source.stem in {"vector", "storage", "matrix", "transform", "indexed", "native_api"}
            command(
                source.stem + "-device-syntax",
                native + syntax_flags + ["-fsyntax-only", source],
                None if restricted else 0,
                "PTO Linx experimental:" if restricted else None,
            )
            command(
                source.stem + "-device-assembly",
                native + ["-S", source, "-o", output / (source.stem + ".s")],
                None if restricted else 0,
                "PTO Linx experimental:" if restricted else None,
            )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(execute, sorted(p for p in fixtures.glob("*.cpp") if p.stem != "negative")))
    negative = fixtures / "negative.cpp"
    diagnostics = {
        1: "retired",
        2: "Tile size",
        3: "dtype/count",
        4: "matching GM and CUBE dtypes",
        5: "B.FPATR TransA requires",
        6: "private",
        7: "INIT",
        8: "PEMask",
    }
    for case, diagnostic in diagnostics.items():
        flags = ["-fsyntax-only", f"-DLINX_NEGATIVE={case}", negative]
        command(f"reject-cpu-{case}", base + flags, None, diagnostic)
        if args.linx_compiler:
            command(f"reject-device-{case}", native + flags, None, diagnostic)
    binary = output / "negative-runtime"
    if command("negative-runtime-compile", base + [negative, "-o", binary])["passed"]:
        for case in (1, 2, 3, 4):
            command(f"reject-assembly-order-{case}", [binary, str(case)], None)
    # Target selection and both common-header-first inclusion orders.
    entry = output / "entry.cpp"
    entry.write_text(
        "#include <pto/common/type.hpp>\n#include <pto/common/pto_tile.hpp>\n#include <pto/pto-inst.hpp>\nstatic_assert(pto::CPU_SIM_TARGET == pto::CpuSimTarget::Linx);\n"
    )
    command("common-entry", base + ["-fsyntax-only", entry])
    for target in (
        "PTO_CPU_SIM_TARGET_A2A3",
        "PTO_CPU_SIM_TARGET_A5",
        "PTO_NPU_ARCH_A5",
        "__NPU_ARCH__=2201",
        "__COSTMODEL",
    ):
        command(
            "reject-mixed-" + target.split("=")[0],
            base + ["-D" + target, "-fsyntax-only", entry],
            None,
            "target" if target != "__COSTMODEL" else "cannot be combined",
        )
    command(
        "reject-device-flag-on-host",
        [
            args.compiler,
            compiler["standard_flag"],
            "-I" + str(root / "include"),
            "-DPTO_NPU_ARCH_LINX",
            "-fsyntax-only",
            entry,
        ],
        None,
        "native Linx compiler",
    )
    command(
        "reject-cpu-flag-without-sim",
        [
            args.compiler,
            compiler["standard_flag"],
            "-I" + str(root / "include"),
            "-DPTO_CPU_SIM_TARGET_LINX",
            "-fsyntax-only",
            entry,
        ],
        None,
        "require __CPU_SIM",
    )

    if args.survey:
        upstream = args.upstream.resolve()
        # The native driver prepends its bundled SDK as -I before user flags.
        # Mark that same directory as system so Clang de-duplicates the injected
        # -I, leaving the pinned checkout first. Verify the resolved header;
        # otherwise a baseline can silently compile an older SDK.
        resource = command("native-resource-dir", [args.linx_compiler, "-print-resource-dir"])
        if not resource["passed"]:
            save("failed", "cannot resolve the native compiler resource directory")
            return 1
        bundled = Path(Path(resource["log_path"]).read_text().strip()) / "include/tileop-api"
        upstream_flags = [
            args.linx_compiler,
            "-std=c++20",
            "-mlxbc",
            "-I" + str(upstream / "include"),
            "-isystem",
            bundled,
        ]
        probe = output / "native-source.cpp"
        probe.write_text("#include <common/pto_tileop.hpp>\n")
        resolved = command(
            "native-source-includes",
            upstream_flags + ["-H", "-fsyntax-only", probe],
            needle=str(upstream / "include/common/pto_tileop.hpp"),
        )
        if not resolved["passed"]:
            save("failed", "native baseline did not resolve the pinned source header")
            return 1
        transformed = output / "survey"
        fixtures_root = upstream / "test/tileop_api"
        shutil.copytree(fixtures_root, transformed)
        # Helper headers (notably ../data.hpp) also include native headers.
        # Relocate the complete fixture tree so none accidentally resolves to
        # the toolchain's bundled, older TileOP SDK.
        for path in transformed.rglob("*"):
            if path.suffix in {".cpp", ".hpp", ".h"}:
                path.write_text(
                    re.sub(
                        r'#include\s*["<](?:common|jcore)/[^">]+[">]', "#include <pto/pto-inst.hpp>", path.read_text()
                    )
                )

        def inspect(source):
            copy = transformed / source.relative_to(fixtures_root)
            includes = ["-I" + str(upstream / "test/tileop_api"), "-I" + str(upstream / "test/tileop_api/src")]
            orig = command(
                "upstream-" + source.stem, upstream_flags + ["-fsyntax-only", *includes, source], record=False
            )
            relocated_includes = ["-I" + str(transformed), "-I" + str(transformed / "src")]
            dev = command(
                "survey-device-" + source.stem, native + relocated_includes + ["-fsyntax-only", copy], record=False
            )
            host = command(
                "survey-cpu-" + source.stem,
                base + relocated_includes + ["-D__vec__=", "-fsyntax-only", copy],
                record=False,
            )
            passed = all(x["exit"] is not None and not x["timed_out"] for x in (orig, dev, host))
            restricted = dev["exit"] not in (0, None) and "PTO Linx experimental:" in Path(dev["log_path"]).read_text()
            passed = passed and (((orig["exit"] == 0) == (dev["exit"] == 0)) or restricted)
            passed = passed and (orig["exit"] != 0 or host["exit"] == 0)
            return {
                "file": source.name,
                "upstream": orig,
                "device": dev,
                "cpu": host,
                "passed": passed,
                "classification": (
                    "restricted-native"
                    if restricted and orig["exit"] == 0
                    else "native-valid"
                    if dev["exit"] == 0
                    else "upstream-rejected"
                ),
            }

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for item in pool.map(inspect, sorted((upstream / "test/tileop_api/src").glob("*.cpp"))):
                survey.append(item)
                print(
                    f"[{'PASS' if item['passed'] else 'FAIL'}] API parity {item['file']} ({item['classification']})",
                    flush=True,
                )
                save("running")
    passed = all(item["passed"] for item in records + survey)
    save("passed" if passed else "failed")
    print(f"{len(records)} checks, {len(survey)} upstream fixtures; result: {output / 'result.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
